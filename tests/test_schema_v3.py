from dataclasses import replace

import torch

from wwgpt.config import TrainConfig, ladder, level_model_config, historical_level0_model_config, scaling_level0_model_config
from wwgpt.data import NonRepeatingTokenReader, RandomWindowTokenReader, random_probe, stable_seed
from wwgpt.model import GPT
from wwgpt.optim import build_optimizer_bundle, muon_parameter_names, apply_lr_schedule, resolve_warmup_steps
from wwgpt.train import NoExtension, WWPGDExtension
from wwgpt.ww import composite_matrices


def test_schema_v3_architecture_defaults_and_report():
    cfg0 = ladder()[0]
    m = GPT(replace(cfg0, vocab_size=32))
    assert (cfg0.n_layer, cfg0.n_head, cfg0.n_embd, cfg0.block_size) == (1, 1, 64, 256)
    for cfg in ladder().values():
        assert cfg.n_embd // cfg.n_head == 64
    b = m.blocks[0]
    for mod in [b.attn.key, b.attn.query, b.attn.value, b.attn.proj, b.mlp[0], b.mlp[2], m.lm_head]:
        assert mod.bias is None
    assert b.ln_1.bias is not None and b.ln_2.bias is not None and m.ln_f.bias is not None
    assert m.lm_head.weight.data_ptr() != m.wte.weight.data_ptr()
    rep = m.parameter_report()
    assert rep.output_head_parameters == m.lm_head.weight.numel()
    assert rep.embedding_parameters == m.wte.weight.numel() + m.wpe.weight.numel()


def _ids(bundle):
    return [id(p) for opt in bundle.optimizers for g in opt.param_groups for p in g["params"]]


def test_optimizer_partitions_and_scheduler_state_roundtrip():
    m = GPT(replace(ladder()[0], vocab_size=32))
    for name in ["adamw", "stableadamw", "muon"]:
        try:
            bundle, _ = build_optimizer_bundle(m, TrainConfig(), name)
        except RuntimeError:
            if name == "stableadamw":
                continue
            raise
        assert sorted(_ids(bundle)) == sorted(id(p) for p in m.parameters() if p.requires_grad)
        state = bundle.state_dict(); bundle.load_state_dict(state)
        rows = apply_lr_schedule(bundle, 0, 10, resolve_warmup_steps(10, .05, None), TrainConfig())
        assert rows
    mnames = muon_parameter_names(m)
    assert not any(n.startswith(("wte.", "wpe.", "lm_head.")) for n in mnames)


def test_extension_schedule_noop_and_due(monkeypatch):
    m = GPT(replace(ladder()[0], vocab_size=32))
    assert NoExtension().after_optimizer_step(model=m, optimizer_step=1, total_optimizer_steps=4, tokens_seen=1) == []
    monkeypatch.setattr("wwgpt.train.weightwatcher_details", lambda model: None)
    monkeypatch.setattr("wwgpt.train.apply_external_wwpgd", lambda *a, **k: [{"changed": True}])
    ext = WWPGDExtension(type("C", (), {"q":1.0,"target_alpha":2.0,"strength":0.1,"min_tail":5,"blend_eta":.5,"cayley_eta":.25,"use_detx":True,"warmup_events":0,"ramp_events":1})(), 2)
    assert ext.after_optimizer_step(model=m, optimizer_step=1, total_optimizer_steps=4, tokens_seen=1) == []
    assert ext.after_optimizer_step(model=m, optimizer_step=2, total_optimizer_steps=4, tokens_seen=2)


def test_random_probe_hashes_and_reader_position():
    tokens = list(range(1000)); reader = NonRepeatingTokenReader(tokens, 8); pos = reader.pos
    h1 = random_probe(tokens, 8, 2, 2, stable_seed(1, "train", 0, "random_per_eval_v1"))[2]
    h2 = random_probe(tokens, 8, 2, 2, stable_seed(1, "train", 1, "random_per_eval_v1"))[2]
    assert h1 != h2
    assert reader.pos == pos


def test_composite_formulas_and_rng_isolation():
    torch.manual_seed(123); m = GPT(replace(ladder()[1], vocab_size=32)); state = torch.random.get_rng_state()
    comps = composite_matrices(m)
    assert torch.equal(state, torch.random.get_rng_state())
    b = m.blocks[0]; wk=b.attn.key.weight.detach().float().cpu(); wq=b.attn.query.weight.detach().float().cpu(); wv=b.attn.value.weight.detach().float().cpu(); wo=b.attn.proj.weight.detach().float().cpu(); wi=b.mlp[0].weight.detach().float().cpu(); wo2=b.mlp[2].weight.detach().float().cpu()
    assert torch.allclose(comps["L0000_KQ"][0], wk @ wq)
    assert torch.allclose(comps["L0000_QK"][0], wq @ wk)
    assert torch.allclose(comps["L0000_QK_effective"][0], wq.T @ wk)
    assert torch.allclose(comps["L0000_KQ_effective"][0], wk.T @ wq)
    assert torch.allclose(comps["L0000_VO"][0], wv @ wo)
    assert torch.allclose(comps["L0000_MLP_IO"][0], wo2 @ wi)
    ov = sum(comps[f"L0000_H{h:03d}_OV"][0] for h in range(b.attn.n_head))
    assert torch.allclose(comps["L0000_OV"][0], ov)


def test_level0_profiles_are_explicit():
    scaling = scaling_level0_model_config()
    assert level_model_config(0) == scaling
    hist = historical_level0_model_config()
    assert (hist.n_layer, hist.n_head, hist.n_embd, hist.block_size) == (1, 1, 64, 64)
    assert hist.init_mode == "pytorch_default"
    assert (scaling.n_layer, scaling.n_head, scaling.n_embd, scaling.block_size) == (1, 1, 64, 256)
    assert scaling.init_mode == "nanogpt_normal_0p02"


def test_random_window_reader_state_and_pair_sharing():
    tokens = list(range(1000))
    a = RandomWindowTokenReader(tokens, 8, stable_seed(7, "pair", "train_reader_v1"))
    b = RandomWindowTokenReader(tokens, 8, stable_seed(7, "pair", "train_reader_v1"))
    ax, ay = a.next_batch(4); bx, by = b.next_batch(4)
    assert (ax == bx).all() and (ay == by).all()
    state = a.state_dict()
    nxt = a.next_batch(4)
    c = RandomWindowTokenReader(tokens, 8, 123)
    c.load_state_dict(state)
    cx, cy = c.next_batch(4)
    assert (nxt[0] == cx).all() and (nxt[1] == cy).all()


def test_eval_hashes_change_and_pair_agree_without_reader_rng_change():
    tokens = list(range(1000))
    reader = RandomWindowTokenReader(tokens, 8, 99)
    state = reader.state_dict()
    h0 = random_probe(tokens, 8, 2, 2, stable_seed(1, "val", 0, "random_per_eval_v1"))[2]
    h0b = random_probe(tokens, 8, 2, 2, stable_seed(1, "val", 0, "random_per_eval_v1"))[2]
    h1 = random_probe(tokens, 8, 2, 2, stable_seed(1, "val", 1, "random_per_eval_v1"))[2]
    assert h0 == h0b and h0 != h1
    assert reader.state_dict() == state
