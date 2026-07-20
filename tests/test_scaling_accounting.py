from dataclasses import replace

from wwgpt.config import ladder
from wwgpt.model import GPT
from wwgpt.scaling import selected_parameter_count


def _manual_counts(model: GPT):
    unique = sum(p.numel() for p in model.parameters() if p.requires_grad)
    token = model.wte.weight.numel()
    pos = model.wpe.weight.numel()
    head = model.lm_head.weight.numel()
    body = sum(p.numel() for p in model.blocks.parameters() if p.requires_grad) + sum(p.numel() for p in model.ln_f.parameters() if p.requires_grad)
    return unique, token, pos, head, body


def test_parameter_counting_every_level_default_tied_weights():
    for _level, cfg in ladder().items():
        model = GPT(cfg)
        rep = model.parameter_report()
        unique, token, pos, head, body = _manual_counts(model)
        assert rep.tied_embedding_head is True
        assert model.lm_head.weight is model.wte.weight
        assert rep.total_unique_trainable_parameters == unique
        assert rep.total_parameters == unique
        assert rep.trainable_parameters == unique
        assert rep.token_embedding_parameters == token
        assert rep.position_embedding_parameters == pos
        assert rep.output_head_parameters == head
        assert rep.embedding_parameters == token + pos
        assert rep.transformer_body_parameters == body
        assert rep.non_embedding_parameters == body
        assert rep.non_position_parameters == unique - pos
        assert selected_parameter_count(rep, "transformer_body") == body


def test_parameter_counting_untied_head_counts_head_once():
    cfg = replace(ladder()[0], tie_weights=False)
    model = GPT(cfg)
    rep = model.parameter_report()
    unique, token, pos, head, body = _manual_counts(model)
    assert rep.tied_embedding_head is False
    assert model.lm_head.weight is not model.wte.weight
    assert rep.total_unique_trainable_parameters == unique
    assert rep.total_parameters == token + pos + body + head
    assert rep.non_embedding_parameters == body
    assert rep.output_head_parameters == head
    assert selected_parameter_count(rep, "total") == unique
    assert selected_parameter_count(rep, "non_embedding") == body
