from pathlib import Path
import json, subprocess, sys
import nbformat, pytest, torch, pandas as pd
from wwgpt.strength_scan import parse_strengths, format_strength_label, target_alpha_to_q, strength_config, run_strength_scan, validate_scan_pairing
from wwgpt.config import ExperimentConfig
from wwgpt.ww import external_projected_layer_names
from wwgpt.model import GPT
from wwgpt.config import ModelConfig
from wwgpt.strength_scan_analysis import analyze_strength_scan, resolve_scan_root

def test_strength_parse_labels_validation():
    assert parse_strengths('0.02,0.1,0.25,0.5,1.0') == [0.02,0.1,0.25,0.5,1.0]
    assert [format_strength_label(x) for x in [0.02,0.1,0.25,0.5,1.0]] == ['strength_0p02','strength_0p1','strength_0p25','strength_0p5','strength_1p0']
    for bad in ['nan','inf','-0.1','1.1','0.1,0.10']:
        with pytest.raises(ValueError): parse_strengths(bad)

def test_strength_config_immutable_and_q():
    cfg=ExperimentConfig(); new=strength_config(cfg,0.5)
    assert cfg.wwpgd.strength == 1.0 and new.wwpgd.strength == 0.5 and new is not cfg
    assert target_alpha_to_q(2.0)==1.0
    with pytest.raises(ValueError): target_alpha_to_q(1.0)

def test_strength_scan_uses_external_layer_selector():
    m=GPT(ModelConfig(n_layer=1,n_head=1,n_embd=64,block_size=4,vocab_size=10))
    assert external_projected_layer_names(m) == [
        'blocks.0.attn.key', 'blocks.0.attn.query', 'blocks.0.attn.value',
        'blocks.0.attn.proj', 'blocks.0.mlp.0', 'blocks.0.mlp.2'
    ]

def test_cli_help():
    assert subprocess.run([sys.executable,'-m','wwgpt.cli','run-strength-scan','--help'],capture_output=True,text=True).returncode==0
    assert subprocess.run([sys.executable,'-m','wwgpt.cli','analyze-strength-scan','--help'],capture_output=True,text=True).returncode==0

def test_tiny_strength_scan(tmp_path):
    with pytest.raises(RuntimeError, match='never fall back to fixtures'):
        run_strength_scan(0,tmp_path/'data',tmp_path/'results',1,seeds=[1],strengths='0.02,0.1',device='cpu',eval_interval=1,spectral_interval=99,checkpoint_interval=99,resume=False)

def test_notebooks_parse_strength():
    for p in ['notebooks/07_strength_scan_overview.ipynb','notebooks/08_strength_scan_weightwatcher.ipynb']:
        nbformat.read(p, as_version=4)
