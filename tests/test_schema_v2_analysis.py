
from pathlib import Path
import nbformat
import pandas as pd
from nbclient import NotebookClient
from wwgpt.analysis import *
FIX=Path('tests/fixtures/schema_v2_results/experiments/level_00/multiplier_20').resolve()

def test_optimizer_normalization():
    assert normalize_optimizer('adamw')['optimizer_family']=='adamw'
    assert normalize_optimizer('adamw_wwpgd_reference')['optimizer_family']=='wwpgd'
    assert not normalize_optimizer('adamw_wwpgd')['allowed_by_default']

def test_canonical_selection_duplicate_incomplete_excluded():
    c=discover_pair_candidates(FIX); selected,audit=select_canonical_pairs(c)
    assert sorted(x.seed for x in selected)==[1337,2027,4099,7919,104729]
    assert len(selected)==5
    assert (audit.query("seed==1337 and status=='excluded'")['exclusion_reason'].str.len()>0).any()

def test_reference_selection_and_legacy_exclusion():
    runs=discover_canonical_runs(FIX)
    assert {r['optimizer_raw'] for r in runs}=={'adamw','adamw_wwpgd_reference'}
    assert len(runs)==10

def test_hash_matching_and_mismatch_rejection(tmp_path):
    import shutil,json
    dst=tmp_path/'r'; shutil.copytree(FIX,dst)
    man=next(dst.glob('pair_2027*/adamw_wwpgd_reference/run_*/manifest.json'))
    d=json.loads(man.read_text()); d['tokenizer_hash']='bad'; man.write_text(json.dumps(d))
    selected,audit=select_canonical_pairs(discover_pair_candidates(dst))
    assert 2027 not in [x.seed for x in selected]
    assert audit.query('seed==2027')['exclusion_reason'].str.contains('tokenizer_hash mismatch').any()

def test_normalizers():
    m=normalize_metrics(pd.DataFrame({'tokens_processed':[1],'val_loss':[2],'elapsed_time':[3],'projection_overhead':[4]}))
    assert {'tokens_seen','validation_loss','elapsed_seconds','projection_seconds'}.issubset(m.columns)
    p=normalize_projection_records(pd.DataFrame({'actual_step':[1],'actual_tokens_seen':[2],'TraceLog_before':[3.0],'TraceLog_after':[4.0]}))
    assert {'step','tokens_seen','trace_log_before','trace_log_after','trace_log_delta'}.issubset(p.columns)

def test_spectral_terminal_alignment_t_summary_scaling():
    runs=discover_canonical_runs(FIX); art=load_run_artifacts(Path(runs[1]['run_dir']))
    assert art['spectral']['valid_weightwatcher'].all()
    term=terminal_results(runs); assert len(term)==5 and 'wwpgd_minus_adamw_validation_loss' in term
    a=load_run_artifacts(Path(runs[0]['run_dir']))['metrics']; w=load_run_artifacts(Path(runs[1]['run_dir']))['metrics']
    g,d=paired_curve_differences([(a,w)],'tokens_seen','validation_loss'); assert d.shape[0]==1
    assert student_t_summary([1,2,3])['ci_high']>2
    inv=discover_scaling_runs(FIX.parents[3]); design=scaling_design_points(inv); ready=scaling_readiness(design)
    assert not bool(ready['ready'].iloc[0])

def test_notebooks_parse_and_execute_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv('WWGPT_RESULTS_ROOT', str(FIX)); monkeypatch.setenv('WWGPT_SCALING_ROOT', str(FIX.parents[3])); monkeypatch.setenv('WWGPT_ALLOW_TEST_FIXTURE','1')
    for nbp in sorted(Path('notebooks').glob('0*.ipynb')):
        nb=nbformat.read(nbp, as_version=4)
        NotebookClient(nb, timeout=120, kernel_name='python3').execute(cwd=str(Path.cwd()))

def test_broader_results_root_env_resolves_to_multiplier_fixture(monkeypatch):
    run_root = FIX.parents[2]
    assert resolve_experiment_root(run_root) == FIX
