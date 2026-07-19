from pathlib import Path
import ast, pandas as pd, torch
from wwgpt.ww import measured_projection_spectral_rows
from wwgpt.checkpointing import save_checkpoint, load_latest_checkpoint, validate_resume, complete_test_checkpoint_state
from wwgpt.device import detect_device
from wwgpt.integrity import audit_experiment

def test_no_fabricated_constants_in_production_source():
    bad=['alpha_before = 2.5','2.5 - 0.1','D_before = 0.02','D_after = 0.02','num_evals_before = 10','num_evals_after = 10','real WW omitted']
    for p in Path('src').rglob('*.py'):
        txt=p.read_text()
        for b in bad:
            assert b not in txt, (p,b)

def test_production_strength_scan_does_not_import_prepare_local_text():
    tree=ast.parse(Path('src/wwgpt/strength_scan.py').read_text())
    names=[]
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom): names += [a.name for a in n.names]
        if isinstance(n, ast.Name): names.append(n.id)
    assert 'prepare_local_text' not in names

def test_missing_weightwatcher_outputs_remain_nan_and_invalid():
    pre=pd.DataFrame([{'longname':'blocks.0.attn.c_attn','alpha':2.2,'xmin':1,'status':'ok'}])
    post=pd.DataFrame([{'longname':'blocks.0.attn.c_attn','xmin':1,'status':'ok'}])
    rows=measured_projection_spectral_rows(pre,post,[{'layer_name':'blocks.0.attn.c_attn','projection_event':0}],2.5)
    assert pd.isna(rows[0]['alpha_after'])
    assert rows[0]['measurement_valid_for_science'] is False
    assert pd.isna(rows[0]['alpha_delta'])

def test_checkpoint_atomic_latest_and_restore(tmp_path):
    state=complete_test_checkpoint_state(current_step=3, next_step=4, step=3, tokens_processed=96, training_reader_position=96, reader_position=96, model_state_dict={'w':torch.tensor([1])}, optimizer_state_dict={'x':1}, compatibility={'configuration_hash':'c','data_hash':'d','tokenizer_hash':'t','initialization_hash':'i','scientific_schema_version':2,'model_configuration_hash':'m','training_configuration_hash':'tr','wwpgd_configuration_hash':'w'})
    (tmp_path/'manifest.json').write_text('{"configuration_hash":"c","data_hash":"d","tokenizer_hash":"t","initialization_hash":"i","scientific_schema_version":2,"model_configuration_hash":"m","training_configuration_hash":"tr","wwpgd_configuration_hash":"w"}')
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'data_manifest.json').write_text('{"corpus_hash":"d"}')
    (tmp_path/'tokenizer_manifest.json').write_text('{"tokenizer_hash":"t"}')
    (tmp_path/'initialization_hash.txt').write_text('i')
    save_checkpoint(tmp_path,state)
    loaded=load_latest_checkpoint(tmp_path)
    assert loaded['step']==3 and loaded['reader_position']==96 and loaded['optimizer_state_dict']['x']==1
    assert validate_resume(tmp_path)['next_step']==4

def test_cpu_fallback_explicit():
    assert str(detect_device('cpu'))=='cpu'

def test_integrity_rejects_legacy_fabricated_scan(tmp_path):
    run=tmp_path/'run_legacy'; run.mkdir(parents=True)
    (run/'manifest.json').write_text('{"valid_for_science":true}')
    (run/'run_complete.json').write_text('{}')
    pd.DataFrame([{'alpha_before':2.5,'alpha_after':2.4}]).to_csv(run/'wwpgd_projection_spectral.csv', index=False)
    summary=audit_experiment(tmp_path)
    assert 'legacy_or_missing_measured_provenance_fields' in (tmp_path/'analysis'/'integrity_audit.csv').read_text()
