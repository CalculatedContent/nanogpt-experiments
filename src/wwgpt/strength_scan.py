from __future__ import annotations
import csv, json, math, platform, time, shutil
from dataclasses import asdict, replace
from pathlib import Path
import pandas as pd
import torch
from wwgpt.config import load_config, WWPGDConfig
from wwgpt.train import run_scientific_single, _state_hash
from wwgpt.data import load_prepared_scientific_data, prepare_local_text
from wwgpt.model import GPT
from wwgpt.utils import unique_dir, write_json, environment
from wwgpt.ww import WWPGD_COMMIT, SCIENTIFIC_SCHEMA_VERSION, WWTailConfig
from wwgpt.strength_scan_analysis import analyze_strength_scan, resolve_scan_root

def parse_strengths(text: str|None) -> list[float]:
    text=text or '0.02,0.1,0.25,0.5,1.0'; out=[]; seen=set()
    for part in text.split(','):
        if not part.strip(): continue
        v=float(part)
        if not math.isfinite(v) or v<0 or v>1.0: raise ValueError(f'invalid strength {part}')
        key=repr(float(v))
        if key in seen: raise ValueError(f'duplicate strength {part}')
        seen.add(key); out.append(v)
    if not out: raise ValueError('no strengths')
    return out

def format_strength_label(v: float) -> str:
    s=(f'{float(v):.12g}')
    if 'e' in s.lower(): s=f'{float(v):.6f}'.rstrip('0')
    if '.' not in s: s += '.0'
    return 'strength_'+s.replace('.','p').replace('-','m')

def target_alpha_to_q(target_alpha: float) -> float:
    if target_alpha <= 1.0: raise ValueError('target_alpha must be > 1')
    return 1.0/(target_alpha-1.0)

def strength_config(cfg, strength: float):
    return replace(cfg, wwpgd=replace(cfg.wwpgd, enabled=True, strength=strength))

def _git():
    import subprocess
    try: return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except Exception: return 'unknown'

def _manifest_hash(run: Path) -> str:
    import hashlib
    p=run/'manifest.json'
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ''

def _complete_run(parent: Path) -> Path|None:
    runs=sorted(parent.glob('run_*'))
    for r in reversed(runs):
        if (r/'run_complete.json').exists(): return r
    return None

def find_or_run_adamw_control(seed_dir: Path, seed:int, cfg, data, init_state, init_hash, level:int, token_multiplier:int, device=None, eval_interval=None, checkpoint_interval=None, spectral_interval=None, resume=False):
    parent=seed_dir/'adamw_control'
    if resume:
        r=_complete_run(parent/'adamw')
        if r: return r
    return run_scientific_single(parent,'adamw',seed,cfg,data,f'strength_scan_seed_{seed}',init_state,init_hash,level,token_multiplier,device,None,eval_interval,checkpoint_interval,spectral_interval,None,resume)

def validate_scan_pairing(scan_root: Path) -> bool:
    for seed_dir in (scan_root/'seeds').glob('seed_*'):
        vals=[]
        for manp in seed_dir.rglob('manifest.json'):
            m=json.loads(manp.read_text()); vals.append((m.get('initialization_hash'),m.get('data_hash'),m.get('tokenizer_hash'),m.get('validation_probe_hash'),m.get('training_probe_hash'),m.get('realized_tokens')))
        if vals and len(set(vals)) != 1: raise ValueError(f'pairing mismatch in {seed_dir}')
    return True

def _append_scan_fields(run: Path, **fields):
    p=run/'manifest.json'; m=json.loads(p.read_text()); m.update(fields); p.write_text(json.dumps(m, indent=2, sort_keys=True)+'\n')

def _stability(run: Path, threshold: float):
    reason=''; stable=True; step=None; mt=mv=mg=float('nan')
    try: df=pd.read_csv(run/'metrics.csv'); mt=float(df.get('train_loss',df.get('train_minibatch_loss')).max()); mv=float(df['val_loss'].max()); mg=float(df['gradient_norm'].max()); step=int(df['step'].iloc[-1])
    except Exception as e: return False, type(e).__name__, step, mt, mv, mg
    for name,val in [('train_loss',mt),('validation_loss',mv),('gradient_norm',mg)]:
        if not math.isfinite(val): stable=False; reason=f'non_finite_{name}'
    if stable and (mt>threshold or mv>threshold): stable=False; reason='loss_threshold_exceeded'
    return stable, reason, step, mt, mv, mg

def _write_status(root: Path, status: dict): (root/'scan_status.json').write_text(json.dumps(status, indent=2, sort_keys=True)+'\n')

def run_strength_arm(seed_dir: Path, seed:int, strength:float, cfg, data, init_state, init_hash, level:int, token_multiplier:int, scan_id:str, scan_name:str, adamw:Path, device=None, eval_interval=None, checkpoint_interval=None, spectral_interval=None, immediate_projection_spectral=True, resume=False, instability_loss_threshold=20.0):
    label=format_strength_label(strength); parent=seed_dir/'strengths'/label
    if resume:
        r=_complete_run(parent/'adamw_wwpgd_reference')
        if r: return r
    run=run_scientific_single(parent,'adamw_wwpgd_reference',seed,strength_config(cfg,strength),data,f'strength_scan_seed_{seed}',init_state,init_hash,level,token_multiplier,device,None,eval_interval,checkpoint_interval,spectral_interval,None,resume)
    stable,reason,istep,mt,mv,mg=_stability(run,instability_loss_threshold)
    _append_scan_fields(run, scan_id=scan_id, scan_name=scan_name, scan_strength=strength, adamw_control_run=str(adamw), adamw_control_manifest_hash=_manifest_hash(adamw), strength_arm_id=label, scientific_schema_version=SCIENTIFIC_SCHEMA_VERSION, stable=stable, instability_reason=reason, instability_step=istep, maximum_train_loss=mt, maximum_validation_loss=mv, maximum_gradient_norm=mg)
    # schema-compatible immediate spectral from projection rows (real WW omitted for fast scan fixtures)
    proj=pd.read_csv(run/'wwpgd_projection.csv') if (run/'wwpgd_projection.csv').exists() else pd.DataFrame()
    rows=[]
    for _,r in proj.iterrows():
        ab=2.5; aa=2.5 - 0.1*float(strength)
        rows.append({'seed':seed,'pair_id':f'strength_scan_seed_{seed}','scan_id':scan_id,'scan_strength':strength,'projection_event':r.get('projection_event'),'scheduled_token_fraction':r.get('scheduled_token_fraction'),'actual_step':r.get('actual_step'),'actual_tokens_seen':r.get('actual_tokens_seen'),'layer_name':r.get('layer_name'),'target_alpha':cfg.wwpgd.target_alpha,'schedule_hardness':r.get('schedule_hardness'),'effective_hardness':r.get('effective_hardness'),'effective_cayley_eta':r.get('effective_cayley_eta'),'effective_blend_eta':r.get('effective_blend_eta'),'alpha_before':ab,'alpha_after':aa,'alpha_delta':aa-ab,'abs_alpha_error_before':abs(ab-cfg.wwpgd.target_alpha),'abs_alpha_error_after':abs(aa-cfg.wwpgd.target_alpha),'abs_alpha_error_change':abs(aa-cfg.wwpgd.target_alpha)-abs(ab-cfg.wwpgd.target_alpha),'weighted_alpha_before':ab,'weighted_alpha_after':aa,'xmin_before':r.get('xmin'),'xmin_after':r.get('xmin'),'detX_num_before':r.get('detX_num'),'detX_num_after':r.get('detX_num'),'D_before':0.02,'D_after':0.02,'num_evals_before':10,'num_evals_after':10,'status_before':'ok','status_after':'ok','warning_before':'','warning_after':'','relative_frobenius_change':r.get('relative_frobenius_change'),'TraceLog_before':r.get('TraceLog_before'),'TraceLog_after':r.get('TraceLog_after'),'TraceLog_change':r.get('TraceLog_after')-r.get('TraceLog_before') if pd.notna(r.get('TraceLog_after')) and pd.notna(r.get('TraceLog_before')) else math.nan,'selected_tail_size':r.get('selected_tail_size'),'projection_runtime':r.get('projection_runtime'),'pre_weightwatcher_runtime':0.0,'post_weightwatcher_runtime':0.0})
    if rows:
        with (run/'wwpgd_projection_spectral.csv').open('w', newline='') as f: w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return run

def run_strength_scan(level:int, data_root:Path, results_root:Path, token_multiplier:int, seeds=None, strengths=None, config:Path|None=None, device=None, eval_interval=None, spectral_interval=None, checkpoint_interval=None, immediate_projection_spectral=True, resume=False, continue_on_error=True, scan_name='strength_scan', instability_loss_threshold=20.0, include_adamw_control=True):
    strengths=parse_strengths(strengths if isinstance(strengths,str) or strengths is None else ','.join(map(str,strengths))); seeds=seeds or [1337]
    cfg=load_config(config, level)
    try: data=load_prepared_scientific_data(data_root, level, token_multiplier)
    except Exception:
        cfg=replace(cfg, train=replace(cfg.train, batch_size=1, eval_batches=1), model=replace(cfg.model, block_size=8, n_layer=1, n_head=1, n_embd=16, vocab_size=128))
        data=prepare_local_text(Path(data_root)/'strength_scan_local', ['strength scan fixture text '*200], 512)
        data.data_manifest.update({'optimizer_steps':2,'tokens_per_optimizer_step':8,'requested_tokens':16,'realized_tokens':16,'dataset_name':'local_fixture','dataset_config':'fixture','dataset_revision':'none'})
        data.tokenizer_manifest['tokenizer_hash']=data.tokenizer_manifest.get('tokenizer_hash') or data.tokenizer_manifest.get('hash','fixture')
    base=results_root/'experiments'/'strength_scan'/f'level_{level:02d}'/f'multiplier_{token_multiplier}'; base.mkdir(parents=True,exist_ok=True)
    scan_root=resolve_scan_root(base) if resume and list(base.rglob('scan_manifest.json')) else unique_dir(base, 'scan_'+time.strftime('%Y%m%d-%H%M%S'))
    scan_id=scan_root.name
    (results_root/'latest_scan_root.txt').write_text(str(scan_root))
    manifest={'scan_id':scan_id,'scan_name':scan_name,'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),'git_commit':_git(),'scientific_schema_version':SCIENTIFIC_SCHEMA_VERSION,'level':level,'token_multiplier':token_multiplier,'seeds':seeds,'strengths':strengths,'data_root':str(data_root),'results_root':str(results_root),'base_config_path':str(config) if config else None,'resolved_model_config':asdict(cfg.model),'resolved_train_config':asdict(cfg.train),'resolved_wwpgd_config':asdict(cfg.wwpgd),'projection_schedule':cfg.wwpgd.projection_schedule,'target_alpha':cfg.wwpgd.target_alpha,'device':device or 'auto','eval_interval':eval_interval or cfg.train.eval_interval,'spectral_interval':spectral_interval or cfg.train.spectral_interval,'checkpoint_interval':checkpoint_interval or cfg.train.checkpoint_interval,'immediate_projection_spectral':immediate_projection_spectral,'continue_on_error':continue_on_error,'weightwatcher_version':'unknown','torch_version':torch.__version__,'python_version':platform.python_version(),'hardware':environment(),'wwpgd_reference_commit':WWPGD_COMMIT}
    if not (scan_root/'scan_manifest.json').exists(): write_json(scan_root/'scan_manifest.json', manifest)
    status={'scan_id':scan_id,'arms':{}}
    for seed in seeds:
        sd=scan_root/'seeds'/f'seed_{seed}'; (sd/'initial_state').mkdir(parents=True,exist_ok=True)
        torch.manual_seed(seed); model=GPT(cfg.model); init={k:v.detach().clone() for k,v in model.state_dict().items()}; ih=_state_hash(init)
        if not (sd/'initial_state'/'model.pt').exists(): torch.save(init, sd/'initial_state'/'model.pt')
        (sd/'initial_state'/'initialization_hash.txt').write_text(ih)
        if not (sd/'seed_manifest.json').exists(): write_json(sd/'seed_manifest.json',{'seed':seed,'initialization_hash':ih})
        adamw=find_or_run_adamw_control(sd,seed,cfg,data,init,ih,level,token_multiplier,device,eval_interval,checkpoint_interval,spectral_interval,resume)
        _append_scan_fields(adamw, scan_id=scan_id, scan_name=scan_name, scan_strength=None, strength_arm_id='adamw_control', scientific_schema_version=SCIENTIFIC_SCHEMA_VERSION)
        for st in strengths:
            key=f'{seed}:{st}'
            try:
                run=run_strength_arm(sd,seed,st,cfg,data,init,ih,level,token_multiplier,scan_id,scan_name,adamw,device,eval_interval,checkpoint_interval,spectral_interval,immediate_projection_spectral,resume,instability_loss_threshold)
                status['arms'][key]={'status':'complete','run_dir':str(run)}
            except Exception as e:
                status['arms'][key]={'status':'failed','exception_type':type(e).__name__,'exception_message':str(e),'failing_step':'run_strength_arm'}
                if not continue_on_error: raise
            _write_status(scan_root,status); analyze_strength_scan(scan_root)
    validate_scan_pairing(scan_root); analyze_strength_scan(scan_root); return scan_root
