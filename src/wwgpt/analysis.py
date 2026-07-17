
from __future__ import annotations

import json, math, re, inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

OPTIMIZER_LABELS={"adamw":"AdamW","wwpgd":"AdamW + WW-PGD"}
OPTIMIZER_COLORS={"adamw":"#1f77b4","wwpgd":"#ff7f0e"}
PROJECTED_LAYER_PATTERNS=(".attn.c_attn",".attn.c_proj",".mlp.0",".mlp.2")
HASH_FIELDS=("seed","initialization_hash","data_hash","corpus_hash","tokenizer_hash","validation_probe_hash","training_probe_hash","realized_tokens")

@dataclass(frozen=True)
class RunRecord:
    pair_id:str; pair_dir:Path; run_dir:Path; optimizer_raw:str; optimizer_family:str; optimizer_label:str; seed:int|None; manifest:dict[str,Any]; complete:bool; legacy:bool; valid_for_science:bool

@dataclass(frozen=True)
class PairCandidate:
    pair_id:str; pair_dir:Path; seed:int|None; runs:dict[str,RunRecord]; valid:bool; exclusion_reason:str; newest_mtime:float

def read_json_file(path:Path)->dict[str,Any]:
    return json.loads(path.read_text()) if path.exists() else {}

def load_csv_file(path:Path)->pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def resolve_experiment_root(path: str | Path) -> Path:
    """Resolve a user/papermill result root to a single-level experiment directory.

    The notebooks can be parameterized with either the exact experiment
    directory (containing ``pair_*`` directories) or a broader run root such as
    ``/tmp/wwpgd_v2/real_level0_results_v4``.  For the latter case, prefer the
    conventional schema-v2 single-level location
    ``experiments/level_00/multiplier_20`` when it exists.
    """
    root = Path(path).expanduser().resolve()
    if any(root.glob("pair_*")):
        return root
    conventional = root / "experiments" / "level_00" / "multiplier_20"
    if conventional.exists() and any(conventional.glob("pair_*")):
        return conventional.resolve()
    return root

def normalize_optimizer(raw:str, include_legacy:bool=False)->dict[str,Any]:
    if raw=="adamw": fam="adamw"; legacy=False
    elif raw=="adamw_wwpgd_reference": fam="wwpgd"; legacy=False
    elif raw=="adamw_wwpgd": fam="wwpgd"; legacy=True
    else: fam=raw; legacy=True
    allowed = raw in {"adamw","adamw_wwpgd_reference"} or (include_legacy and raw=="adamw_wwpgd")
    return {"optimizer_raw":raw,"optimizer_family":fam,"optimizer_label":OPTIMIZER_LABELS.get(fam,raw),"legacy_optimizer":legacy,"allowed_by_default":allowed}

def _run_mtime(run:Path)->float:
    try: return max([run.stat().st_mtime]+[p.stat().st_mtime for p in run.glob('*')])
    except Exception: return 0.0

def _manifest_value(man:dict[str,Any], key:str)->Any:
    if key in man: return man[key]
    for sub in ("config","model_config","data","training","parameter_report"):
        if isinstance(man.get(sub),dict) and key in man[sub]: return man[sub][key]
    return None

def _run_record(pair_dir:Path,opt_dir:Path,include_legacy:bool=False)->RunRecord|None:
    runs=sorted([p for p in opt_dir.iterdir() if p.is_dir() and p.name.startswith('run_')], key=lambda p:(_run_mtime(p),p.name), reverse=True) if opt_dir.exists() else []
    if not runs: return None
    # newest directory; candidate validity assessed at pair level
    run=runs[0]; man=read_json_file(run/'manifest.json')
    raw=str(man.get('optimizer') or opt_dir.name)
    norm=normalize_optimizer(raw,include_legacy)
    seed=_manifest_value(man,'seed')
    if seed is None:
        m=re.search(r'pair_(\d+)',pair_dir.name); seed=int(m.group(1)) if m else None
    schema=float(man.get('scientific_schema_version') or 0)
    legacy=bool(norm['legacy_optimizer'] or schema<2)
    valid=bool(man.get('valid_for_science', True) is True and schema>=2 and norm['allowed_by_default'])
    complete=(run/'run_complete.json').exists()
    return RunRecord(pair_dir.name,pair_dir,run,raw,norm['optimizer_family'],norm['optimizer_label'],int(seed) if seed is not None else None,man,complete,legacy,valid)

def discover_pair_candidates(results_root:Path, include_legacy:bool=False)->list[PairCandidate]:
    root=resolve_experiment_root(results_root); out=[]
    if not root.exists(): return out
    for pair in sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith('pair_')]):
        runs={}
        for opt in (['adamw','adamw_wwpgd_reference'] + (['adamw_wwpgd'] if include_legacy else [])):
            r=_run_record(pair,pair/opt,include_legacy)
            if r: runs[r.optimizer_family]=r
        reasons=[]
        for fam in ['adamw','wwpgd']:
            r=runs.get(fam)
            if not r: reasons.append(f'missing {fam} run')
            else:
                for fn in ['manifest.json','metrics.csv','run_complete.json']:
                    if not (r.run_dir/fn).exists(): reasons.append(f'{r.optimizer_raw} missing {fn}')
                if not r.complete: reasons.append(f'{r.optimizer_raw} incomplete')
                if r.legacy: reasons.append(f'{r.optimizer_raw} legacy schema or optimizer')
                if not r.valid_for_science: reasons.append(f'{r.optimizer_raw} invalid for science')
        if not reasons and runs['adamw'].seed != runs['wwpgd'].seed: reasons.append('seed mismatch')
        if not reasons:
            for key in ['initialization_hash','tokenizer_hash','validation_probe_hash','training_probe_hash','realized_tokens']:
                if _manifest_value(runs['adamw'].manifest,key)!=_manifest_value(runs['wwpgd'].manifest,key): reasons.append(f'{key} mismatch')
            a_data=_manifest_value(runs['adamw'].manifest,'data_hash') or _manifest_value(runs['adamw'].manifest,'corpus_hash')
            w_data=_manifest_value(runs['wwpgd'].manifest,'data_hash') or _manifest_value(runs['wwpgd'].manifest,'corpus_hash')
            if a_data!=w_data: reasons.append('data_hash/corpus_hash mismatch')
        mt=max([_run_mtime(r.run_dir) for r in runs.values()] or [pair.stat().st_mtime])
        seed=next((r.seed for r in runs.values() if r.seed is not None),None)
        out.append(PairCandidate(pair.name,pair,seed,runs,not reasons,'; '.join(dict.fromkeys(reasons)),mt))
    return out

def select_canonical_pairs(candidates:list[PairCandidate])->tuple[list[PairCandidate],pd.DataFrame]:
    selected=[]; rows=[]
    by_seed={}
    for c in candidates: by_seed.setdefault(c.seed,[]).append(c)
    for seed,cs in by_seed.items():
        val=sorted([c for c in cs if c.valid], key=lambda c:(c.newest_mtime,c.pair_id), reverse=True)
        chosen=val[0] if val else None
        if chosen: selected.append(chosen)
        for c in cs:
            status='selected' if c is chosen else 'excluded'
            reason='' if c is chosen else (c.exclusion_reason or ('duplicate older complete pair' if c.valid else 'invalid'))
            rows.append({'pair_id':c.pair_id,'pair_dir':str(c.pair_dir),'seed':c.seed,'status':status,'exclusion_reason':reason,'valid_complete_pair':c.valid,'newest_mtime':c.newest_mtime})
    return sorted(selected,key=lambda c:(c.seed or -1)), pd.DataFrame(rows)

def discover_canonical_runs(results_root:Path, include_legacy:bool=False)->list[dict[str,Any]]:
    pairs,_=select_canonical_pairs(discover_pair_candidates(results_root,include_legacy))
    rows=[]
    for c in pairs:
        for fam in ['adamw','wwpgd']:
            r=c.runs[fam]; rows.append({'pair_id':c.pair_id,'pair_dir':c.pair_dir,'run_dir':r.run_dir,'seed':r.seed,'optimizer_raw':r.optimizer_raw,'optimizer_family':r.optimizer_family,'optimizer_label':r.optimizer_label,'manifest':r.manifest,'complete':r.complete,'valid_for_science':r.valid_for_science})
    return rows

def load_run_artifacts(run_dir:Path)->dict[str,Any]:
    return {'run_dir':run_dir,'manifest':read_json_file(run_dir/'manifest.json'),'complete':read_json_file(run_dir/'run_complete.json'),'metrics':normalize_metrics(load_csv_file(run_dir/'metrics.csv')),'spectral':normalize_spectral_records(load_csv_file(run_dir/'spectral.csv')),'projection':normalize_projection_records(load_csv_file(run_dir/'wwpgd_projection.csv'))}

def normalize_metrics(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy(); aliases={'tokens_processed':'tokens_seen','val_loss':'validation_loss','elapsed_time':'elapsed_seconds','projection_overhead':'projection_seconds'}
    for s,d in aliases.items():
        if s in out.columns and d not in out.columns: out[d]=out[s]
    return out

def normalize_projection_records(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy(); aliases={'actual_step':'step','actual_tokens_seen':'tokens_seen','TraceLog_before':'trace_log_before','TraceLog_after':'trace_log_after'}
    for s,d in aliases.items():
        if s in out.columns and d not in out.columns: out[d]=out[s]
    if {'trace_log_after','trace_log_before'}.issubset(out.columns): out['trace_log_delta']=pd.to_numeric(out.trace_log_after,errors='coerce')-pd.to_numeric(out.trace_log_before,errors='coerce')
    return out

def normalize_spectral_records(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy()
    if 'tokens_processed' in out.columns and 'tokens_seen' not in out.columns: out['tokens_seen']=out['tokens_processed']
    if 'layer_name' not in out.columns:
        for c in ['longname','name']:
            if c in out.columns: out['layer_name']=out[c]; break
    if 'spectral_estimator' not in out.columns: out['spectral_estimator']=pd.NA
    if 'valid_weightwatcher' not in out.columns: out['valid_weightwatcher']=out.get('spectral_estimator').astype(str).str.lower().eq('weightwatcher') if 'spectral_estimator' in out.columns else False
    if 'layer_group' not in out.columns and 'layer_name' in out.columns:
        ln=out['layer_name'].astype(str)
        out['layer_group']=np.select([ln.str.contains('wte|wpe'),ln.str.contains('lm_head'),ln.apply(is_projected_transformer_matrix)],['embedding','lm_head','projected_transformer_matrix'],default='other')
    return out

def is_projected_transformer_matrix(name:str)->bool:
    return str(name).startswith('blocks.') and any(p in str(name) for p in PROJECTED_LAYER_PATTERNS)

def build_run_inventory(runs:list[dict[str,Any]])->pd.DataFrame:
    rows=[]
    for r in runs:
        art=load_run_artifacts(Path(r['run_dir'])); m=art['metrics']; man=r.get('manifest') or art['manifest']; final=m.sort_values('tokens_seen' if 'tokens_seen' in m.columns else 'step').tail(1)
        rows.append({'seed':r['seed'],'pair_id':r['pair_id'],'optimizer_raw':r['optimizer_raw'],'optimizer_family':r['optimizer_family'],'optimizer_label':r['optimizer_label'],'run_dir':str(r['run_dir']),'completion_status':'complete' if (Path(r['run_dir'])/'run_complete.json').exists() else 'incomplete','scientific_schema_version':man.get('scientific_schema_version'),'steps':m['step'].max() if 'step' in m else np.nan,'tokens_seen':m['tokens_seen'].max() if 'tokens_seen' in m else np.nan,'realized_tokens':_manifest_value(man,'realized_tokens'),'tokens_per_parameter':_manifest_value(man,'tokens_per_parameter'),'final_train_loss':final['train_loss'].iloc[0] if len(final) and 'train_loss' in final else np.nan,'final_validation_loss':final['validation_loss'].iloc[0] if len(final) and 'validation_loss' in final else np.nan,'minimum_validation_loss':pd.to_numeric(m.get('validation_loss'),errors='coerce').min() if 'validation_loss' in m else np.nan,'token_or_step_at_min_validation_loss': (m.loc[pd.to_numeric(m['validation_loss'],errors='coerce').idxmin(), 'tokens_seen' if 'tokens_seen' in m else 'step'] if 'validation_loss' in m and m['validation_loss'].notna().any() else np.nan),'elapsed_seconds':m['elapsed_seconds'].max() if 'elapsed_seconds' in m else np.nan,'tokens_per_second':m['tokens_per_second'].mean() if 'tokens_per_second' in m else np.nan,'projection_seconds':art['projection']['projection_runtime'].sum() if not art['projection'].empty and 'projection_runtime' in art['projection'] else (m['projection_seconds'].sum() if 'projection_seconds' in m else np.nan),'weightwatcher_overhead':m['weightwatcher_overhead'].sum() if 'weightwatcher_overhead' in m else np.nan,'estimated_flops':_manifest_value(man,'estimated_flops'),'parameter_count':_manifest_value(man,'total_parameters') or _manifest_value(man,'parameter_count'),'initialization_hash':_manifest_value(man,'initialization_hash'),'validation_probe_hash':_manifest_value(man,'validation_probe_hash'),'training_probe_hash':_manifest_value(man,'training_probe_hash'),'git_commit':_manifest_value(man,'git_commit')})
    return pd.DataFrame(rows)

def build_pair_audit(candidates:list[PairCandidate])->pd.DataFrame:
    _, audit=select_canonical_pairs(candidates); return audit

def terminal_results(runs:list[dict[str,Any]], metric:str='validation_loss')->pd.DataFrame:
    rows=[]
    for r in runs:
        m=load_run_artifacts(Path(r['run_dir']))['metrics']
        if metric not in m: continue
        m=m.sort_values('tokens_seen' if 'tokens_seen' in m else 'step')
        vals=pd.to_numeric(m[metric],errors='coerce').dropna()
        if vals.empty: continue
        rows.append({'seed':r['seed'],'pair_id':r['pair_id'],'optimizer_family':r['optimizer_family'],'final':float(vals.iloc[-1]),'minimum':float(vals.min())})
    d=pd.DataFrame(rows)
    if d.empty: return d
    p=d.pivot_table(index=['seed','pair_id'],columns='optimizer_family',values=['final','minimum'],aggfunc='first'); p.columns=[f'{fam}_{met}_{metric}' for met,fam in p.columns]; p=p.reset_index()
    if {f'wwpgd_final_{metric}',f'adamw_final_{metric}'}.issubset(p.columns):
        p[f'wwpgd_minus_adamw_{metric}']=p[f'wwpgd_final_{metric}']-p[f'adamw_final_{metric}']; p[f'adamw_minus_wwpgd_{metric}_improvement']=-p[f'wwpgd_minus_adamw_{metric}']
    return p

def student_t_summary(values:Iterable[float], confidence:float=.95)->dict[str,float|int]:
    s=pd.to_numeric(pd.Series(list(values)),errors='coerce').dropna(); n=len(s); mean=float(s.mean()) if n else math.nan; sd=float(s.std(ddof=1)) if n>1 else 0.0; se=sd/math.sqrt(n) if n else math.nan; half=float(stats.t.ppf((1+confidence)/2,n-1)*se) if n>1 else 0.0
    return {'n':int(n),'mean':mean,'std':sd,'se':se,'ci_low':mean-half if n else math.nan,'ci_high':mean+half if n else math.nan,'median':float(s.median()) if n else math.nan}

def align_curves(curves:list[pd.DataFrame], x_col:str, y_col:str, points:int=200)->tuple[np.ndarray,np.ndarray]:
    clean=[]
    for df in curves:
        if x_col in df and y_col in df:
            d=df[[x_col,y_col]].apply(pd.to_numeric,errors='coerce').dropna().sort_values(x_col).drop_duplicates(x_col)
            if len(d)>=2: clean.append(d)
    if not clean: return np.array([]), np.empty((0,0))
    lo=max(d[x_col].min() for d in clean); hi=min(d[x_col].max() for d in clean)
    if not np.isfinite(lo) or hi<=lo: return np.array([]), np.empty((0,0))
    grid=np.linspace(lo,hi,points); vals=np.vstack([np.interp(grid,d[x_col],d[y_col]) for d in clean]); return grid,vals

def paired_curve_differences(pairs:list[tuple[pd.DataFrame,pd.DataFrame]], x_col:str, y_col:str, points:int=200)->tuple[np.ndarray,np.ndarray]:
    grids=[]; diffs=[]
    for a,w in pairs:
        g,v=align_curves([a,w],x_col,y_col,points)
        if v.shape[0]==2: grids.append(g); diffs.append(v[1]-v[0])
    if not diffs: return np.array([]),np.empty((0,0))
    lo=max(g.min() for g in grids); hi=min(g.max() for g in grids); grid=np.linspace(lo,hi,points)
    return grid,np.vstack([np.interp(grid,g,d) for g,d in zip(grids,diffs)])

def discover_scaling_runs(root:Path, include_legacy:bool=False)->pd.DataFrame:
    rows=[]
    for d in Path(root).rglob('pair_*'):
        if d.is_dir():
            pairs,_=select_canonical_pairs(discover_pair_candidates(d.parent,include_legacy))
            for c in pairs:
                if c.pair_dir!=d: continue
                for fam,r in c.runs.items():
                    man=r.manifest; rows.append({'level':_infer_level(d),'token_multiplier':_infer_multiplier(d),'seed':r.seed,'pair_id':c.pair_id,'optimizer_family':fam,'optimizer_raw':r.optimizer_raw,'run_dir':str(r.run_dir),'parameter_count':_manifest_value(man,'total_parameters') or _manifest_value(man,'parameter_count'),'non_embedding_parameters':_manifest_value(man,'non_embedding_parameters'),'realized_tokens':_manifest_value(man,'realized_tokens') or _manifest_value(man,'requested_tokens'),'token_budget_source':'realized_tokens' if _manifest_value(man,'realized_tokens') is not None else 'requested_tokens','estimated_flops':_manifest_value(man,'estimated_flops')})
    return pd.DataFrame(rows).drop_duplicates(subset=['run_dir']) if rows else pd.DataFrame()

def _infer_level(p:Path)->str:
    for part in p.parts:
        if part.startswith('level_'): return part
    return 'unknown'
def _infer_multiplier(p:Path)->str:
    for part in p.parts:
        if part.startswith('multiplier_'): return part
    return 'unknown'

def scaling_design_points(run_inventory:pd.DataFrame)->pd.DataFrame:
    rows=[]
    if run_inventory.empty: return pd.DataFrame()
    for keys,g in run_inventory.groupby(['level','token_multiplier','optimizer_family','parameter_count','realized_tokens'],dropna=False):
        vals=[]
        for rd in g['run_dir']:
            m=normalize_metrics(load_csv_file(Path(rd)/'metrics.csv'))
            if 'validation_loss' in m and not m.empty: vals.append(pd.to_numeric(m.sort_values('tokens_seen' if 'tokens_seen' in m else 'step')['validation_loss'],errors='coerce').dropna().iloc[-1])
        summ=student_t_summary(vals)
        rows.append(dict(zip(['level','token_multiplier','optimizer_family','parameter_count','realized_tokens'],keys), seed_count=len(vals), mean_terminal_validation_loss=summ['mean'], loss_std=summ['std'], estimated_flops=pd.to_numeric(g.get('estimated_flops'),errors='coerce').mean(), non_embedding_parameters=pd.to_numeric(g.get('non_embedding_parameters'),errors='coerce').mean(), d_over_n=(float(keys[4])/float(keys[3]) if pd.notna(keys[3]) and float(keys[3]) else np.nan)))
    return pd.DataFrame(rows)

def scaling_readiness(design:pd.DataFrame)->pd.DataFrame:
    if design.empty: return pd.DataFrame([{'ready':False,'reason':'no canonical scientific design points discovered','needed':'completed schema-v2 paired experiments'}])
    nN=design['parameter_count'].nunique(dropna=True); nD=design['realized_tokens'].nunique(dropna=True); pts=len(design[['level','token_multiplier','parameter_count','realized_tokens']].drop_duplicates())
    ready=nN>=2 and nD>=2 and pts>=4
    reason='ready for nonlinear scaling fit' if ready else f'insufficient grid: {nN} parameter count(s), {nD} token budget(s), {pts} design point(s)'
    needed='' if ready else 'add at least one more model level and one more token multiplier with completed paired seeds'
    return pd.DataFrame([{'ready':ready,'reason':reason,'needed':needed,'parameter_counts':nN,'token_budgets':nD,'design_points':pts}])

# Backward-compatible small helpers
def discover_pair_directories(results_root:Path)->list[Path]: return sorted([p for p in Path(results_root).iterdir() if p.is_dir() and p.name.startswith('pair_')]) if Path(results_root).exists() else []
def discover_experiment_runs(results_root:Path)->list[dict[str,Any]]: return discover_canonical_runs(results_root, include_legacy=True)
def select_valid_run_directory(optimizer_dir:Path)->tuple[Path|None,str]:
    r=_run_record(optimizer_dir.parent,optimizer_dir,True); return (r.run_dir,'selected newest run') if r else (None,'no run found')
def vocab_size_from_artifacts(artifacts:dict[str,Any])->int|None: return _manifest_value(artifacts.get('manifest') or artifacts.get('manifest.json') or {},'vocab_size')
def add_generalization_measures(metrics:pd.DataFrame, vocab_size:int|None=None)->pd.DataFrame:
    out=normalize_metrics(metrics)
    if 'validation_loss' in out and 'val_loss' not in out: out['val_loss']=out['validation_loss']
    for s in ['train','val']:
        loss=f'{s}_loss'
        if loss in out:
            
            if f'{s}_perplexity' not in out: out[f'{s}_perplexity']=np.exp(pd.to_numeric(out[loss],errors='coerce').clip(upper=20))
            if f'{s}_bits_per_token' not in out: out[f'{s}_bits_per_token']=pd.to_numeric(out[loss],errors='coerce')/np.log(2)
            if vocab_size and vocab_size>1: out[f'{s}_token_prediction_capacity']=1-out[f'{s}_bits_per_token']/np.log2(vocab_size)
    if {'val_loss','train_loss'}.issubset(out.columns): 
        if 'generalization_gap' not in out: out['generalization_gap']=out['val_loss']-out['train_loss']
    if {'val_perplexity','train_perplexity'}.issubset(out.columns):
        out['perplexity_gap']=out['val_perplexity']-out['train_perplexity']; out['perplexity_ratio']=out['val_perplexity']/out['train_perplexity']
    return out

def summary(s: pd.Series) -> dict[str, float | int]:
    return student_t_summary(pd.to_numeric(s, errors='coerce').dropna())

def completed_runs(root: Path, scientific_only: bool = True) -> list[Path]:
    out=[]
    for p in Path(root).rglob('run_complete.json'):
        man=read_json_file(p.parent/'manifest.json')
        if scientific_only and man.get('valid_for_science', True) is not True: continue
        out.append(p.parent)
    return out

def analyze_results(results_root: Path) -> Path:
    out=Path(results_root)/'analysis'; out.mkdir(parents=True, exist_ok=True)
    runs=discover_canonical_runs(results_root, include_legacy=True)
    inv=build_run_inventory(runs) if runs else pd.DataFrame()
    inv.to_csv(out/'runs_manifest.csv', index=False)
    terminal_results(runs).to_csv(out/'paired_metric_differences.csv', index=False)
    pd.DataFrame([{'status':'not_fit','note':'use notebooks for schema-v2 analysis'}]).to_csv(out/'scaling_fit_results.csv', index=False)
    (out/'analysis_manifest.json').write_text(json.dumps({'source':str(results_root),'completed_runs':len(runs)}))
    return out

# Override legacy compatibility shims for existing tests.
def summary(s: pd.Series) -> dict[str, float | int]:
    s = pd.to_numeric(s, errors='coerce').dropna(); n=len(s); mean=float(s.mean()) if n else float('nan'); sd=float(s.std(ddof=1)) if n>1 else 0.0; se=sd/(n**0.5) if n else float('nan'); half=float(stats.t.ppf(.975,n-1)*se) if n>1 else 0.0
    return {'n':int(n),'mean':mean,'sample_std':sd,'standard_error':se,'ci95_low':mean-half if n else float('nan'),'ci95_high':mean+half if n else float('nan'),'median':float(s.median()) if n else float('nan')}

def load_run_artifacts(run_dir: Path) -> dict[str, Any]:
    d={'run_dir':run_dir,'manifest':read_json_file(run_dir/'manifest.json'),'manifest.json':read_json_file(run_dir/'manifest.json'),'complete':read_json_file(run_dir/'run_complete.json'),'run_complete.json':read_json_file(run_dir/'run_complete.json')}
    d['metrics']=normalize_metrics(load_csv_file(run_dir/'metrics.csv')); d['metrics.csv']=d['metrics']
    d['spectral']=normalize_spectral_records(load_csv_file(run_dir/'spectral.csv')); d['spectral.csv']=d['spectral']
    d['projection']=normalize_projection_records(load_csv_file(run_dir/'wwpgd_projection.csv')); d['wwpgd_projection.csv']=d['projection']
    return d

def select_valid_run_directory(optimizer_dir: Path) -> tuple[Path | None, str]:
    runs=sorted([p for p in optimizer_dir.iterdir() if p.is_dir() and p.name.startswith('run_')], key=lambda p:(_run_mtime(p),p.name), reverse=True) if optimizer_dir.exists() else []
    complete=[p for p in runs if (p/'manifest.json').exists() and (p/'metrics.csv').exists() and (p/'run_complete.json').exists()]
    if complete: return complete[0], 'selected most recent completed valid run'
    valid=[p for p in runs if (p/'manifest.json').exists() and (p/'metrics.csv').exists()]
    if valid: return valid[0], 'selected most recent valid but incomplete run'
    return (runs[0], 'no valid run found; selected newest directory for audit') if runs else (None,'no run found')

def discover_experiment_runs(results_root: Path) -> list[dict[str, Any]]:
    rows=[]
    for pair in discover_pair_directories(results_root):
        for opt in ['adamw','adamw_wwpgd','adamw_wwpgd_reference']:
            rd,note=select_valid_run_directory(pair/opt)
            art=load_run_artifacts(rd) if rd else {'files_loaded':[]}
            man=art.get('manifest',{})
            seed=man.get('seed')
            if seed is None:
                m=re.search(r'pair_(\d+)',pair.name); seed=int(m.group(1)) if m else None
            rows.append({'pair_id':pair.name,'pair_dir':pair,'optimizer':opt,'optimizer_raw':opt,'run_dir':rd,'selection_note':note,'seed':seed,'artifacts':art})
    return rows

def discover_experiment_runs(results_root: Path) -> list[dict[str, Any]]:
    rows=[]
    for pair in discover_pair_directories(results_root):
        opts=['adamw','adamw_wwpgd'] + (['adamw_wwpgd_reference'] if (pair/'adamw_wwpgd_reference').exists() else [])
        for opt in opts:
            rd,note=select_valid_run_directory(pair/opt)
            art=load_run_artifacts(rd) if rd else {'files_loaded':[]}
            man=art.get('manifest',{})
            seed=man.get('seed')
            if seed is None:
                m=re.search(r'pair_(\d+)',pair.name); seed=int(m.group(1)) if m else None
            norm=normalize_optimizer(opt, include_legacy=True)
            rows.append({'pair_id':pair.name,'pair_dir':pair,'optimizer':opt,'optimizer_raw':opt,'optimizer_family':norm['optimizer_family'],'optimizer_label':norm['optimizer_label'],'run_dir':rd,'selection_note':note,'seed':seed,'artifacts':art})
    return rows

def add_generalization_measures(metrics:pd.DataFrame, vocab_size:int|None=None)->pd.DataFrame:
    out=normalize_metrics(metrics)
    if 'validation_loss' in out and 'val_loss' not in out: out['val_loss']=out['validation_loss']
    for split in ['train','val']:
        loss=f'{split}_loss'
        if loss in out:
            if f'{split}_perplexity' not in out: out[f'{split}_perplexity']=np.exp(pd.to_numeric(out[loss],errors='coerce').clip(upper=20))
            if f'{split}_bits_per_token' not in out: out[f'{split}_bits_per_token']=pd.to_numeric(out[loss],errors='coerce')/np.log(2)
            if vocab_size and vocab_size>1: out[f'{split}_token_prediction_capacity']=1-out[f'{split}_bits_per_token']/np.log2(vocab_size)
    if {'val_loss','train_loss'}.issubset(out.columns) and 'generalization_gap' not in out: out['generalization_gap']=out['val_loss']-out['train_loss']
    if {'val_perplexity','train_perplexity'}.issubset(out.columns): out['perplexity_gap']=out['val_perplexity']-out['train_perplexity']; out['perplexity_ratio']=out['val_perplexity']/out['train_perplexity']
    if {'val_token_prediction_capacity','train_token_prediction_capacity'}.issubset(out.columns): out['capacity_generalization_gap']=out['train_token_prediction_capacity']-out['val_token_prediction_capacity']
    return out

def terminal_results(runs:list[dict[str,Any]], metric:str='validation_loss')->pd.DataFrame:
    rows=[]
    for r in runs:
        if not r.get('run_dir'): continue
        m=(r.get('artifacts') or {}).get('metrics') if isinstance(r.get('artifacts'),dict) else None
        if m is None: m=load_run_artifacts(Path(r['run_dir']))['metrics']
        if metric not in m: continue
        vals=pd.to_numeric(m.sort_values('tokens_seen' if 'tokens_seen' in m else 'step')[metric],errors='coerce').dropna()
        if vals.empty: continue
        fam=r.get('optimizer_family') or normalize_optimizer(r.get('optimizer_raw') or r.get('optimizer',''), True)['optimizer_family']
        rows.append({'seed':r.get('seed'),'pair_id':r.get('pair_id'),'optimizer_family':fam,'optimizer':r.get('optimizer_raw') or r.get('optimizer'),'final':float(vals.iloc[-1]),'minimum':float(vals.min())})
    d=pd.DataFrame(rows)
    if d.empty: return d
    p=d.pivot_table(index=['pair_id','seed'],columns='optimizer_family',values=['final','minimum'],aggfunc='first'); p.columns=[f'{fam}_{met}_{metric}' for met,fam in p.columns]; p=p.reset_index()
    if {f'wwpgd_final_{metric}',f'adamw_final_{metric}'}.issubset(p.columns):
        p[f'wwpgd_minus_adamw_{metric}']=p[f'wwpgd_final_{metric}']-p[f'adamw_final_{metric}']; p[f'adamw_minus_wwpgd_{metric}_improvement']=-p[f'wwpgd_minus_adamw_{metric}']
        # old compatibility names
        if metric=='validation_loss':
            p['adamw_final_validation_loss']=p[f'adamw_final_{metric}']; p['adamw_wwpgd_final_validation_loss']=p[f'wwpgd_final_{metric}']; p['wwpgd_minus_adamw_final_validation_loss']=p[f'wwpgd_minus_adamw_{metric}']
    return p
