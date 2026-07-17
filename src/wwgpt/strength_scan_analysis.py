from __future__ import annotations
import json, math
from pathlib import Path
import pandas as pd
import numpy as np

RUN_COLS=['scan_id','seed','strength','optimizer_family','optimizer_raw','run_dir','complete','stable','instability_reason','steps','tokens_seen','final_validation_loss','minimum_validation_loss','elapsed_seconds','tokens_per_second','estimated_flops','total_projection_runtime','total_periodic_weightwatcher_runtime','total_immediate_weightwatcher_runtime','mean_relative_frobenius_change','maximum_relative_frobenius_change','projection_events','changed_layer_rows']

def resolve_scan_root(path: Path) -> Path:
    path=Path(path)
    if (path/'scan_manifest.json').exists(): return path
    if (path/'latest_scan_root.txt').exists():
        p=Path((path/'latest_scan_root.txt').read_text().strip());
        if (p/'scan_manifest.json').exists(): print(f'Selected strength scan: {p}'); return p
    cands=sorted(path.rglob('scan_manifest.json'), key=lambda p: str(p.parent))
    if not cands: raise FileNotFoundError(f'no strength scan under {path}')
    print(f'Selected strength scan: {cands[-1].parent}')
    return cands[-1].parent

def _j(p):
    return json.loads(p.read_text()) if p.exists() else {}

def _run_rows(scan_root: Path):
    scan=_j(scan_root/'scan_manifest.json'); scan_id=scan.get('scan_id', scan_root.name)
    rows=[]
    for manp in scan_root.rglob('manifest.json'):
        run=manp.parent; man=_j(manp); opt=man.get('optimizer','')
        if opt not in ('adamw','adamw_wwpgd_reference'): continue
        metrics=pd.read_csv(run/'metrics.csv') if (run/'metrics.csv').exists() else pd.DataFrame()
        proj=pd.read_csv(run/'wwpgd_projection.csv') if (run/'wwpgd_projection.csv').exists() else pd.DataFrame()
        im=pd.read_csv(run/'wwpgd_projection_spectral.csv') if (run/'wwpgd_projection_spectral.csv').exists() else pd.DataFrame()
        rc=_j(run/'run_complete.json')
        val=metrics['val_loss'] if 'val_loss' in metrics else pd.Series(dtype=float)
        rows.append({
          'scan_id':scan_id,'seed':man.get('seed'),'strength':man.get('scan_strength', np.nan if opt=='adamw' else man.get('wwpgd_strength',np.nan)),
          'optimizer_family':'adamw' if opt=='adamw' else 'wwpgd','optimizer_raw':opt,'run_dir':str(run),'complete':(run/'run_complete.json').exists(),
          'stable':man.get('stable', True),'instability_reason':man.get('instability_reason',''),'steps':rc.get('step', metrics['step'].max() if 'step' in metrics else 0),
          'tokens_seen':metrics['tokens_processed'].max() if 'tokens_processed' in metrics else man.get('realized_tokens',0),
          'final_validation_loss':float(val.iloc[-1]) if len(val) else math.nan,'minimum_validation_loss':float(val.min()) if len(val) else math.nan,
          'elapsed_seconds':float(metrics['elapsed_time'].max()) if 'elapsed_time' in metrics and len(metrics) else math.nan,
          'tokens_per_second':float(metrics['tokens_per_second'].iloc[-1]) if 'tokens_per_second' in metrics and len(metrics) else math.nan,
          'estimated_flops':man.get('estimated_flops',0),'total_projection_runtime':float(proj.get('projection_runtime',pd.Series(dtype=float)).sum()),
          'total_periodic_weightwatcher_runtime':float(metrics.get('weightwatcher_overhead',pd.Series(dtype=float)).max()) if len(metrics) else 0.0,
          'total_immediate_weightwatcher_runtime':float(im.get('pre_weightwatcher_runtime',pd.Series(dtype=float)).sum()+im.get('post_weightwatcher_runtime',pd.Series(dtype=float)).sum()),
          'mean_relative_frobenius_change':float(proj.get('relative_frobenius_change',pd.Series(dtype=float)).mean()) if len(proj) else math.nan,
          'maximum_relative_frobenius_change':float(proj.get('relative_frobenius_change',pd.Series(dtype=float)).max()) if len(proj) else math.nan,
          'projection_events':int(proj.get('projection_event',pd.Series(dtype=float)).nunique()) if len(proj) else 0,
          'changed_layer_rows':int(proj.get('changed',pd.Series(dtype=bool)).sum()) if len(proj) else 0})
    return pd.DataFrame(rows, columns=RUN_COLS)

def analyze_strength_scan(scan_root: Path) -> Path:
    scan_root=resolve_scan_root(scan_root); out=scan_root/'analysis'; out.mkdir(exist_ok=True)
    runs=_run_rows(scan_root); runs.to_csv(out/'strength_scan_runs.csv',index=False)
    controls=runs[runs.optimizer_family=='adamw'].set_index('seed')
    terms=[]
    for _,r in runs[runs.optimizer_family=='wwpgd'].iterrows():
        c=controls.loc[r.seed] if r.seed in controls.index else None
        terms.append({'seed':r.seed,'strength':r.strength,'adamw_control_run': '' if c is None else c.run_dir,'wwpgd_run':r.run_dir,
        'adamw_final_validation_loss':math.nan if c is None else c.final_validation_loss,'wwpgd_final_validation_loss':r.final_validation_loss,'wwpgd_minus_adamw_final_loss':math.nan if c is None else r.final_validation_loss-c.final_validation_loss,
        'adamw_minimum_validation_loss':math.nan if c is None else c.minimum_validation_loss,'wwpgd_minimum_validation_loss':r.minimum_validation_loss,'wwpgd_minus_adamw_minimum_loss':math.nan if c is None else r.minimum_validation_loss-c.minimum_validation_loss,'stable':r.stable})
    terminal=pd.DataFrame(terms); terminal.to_csv(out/'strength_scan_terminal.csv',index=False)
    projs=[]; specs=[]
    for _,r in runs[runs.optimizer_family=='wwpgd'].iterrows():
        run=Path(r.run_dir)
        if (run/'wwpgd_projection.csv').exists():
            df=pd.read_csv(run/'wwpgd_projection.csv');
            if len(df):
                g=df.groupby('projection_event');
                for ev,d in g: projs.append({'seed':r.seed,'strength':r.strength,'projection_event':ev,'scheduled_token_fraction':d.get('scheduled_token_fraction',pd.Series([np.nan])).iloc[0],'actual_tokens_seen':d.get('actual_tokens_seen',pd.Series([np.nan])).iloc[0],'schedule_hardness':d.get('schedule_hardness',pd.Series([np.nan])).median(),'effective_hardness':d.get('effective_hardness',pd.Series([np.nan])).median(),'eligible_layers':len(d),'changed_layers':int(d.get('changed',pd.Series(False,index=d.index)).sum()),'skipped_layers':int((~d.get('changed',pd.Series(False,index=d.index)).astype(bool)).sum()),'median_relative_frobenius_change':d.get('relative_frobenius_change',pd.Series(dtype=float)).median(),'maximum_relative_frobenius_change':d.get('relative_frobenius_change',pd.Series(dtype=float)).max(),'total_projection_runtime':d.get('projection_runtime',pd.Series(dtype=float)).sum(),'median_selected_tail_size':d.get('selected_tail_size',pd.Series(dtype=float)).median(),'median_TraceLog_change':(d.get('TraceLog_after',0)-d.get('TraceLog_before',0)).median()})
        if (run/'wwpgd_projection_spectral.csv').exists():
            df=pd.read_csv(run/'wwpgd_projection_spectral.csv')
            for ev,d in df.groupby('projection_event'):
                good=d['alpha_before'].notna() & d['alpha_after'].notna() if 'alpha_before' in d else pd.Series(False,index=d.index)
                specs.append({'seed':r.seed,'strength':r.strength,'projection_event':ev,'eligible_alpha_fits':int(good.sum()),'median_alpha_before':d.get('alpha_before',pd.Series(dtype=float)).median(),'median_alpha_after':d.get('alpha_after',pd.Series(dtype=float)).median(),'median_abs_alpha_error_before':d.get('abs_alpha_error_before',pd.Series(dtype=float)).median(),'median_abs_alpha_error_after':d.get('abs_alpha_error_after',pd.Series(dtype=float)).median(),'median_abs_alpha_error_change':d.get('abs_alpha_error_change',pd.Series(dtype=float)).median(),'fraction_layers_closer_to_target':float((d.get('abs_alpha_error_change',pd.Series(dtype=float))<0).mean()),'fraction_layers_farther_from_target':float((d.get('abs_alpha_error_change',pd.Series(dtype=float))>0).mean()),'median_D_before':d.get('D_before',pd.Series(dtype=float)).median(),'median_D_after':d.get('D_after',pd.Series(dtype=float)).median(),'median_num_evals':d.get('num_evals_before',pd.Series(dtype=float)).median(),'median_relative_frobenius_change':d.get('relative_frobenius_change',pd.Series(dtype=float)).median()})
    projection=pd.DataFrame(projs); spectral=pd.DataFrame(specs); projection.to_csv(out/'strength_scan_projection.csv',index=False); spectral.to_csv(out/'strength_scan_spectral.csv',index=False)
    summ=[]
    for strength,d in terminal.groupby('strength'):
        dif=d['wwpgd_minus_adamw_final_loss'].dropna(); n=len(dif); mean=dif.mean() if n else math.nan; std=dif.std(ddof=1) if n>1 else math.nan; se=std/math.sqrt(n) if n>1 else math.nan; ci=1.96*se if n>1 else math.nan
        p=projection[projection.strength==strength] if len(projection) else pd.DataFrame(); sp=spectral[spectral.strength==strength] if len(spectral) else pd.DataFrame(); rr=runs[(runs.optimizer_family=='wwpgd')&(runs.strength==strength)]
        ctrl=runs[runs.optimizer_family=='adamw'].groupby('seed')['elapsed_seconds'].mean()
        overhead=[]
        for _,row in rr.iterrows():
            if row.seed in ctrl and ctrl[row.seed]: overhead.append(row.elapsed_seconds/ctrl[row.seed]-1)
        summ.append({'strength':strength,'seed_count':n,'stable_seed_count':int(d.get('stable',False).sum()),'stable_run_fraction':float(d.get('stable',False).mean()) if len(d) else math.nan,'mean_final_loss_difference':mean,'sample_std_final_loss_difference':std,'standard_error_final_loss_difference':se,'ci95_final_loss_difference_low':mean-ci if n>1 else math.nan,'ci95_final_loss_difference_high':mean+ci if n>1 else math.nan,'median_final_loss_difference':dif.median() if n else math.nan,'wwpgd_wins':int((dif<0).sum()),'adamw_wins':int((dif>0).sum()),'ties':int((dif==0).sum()),'mean_projection_event_alpha_error_change':sp.get('median_abs_alpha_error_change',pd.Series(dtype=float)).mean(),'mean_fraction_layers_closer_to_target':sp.get('fraction_layers_closer_to_target',pd.Series(dtype=float)).mean(),'mean_projection_norm':p.get('median_relative_frobenius_change',pd.Series(dtype=float)).mean(),'maximum_projection_norm':p.get('maximum_relative_frobenius_change',pd.Series(dtype=float)).max(),'mean_total_projection_runtime':rr['total_projection_runtime'].mean(),'mean_total_immediate_weightwatcher_runtime':rr['total_immediate_weightwatcher_runtime'].mean(),'mean_tokens_per_second':rr['tokens_per_second'].mean(),'mean_runtime_overhead_vs_control':float(np.mean(overhead)) if overhead else math.nan})
    pd.DataFrame(summ).sort_values('strength').to_csv(out/'strength_scan_summary.csv',index=False)
    return out
