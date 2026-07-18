from __future__ import annotations
import json, csv, math
from pathlib import Path
import pandas as pd


def audit_run(run: Path):
    run=Path(run); reasons=[]
    man={}
    if (run/'manifest.json').exists(): man=json.loads((run/'manifest.json').read_text())
    fixture=not man.get('valid_for_science', False) or man.get('dataset_name')=='local_fixture'
    if fixture: reasons.append('fixture_or_invalid_for_science')
    imm=run/'wwpgd_projection_spectral.csv'
    valid_immediate=False
    if imm.exists():
        df=pd.read_csv(imm)
        required={'immediate_spectral_source','measurement_valid_for_science','alpha_before','alpha_after','weightwatcher_configuration'}
        if not required.issubset(df.columns): reasons.append('legacy_or_missing_measured_provenance_fields')
        elif (df['immediate_spectral_source']=='weightwatcher_measured').all() and df['measurement_valid_for_science'].astype(str).str.lower().isin(['true','1']).any(): valid_immediate=True
        else: reasons.append('no_valid_immediate_weightwatcher_rows')
    complete=(run/'run_complete.json').exists()
    if not complete: reasons.append('run_incomplete')
    valid_loss=complete and not fixture and (run/'metrics.csv').exists()
    out={
        'run_dir':str(run),'valid_for_loss_analysis':valid_loss,'valid_for_accuracy_analysis':valid_loss,
        'valid_for_periodic_weightwatcher_analysis':complete and not fixture and (run/'spectral.csv').exists(),
        'valid_for_immediate_weightwatcher_analysis':valid_immediate,
        'valid_for_projection_analysis':complete and not fixture and (run/'wwpgd_projection.csv').exists(),
        'valid_for_publication':False,'reasons':';'.join(reasons)
    }
    out['valid_for_publication']=all(out[k] for k in out if k.startswith('valid_for_') and k!='valid_for_publication') and not reasons
    return out

def audit_experiment(root: Path):
    root=Path(root); analysis=root/'analysis'; analysis.mkdir(parents=True, exist_ok=True)
    runs=[p.parent for p in root.rglob('manifest.json') if p.parent.name.startswith('run_')]
    rows=[audit_run(r) for r in runs]
    fields=list(rows[0]) if rows else ['run_dir','valid_for_publication','reasons']
    with (analysis/'integrity_audit.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary={'experiment_root':str(root),'run_count':len(rows),'publication_eligible_runs':sum(bool(r.get('valid_for_publication')) for r in rows),'valid_for_publication':bool(rows) and all(r.get('valid_for_publication') for r in rows),'failures':[r for r in rows if not r.get('valid_for_publication')]}
    (analysis/'integrity_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    (analysis/'integrity_report.md').write_text('# Integrity audit\n\n'+json.dumps(summary, indent=2)+'\n')
    return analysis/'integrity_summary.json'

def audit_strength_scan(scan_root: Path):
    return audit_experiment(scan_root)
