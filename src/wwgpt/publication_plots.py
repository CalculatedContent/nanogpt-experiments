from __future__ import annotations
import json, subprocess, time
from pathlib import Path
import pandas as pd

def student_t_ci_by_seed(df: pd.DataFrame, metric: str, group_cols: list[str]):
    import numpy as np
    try:
        from scipy.stats import t
    except Exception:
        t=None
    agg=df.groupby(group_cols+['seed'], dropna=False)[metric].mean().reset_index()
    rows=[]
    for keys,g in agg.groupby(group_cols, dropna=False):
        vals=g[metric].dropna().to_numpy(); n=len(vals); mean=float(vals.mean()) if n else float('nan')
        ci=float('nan')
        if n>1:
            crit=float(t.ppf(0.975,n-1)) if t else 12.706 if n==2 else 4.303
            ci=crit*float(vals.std(ddof=1))/(n**0.5)
        if not isinstance(keys, tuple): keys=(keys,)
        rows.append(dict(zip(group_cols,keys), mean=mean, ci95=ci, seed_count=n, ci_method='Student-t across independent seeds' if n>1 else 'unavailable: fewer than two independent seeds'))
    return pd.DataFrame(rows)

def save_publication_figure(fig, source_df: pd.DataFrame, analysis_dir: Path, figure_name: str, metadata: dict):
    analysis_dir=Path(analysis_dir); analysis_dir.mkdir(parents=True, exist_ok=True)
    png=analysis_dir/f'{figure_name}.png'; pdf=analysis_dir/f'{figure_name}.pdf'; csv=analysis_dir/f'{figure_name}_data.csv'; meta=analysis_dir/f'{figure_name}_metadata.json'
    fig.savefig(png, dpi=300, bbox_inches='tight'); fig.savefig(pdf, bbox_inches='tight'); source_df.to_csv(csv,index=False)
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    except Exception: commit='unknown'
    md={**metadata,'generated_timestamp':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),'git_commit':commit,'figure_generation_code_version':'publication_plots_v1','source_data':str(csv)}
    meta.write_text(json.dumps(md, indent=2, sort_keys=True)+'\n')
    return {'png':png,'pdf':pdf,'csv':csv,'metadata':meta}
