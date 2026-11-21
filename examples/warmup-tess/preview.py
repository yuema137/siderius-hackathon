#!/usr/bin/env python3
"""Tiny real-TESS CPU preview; no SIDERIUS training and no API calls."""
import argparse, csv, hashlib, html, io, json, urllib.request
from pathlib import Path

EXP_REV = 'c30ebb8a9f54dca8de89d016f9ebf313aa489daf'
DATA_REV = '9f203f4c338645a1e4b2c9dc7d6f820269ca5114'
FILES = {
 'identity.csv': (f'https://raw.githubusercontent.com/yuema137/siderius-exp/{EXP_REV}/tasks/phyts_tess/data/manifests/rotation_identity.csv', 'b03b8872943b696ab3dc7a63a8579f5dec379c2bc0562ab1c613122e3cea50ad'),
 'validation.parquet': (f'https://huggingface.co/datasets/PhyTS-team/PhyTS-bench/resolve/{DATA_REV}/TESS/split/tess_regression_val.parquet', '76ba732c8cb5cea71f7a651a611e08494a79fe36962216dfbea49dc60e335502'),
}
def fetch(root, offline=False):
 root.mkdir(parents=True, exist_ok=True)
 for name,(url,digest) in FILES.items():
  p=root/name
  if not p.exists():
   if offline: raise ValueError(f'Missing cached {name}; run once without --offline.')
   temporary=p.with_suffix('.partial')
   try:
    count=0
    with urllib.request.urlopen(url,timeout=60) as source, temporary.open('wb') as out:
     while chunk:=source.read(1024*1024):
      count+=len(chunk)
      if count>32*1024*1024:raise ValueError('Download exceeds the 32 MiB per-file warm-up limit.')
      out.write(chunk)
    if hashlib.sha256(temporary.read_bytes()).hexdigest()!=digest:raise ValueError(f'Checksum mismatch: {name}')
    temporary.replace(p)
   finally:temporary.unlink(missing_ok=True)
  if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError(f'Cached checksum mismatch: {name}')

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--offline',action='store_true');args=ap.parse_args()
 import numpy as np
 import pyarrow.parquet as pq
 root=args.output_dir.resolve();fetch(root/'data',args.offline)
 rows=list(csv.DictReader((root/'data/identity.csv').open()))
 train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
 assert len(train)==3338 and len(val)==442
 assert not ({r['gaia_id'] for r in train}&{r['gaia_id'] for r in val})
 mean=float(np.mean([float(r['frot']) for r in train]));truth=np.array([float(r['frot']) for r in val]);err=truth-mean
 result={'task':'TESS near-core rotation frequency, cycles/day','model':'training_mean','prediction':mean,'train_count':len(train),'validation_count':len(val),'r2':float(1-(err@err)/((truth-truth.mean())@(truth-truth.mean()))),'rmse':float(np.sqrt(np.mean(err**2))),'mae':float(np.mean(abs(err))),'training':'Mean fitted from training targets only; no neural-network training.','scope':'All 442 released validation curves; stars disjoint from training. Not held-out test or competition score.','native_siderius_training':False,'api_calls':0,'exp_revision':EXP_REV,'data_revision':DATA_REV}
 table=pq.read_table(root/'data/validation.parquet',columns=['GaiaID','sector','flux']).to_pylist()
 expected={(int(r['gaia_id']),int(r['sector'])) for r in val}
 assert {(int(r['GaiaID']),int(r['sector'])) for r in table}==expected
 panels=[]
 for row in table[:3]:
  a=np.asarray(row['flux'],float);a=a[np.isfinite(a)]
  # Preview only: show up to 1024 samples; this is not a replacement task preprocessing implementation.
  a=a[:1024];lo,hi=float(a.min()),float(a.max());span=max(hi-lo,1e-12)
  points=' '.join(f'{40+i*900/max(1,len(a)-1):.1f},{210-(y-lo)*170/span:.1f}' for i,y in enumerate(a))
  panels.append(f'<h3>Gaia {row["GaiaID"]}, sector {row["sector"]}</h3><svg viewBox="0 0 980 260"><path d="M40 25V210H950" fill="none" stroke="#65788d"/><polyline points="{points}" fill="none" stroke="#34c9c4" stroke-width="1.4"/><text x="40" y="245" fill="white">Sample index (first 1024) — raw relative flux; each plot has its own vertical scale</text></svg>')
 summary=f'The same prediction for every star gives validation R² = {result["r2"]:.4f}. It ignores the light curve: try a model that uses its shape, then compare on these same validation stars.'
 (root/'baseline.json').write_text(json.dumps(result,indent=2))
 (root/'preview.html').write_text('<!doctype html><meta charset="utf-8"><title>TESS CPU warm-up</title><style>body{max-width:1000px;margin:40px auto;background:#081724;color:#e4eef8;font:20px system-ui;padding:20px}svg{width:100%}pre{font-size:15px;white-space:pre-wrap}h1,h2{color:#42d1ca}</style><h1>From a star’s brightness to its rotation</h1><p>Input: one TESS light curve. Target: near-core rotation frequency, in cycles per day. The benchmark’s target is not automatically the largest peak in a light-curve Fourier spectrum.</p><h2>First baseline: always predict the training mean</h2><p>'+html.escape(summary)+'</p>'+''.join(panels)+'<h2>What happened?</h2><p>Real public data, a CPU-only constant baseline, and no API calls. This is preparation for a small reasoning/proposal orchestration, not a SIDERIUS training loop.</p><pre>'+html.escape(json.dumps(result,indent=2))+'</pre>')
 (root/'status.json').write_text(json.dumps({'preview':'completed','luna_orchestration':'not_run','native_training':'not_run'},indent=2))
 print(summary);print('Open:',root/'preview.html');print('Measured result:',root/'baseline.json')
if __name__=='__main__':main()
