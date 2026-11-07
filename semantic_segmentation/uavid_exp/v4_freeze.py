"""Freeze the chosen fourth-round final weights before reference evaluation."""
from pathlib import Path
import hashlib,json
from v2_data import atomic_json
RUN=Path('/data/aerospace-intelligence/runs/uavid-20261105-v4-human-recall')
OUT=RUN/'final_reference'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 selection=json.loads((RUN/'selection.json').read_text())
 if selection['selected']=='v2_fallback' or set(selection['models'])!={'global','tile','sfa'}: raise ValueError('unexpected frozen fourth-round selection')
 v=selection['development']['variants'][selection['selected']]
 if not v['eligible']: raise ValueError('fourth-round candidate did not meet its acceptance rule')
 selection['checkpoint_sha256']={v['checkpoint']:sha(v['checkpoint']) for v in selection['models'].values()}
 selection['frozen_before_official_and_test']=True
 selection['test_qualification']='test labels were previously observed; comparison is descriptive and not a new blind test'
 OUT.mkdir(parents=True,exist_ok=True)
 path=OUT/'selection_frozen.json'
 if path.exists():
  if json.loads(path.read_text())!=selection: raise ValueError('existing freeze differs; refusing overwrite')
 else: atomic_json(path,selection)
 atomic_json(OUT/'freeze_sha256.json',{'sha256':sha(path)})
 print(json.dumps({'stage':'freeze_complete','selection_sha256':sha(path),'models':len(selection['models'])}),flush=True)
if __name__=='__main__':main()
