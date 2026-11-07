"""Summarize paired reference scores and verify the previous test replay."""
import json
from pathlib import Path
from v2_data import atomic_json
ROOT=Path('/data/aerospace-intelligence/runs')
V2=ROOT/'uavid-20261103-v2-error-attack'
V3=ROOT/'uavid-20261105-v3-error-attack'
OUT=ROOT/'uavid-20261105-v4-human-recall/final_reference'
def main():
 val2=json.loads((V2/'official_validation.json').read_text());val4=json.loads((OUT/'official_validation.json').read_text())
 tst=json.loads((OUT/'public_test.json').read_text());previous=json.loads((V3/'public_test_verified.json').read_text())
 summary=json.loads((OUT/'prediction_summary.json').read_text())
 if val4['images']!=70 or tst['images']!=150 or summary['test_images']!=150: raise ValueError('incomplete evaluation')
 if tst['models']['v2']!=previous['models']['v2']: raise ValueError('previous model test replay changed')
 if tst['selection_sha256_before_test']!=json.loads((OUT/'freeze_sha256.json').read_text())['sha256']: raise ValueError('selection fingerprint mismatch')
 result={'status':'complete','validation':{'images':70,'v2':val2['metrics'],'v4':val4['metrics']},
         'test':{'images':150,'v2':tst['models']['v2'],'v4':tst['models']['v3'],'ignored_pixels':tst['ignored_pixels'],
                 'source':tst['source'],'source_revision':tst['source_revision'],'source_status':tst['source_status']},
         'qualification':'Validation was previously observed; test labels were previously observed. Paired comparisons are descriptive, not blind test estimates.',
         'selection_sha256':tst['selection_sha256_before_test'],'test_predictions':str(OUT/'test_predictions')}
 atomic_json(OUT/'final_report.json',result)
 print(json.dumps({'stage':'report_complete','val_v4_miou':val4['metrics']['miou'],'test_v4_miou':tst['models']['v3']['miou']}),flush=True)
if __name__=='__main__':main()
