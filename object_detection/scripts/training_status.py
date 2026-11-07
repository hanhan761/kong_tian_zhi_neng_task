"""Read training CSV files; this command does not start another training process."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
rows=[]
for path in sorted((ROOT/'runs').glob('*/*/results.csv')):
    with path.open(encoding='utf-8-sig') as stream:
        records=[{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader(stream)]
    if records:
        best=max(records,key=lambda r:float(r['metrics/mAP_0.5']))
        rows.append(dict(run=str(path.parent.relative_to(ROOT)),completed_epochs=len(records),
                         best_epoch_display=int(best['epoch'])+1,
                         best_map50_percent=100*float(best['metrics/mAP_0.5']),
                         threshold91_reached=float(best['metrics/mAP_0.5'])>=.91,
                         last_map50_percent=100*float(records[-1]['metrics/mAP_0.5'])))
print(json.dumps(rows,ensure_ascii=False,indent=2))
