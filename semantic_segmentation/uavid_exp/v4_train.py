"""Fine-tune nested-fold tile models with a fixed human-weighted objective."""
from pathlib import Path
import argparse,json,sys
import torch
import torch.nn.functional as F
import v2_train,v3_train
from v2_data import atomic_json

def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--human-weight',type=float,required=True)
    p.add_argument('--out',required=True);a,rest=p.parse_known_args()
    if a.human_weight not in (2.,4.): raise ValueError('unsupported weight')
    path=Path(a.out)/'v4_contract.json'; contract={'human_weight':a.human_weight,'loss':'weighted_ce_plus_rare_dice_0.3'}
    if path.exists() and json.loads(path.read_text())!=contract: raise ValueError('v4 objective changed')
    if not path.exists(): atomic_json(path,contract)
    def loss_for(logits,target,expert,loss_name):
        if expert or loss_name!='ce_dice': raise ValueError('expected multiclass CE plus Dice')
        weights=torch.ones(8,device=logits.device);weights[6]=a.human_weight
        loss=F.cross_entropy(logits,target,weight=weights,ignore_index=255)
        probs=logits.float().softmax(1);valid=target!=255;terms=[]
        for cls in (3,6,7):
            actual=(target==cls)&valid
            if actual.any():
                pred=probs[:,cls]*valid
                terms.append(1-(2*(pred*actual).sum()+1)/(pred.sum()+actual.sum()+1))
        return loss+(.3*torch.stack(terms).mean() if terms else 0)
    v2_train.loss_for=loss_for
    sys.argv=[sys.argv[0],'--out',a.out,*rest]
    v3_train.main()
if __name__=='__main__': main()
