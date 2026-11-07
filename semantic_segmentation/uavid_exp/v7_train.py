"""Apply native crops and inference for MaxViT or DINOv2-L."""
import argparse,sys
import v2_train,v3_train
from v5_data import DinoImages
from v7_data import MaxVitImages
from v7_infer import probabilities
if __name__=='__main__':
 p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--model',required=True);a,_=p.parse_known_args()
 v3_train.HardNegativeImages=MaxVitImages if a.model=='unet_maxvit_small' else DinoImages
 v2_train.probabilities=probabilities
 v3_train.main()
