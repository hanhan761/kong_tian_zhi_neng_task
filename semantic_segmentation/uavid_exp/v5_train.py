"""Run the nested trainer on native DINO image crops."""
import v2_train,v3_train
from v5_data import DinoImages
from v5_infer import probabilities
if __name__=='__main__':
    v3_train.HardNegativeImages=DinoImages
    v2_train.probabilities=probabilities
    v3_train.main()
