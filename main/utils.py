########################
# Utils Functions
########################

import torch
from dataclasses import dataclass
 
torch.set_printoptions(precision=4, threshold=10)


@dataclass
class WorkArgs:
    level: int
    problem_id: int # logically indexed
    sample_id: int




