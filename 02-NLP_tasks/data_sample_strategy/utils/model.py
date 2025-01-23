import logging
import torch
from torch import nn
import time

logger = logging.getLogger(__name__)



class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.linear1 = nn.Linear(2, 50)
        self.linear2 = nn.Linear(50, 2)

    def forward(self, batch):
        batch = batch.to(self.linear1.weight.device).float()
        print(batch)
        # time.sleep(1)
        res = self.linear1(batch)
        res = self.linear2(res)
        res = res.mean()
        return res
    
    def save_pretrained(self, output : str):
        print(f"save model to {output}")
