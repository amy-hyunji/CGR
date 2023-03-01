import torch.nn as nn

class Similarity(nn.Module):
   def __init__(self, temp):
      super().__init__()
      self.temp = temp 
      self.cos = nn.CosineSimilarity(dim=-1)
   def forward(self, x, y):
      return self.cos(x, y)/self.temp

def np_only(hparams):
    if hparams.model_type in ["hyper-mem-np-only", "hyper-wo-vd"]:
       return True 
    else:
       return False
