import math
import torch
from torch import nn
import torch.nn.functional as F

def load_state(path, key=None):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if key is not None:
        state = state[key]
    return {name.replace('_orig_mod.', ''): value for name, value in state.items()}

class HostDominantGatedFusion(nn.Module):

    def __init__(self, dim=256, max_image_gate=0.35, initial_gate=0.18):
        super().__init__()
        self.max_image_gate = float(max_image_gate)
        probability = initial_gate / max_image_gate
        self.gate_logit = nn.Parameter(torch.tensor(math.log(probability / (1 - probability))))
        self.interaction = nn.Sequential(nn.Linear(dim * 3, dim * 2), nn.LayerNorm(dim * 2), nn.GELU(), nn.Dropout(0.15), nn.Linear(dim * 2, dim), nn.Tanh())

    def gate(self):
        return self.max_image_gate * torch.sigmoid(self.gate_logit)

    def forward(self, vector, image):
        if vector.dim() == 3:
            vector = vector.mean(1)
        if image.dim() == 3:
            image = image.mean(1)
        vector = F.normalize(vector, dim=1)
        image = F.normalize(image, dim=1)
        correction = self.interaction(torch.cat([vector, image, vector * image], dim=1))
        return F.normalize(vector + self.gate() * correction, dim=1)
