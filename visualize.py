import torch
from model import FluxFormer_v2
from torch.utils.data import DataLoader
from dataset import Fluxnet
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from loss import r2_loss, corr_coeff, MAELoss
from tqdm import tqdm

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


model_path = "checkpoints/fluxformer_60.pth"
model = FluxFormer_v2()
model.load_state_dict(torch.load(model_path))
model.eval()
# model.transformer.encoder.layers[0].self_attn.out_proj.register_forward_hook(get_activation('fc3'))
model.transformer.encoder.layers[0].self_attn.register_forward_hook(get_activation('fc3'))
data = torch.zeros((1, 8, 7), dtype=torch.float32)
sparse = torch.zeros((1, 4), dtype=torch.float32)
output = model(data, sparse)
print(activation['fc3'].shape)
