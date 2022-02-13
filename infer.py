import torch
from model import FluxFormer_v2
from torch.utils.data import DataLoader
from dataset import Fluxnet
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from loss import r2_loss, corr_coeff, MAELoss
from tqdm import tqdm


if __name__ == "__main__":
    model_path = "checkpoints/fluxformer_60.pth"
    model = FluxFormer_v2()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    rootdir = "/home/zmj/FluxFormer/data_corrected"
    val_split = join("/home/zmj/FluxFormer/split_corrected", "infer.txt")
    val_dataset = Fluxnet(root_dir=rootdir, split_file=val_split, is_train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    maeloss = MAELoss()
    with torch.no_grad():
        for sample in tqdm(val_dataloader):
            site = sample["site_name"][0]
            date = sample["site_date"][0]
            data = sample['inputs']
            le_all = sample["le_all"]
            sparse = sample['sparse']
            sparse_max = sample["sparse_max"]
            sparse_min = sample["sparse_min"]
            sparse = (sparse - sparse_min) / (sparse_max - sparse_min + 1e-2)
            normed_output = model(data, sparse)
            output = normed_output * (sparse_max - sparse_min) + sparse_min
            pred_le_torch = torch.unsqueeze(output[0, :, -1], -1)
            pred_le = pred_le_torch.detach().cpu().numpy()[:, 0]
            target_le_torch = torch.unsqueeze(le_all[0, :], -1)
            target_le = target_le_torch.detach().cpu().numpy()[:, 0]
            metric = dict()
            metric["r2"] = r2_loss(pred_le_torch, target_le_torch).detach().cpu().numpy()
            metric["corr"] = corr_coeff(pred_le_torch, target_le_torch).detach().cpu().numpy()
            metric["mae"] = maeloss(pred_le_torch, target_le_torch).detach().cpu().numpy()
            time_hour = np.arange(0, 24, 0.5)
            plt.plot(time_hour, pred_le, label="pred")
            plt.plot(time_hour, target_le, label="target")
            plt.legend()
            title = ""
            for key, value in metric.items():
                title += "{}: {:.2f}, ".format(key, value)
            plt.title(title)
            plt.savefig(f"tmp/{site}_{date}.png")
            plt.close()
