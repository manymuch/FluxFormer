from os.path import join
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
pd.options.mode.chained_assignment = None


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-5)


def zmean(x):
    return x - x.mean()


class Fluxnet(Dataset):
    """Fluxnet dataset."""

    def __init__(self, root_dir, split_file, is_train=True, transform=None):
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        with open(split_file, "r") as f:
            lines = f.readlines()
        self.samples = []
        self.ground_truth = []
        self.sparse = []
        self.sparse_max = []
        self.sparse_min = []
        print("Loading data...")
        for line in lines:
            print(line)
            csv_path = join(root_dir, line.strip())
            site = pd.read_csv(csv_path)
            dates = sorted(site['Date'].unique())
            for date in dates:
                site_date = site[site['Date'] == date]
                inputs = site_date[['VPD_F', 'TA_F', 'PA_F', 'WS_F', 'P_F', 'LW_IN_F', 'SW_IN_F']]
                inputs = self.gaussify_input(inputs).to_numpy()
                sparse = site_date["LE_F_MDS"].to_numpy()[[4, 22, 28, 47]]
                self.samples.append(inputs.astype(np.float32))
                sparse = sparse.astype(np.float32)
                max_LE = np.asarray([np.max(sparse)])
                min_LE = np.asarray([np.min(sparse)])
                self.sparse_max.append(max_LE)
                self.sparse_min.append(min_LE)
                # sparse = (sparse - min_LE) / ((max_LE - min_LE) + 1e-2)
                self.sparse.append(sparse)
                target_array = site_date["LE_F_MDS"].to_numpy()
                # target_array = (target_array - min_LE) / ((max_LE - min_LE) + 1e-2)
                target = target_array.mean().astype(np.float32)
                self.ground_truth.append(target)

    def normalize_input(self, inputs):
        inputs.loc[:, "VPD_F"] = (np.clip(inputs["VPD_F"].to_numpy(), 0, 100))/100
        inputs.loc[:, "WS_F"] = (np.clip(inputs["WS_F"].to_numpy(), 0, 100))/100
        inputs.loc[:, "P_F"] = (np.clip(inputs["P_F"].to_numpy(), 0, 10))/10
        inputs.loc[:, "LW_IN_F"] = (np.clip(inputs["LW_IN_F"].to_numpy(), 0, 1000))/1000
        inputs.loc[:, "SW_IN_F"] = (np.clip(inputs["SW_IN_F"].to_numpy(), 0, 2000))/2000
        inputs.loc[:, "TA_F"] = (np.clip(inputs["TA_F"].to_numpy(), -70, 70) + 70)/140
        inputs.loc[:, "PA_F"] = (np.clip(inputs["PA_F"].to_numpy(), 0, 102))/102
        return inputs

    def gaussify_input(self, inputs):
        # normalize inputs by its mean and std
        inputs.loc[:, "VPD_F"] = zscore(inputs["VPD_F"].to_numpy())
        inputs.loc[:, "WS_F"] = zscore(inputs["WS_F"].to_numpy())
        inputs.loc[:, "P_F"] = zscore(inputs["P_F"].to_numpy())
        inputs.loc[:, "LW_IN_F"] = zscore(inputs["LW_IN_F"].to_numpy())
        inputs.loc[:, "SW_IN_F"] = zscore(inputs["SW_IN_F"].to_numpy())
        inputs.loc[:, "TA_F"] = zscore(inputs["TA_F"].to_numpy())
        inputs.loc[:, "PA_F"] = zscore(inputs["PA_F"].to_numpy())
        return inputs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = {}
        sample['inputs'] = self.samples[idx]
        sample['target'] = np.expand_dims(np.asarray(self.ground_truth[idx]), 0)
        sample['sparse'] = self.sparse[idx]
        sample['sparse_max'] = self.sparse_max[idx]
        sample['sparse_min'] = self.sparse_min[idx]
        return sample


if __name__ == '__main__':
    rootdir = "/home/zmj/FluxFormer/data"
    slit_file = "/home/zmj/FluxFormer/split/debug.txt"
    dataset = Fluxnet(root_dir=rootdir, split_file=slit_file, is_train=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)
    print(len(dataset))
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['inputs'].dtype, sample_batched['target'])
        exit()
