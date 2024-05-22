import argparse
from os.path import join
import pathlib
import torch
from model import FluxAttention
from torch.utils.data import DataLoader
from dataset import Fluxnet
import numpy as np
import cv2
import pandas as pd


class FluxFormerInfer:
    def __init__(self, model_path, data_dir, vis_attention=False):
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = FluxAttention()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.vis_attention = vis_attention
        if vis_attention:
            self.model.decoder.layers[0].multihead_attn.register_forward_hook(
                self.attention_hook)
        self.data_dir = data_dir

    def attention_hook(self, model, input, output):
        # print("query:", input[0].shape)
        # print("key:", input[1].shape)
        # print("attention:", output[0].shape)
        # print("attention weights:", output[1].shape)
        attention_weight = output[1][0].detach().cpu().numpy()
        self.attention_map.append(attention_weight)

    def visualize_attention(self, attention_weights):
        attention_day = attention_weights.mean(axis=0)
        attention_day = attention_day.reshape(7, 8)
        attention_day = np.abs(attention_day)
        normed_attention = (attention_day - np.min(attention_day)) / \
            (np.max(attention_day) - np.min(attention_day) + 1e-3)
        heatmap = cv2.applyColorMap(
            np.uint8(255 * normed_attention), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (0, 0), fx=16, fy=16,
                             interpolation=cv2.INTER_NEAREST)
        return heatmap

    def infer_site(self, split):
        self.attention_map = []
        dataset = Fluxnet(root_dir=self.data_dir,
                          site_csv_names=[split], is_train=False)
        loader = DataLoader(dataset, batch_size=4, shuffle=False,
                            num_workers=0, drop_last=True)
        # start inference
        df = pd.DataFrame(columns=["date", "LE_predict",
                                   "LE_measure", "norm_LE_predict", "norm_LE_measure"])
        with torch.no_grad():
            for sample in loader:
                data = sample['inputs'].to(self.device)
                le_all = sample["le_all"].to(self.device)
                sparse = sample['sparse'].to(self.device)
                sparse_max = sample["sparse_max"].to(self.device)
                sparse_min = sample["sparse_min"].to(self.device)
                site_name = sample["site_name"]
                site_date = sample["site_date"]

                # nomalization for LE
                naive_output = torch.mean(sparse, dim=1).squeeze()
                sparse = (sparse - sparse_min) / \
                    (sparse_max - sparse_min + 1e-2)

                target = torch.mean(le_all, dim=1, keepdim=True)
                normed_target = (target - sparse_min) / \
                    (sparse_max - sparse_min + 1e-2)
                normed_output = self.model(data, sparse)
                output = (normed_output * (sparse_max -
                                           sparse_min) + sparse_min).squeeze()
                output = output.detach().cpu().numpy()
                target = target.detach().cpu().numpy().squeeze()
                naive_output = naive_output.detach().cpu().numpy()
                normed_output = normed_output.squeeze().detach().cpu().numpy()
                normed_target = normed_target.squeeze().detach().cpu().numpy()

                # Create a new DataFrame for concatenation
                new_data = pd.DataFrame({
                    "date": site_date,
                    "LE_predict": output,
                    "LE_measure": target,
                    "norm_LE_predict": normed_output,
                    "norm_LE_measure": normed_target
                })
                new_data = new_data.dropna(axis=1, how='all')
                df = df.dropna(axis=1, how='all')
                df = pd.concat([df, new_data], ignore_index=True)

        df = df.sort_values(by="date")
        if self.vis_attention:
            attention_map = np.concatenate(self.attention_map, axis=0)
            return df, attention_map
        else:
            return df, None

    def add_texts(self, image):
        row_texts = ['VPD', 'TA', 'PA', 'WS', 'P', 'LW_IN', 'SW_IN']
        col_texts = ['1:30', '4:30', '7:30', '10:30',
                     '13:30', '16:30', '19:30', '22:30']

        h, w, _ = image.shape
        image = np.concatenate(
            [np.ones((h, w//2, 3), dtype=np.uint8) * 255, image], axis=1)

        for id, text in enumerate(row_texts):
            cv2.putText(image, text, (10, 12+16*id),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        h, w, _ = image.shape
        white_space = np.ones((w, h//2, 3), dtype=np.uint8) * 255
        for id, text in enumerate(col_texts):
            cv2.putText(white_space, text, (5, 75+16*id),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        white_space = cv2.rotate(white_space, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = np.concatenate([white_space, image], axis=0)
        return image

    def get_statistic(self, prediction, measurement):
        residual = prediction - measurement
        residual_square = residual**2
        R2 = 1 - residual_square.sum() / measurement.var() / \
            measurement.shape[0]
        RMSE = residual_square.mean()**0.5
        MB = residual.mean()
        result = {"R2": R2, "RMSE": RMSE, "MB": MB}
        return result


def main():
    parser = argparse.ArgumentParser(description='PyTorch FLUXNET')
    parser.add_argument('--vis', default=False,
                        action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    # Load model
    model_path = "checkpoints/fluxAttention.pth"
    # Prepare dataset
    rootdir = "/home/mijun/Code/jiaxin/FluxFormer"
    data_dir = join(rootdir, "data")
    output_dir = join(rootdir, "output")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    splits = ["FLX_IT-SR2.csv",
              # "FLX_AU-Ade.csv",
              # "FLX_CA-Obs.csv",
              # "FLX_DE-Gri.csv",
              # "FLX_IT-Col.csv",
              # "FLX_US-ARM.csv",
              # "FLX_US-WCr.csv",
              # "FLX_RU-Cok.csv",
              # "FLX_JP-SMF.csv",
              # "FLX_CH-Fru.csv",
              # "FLX_FR-LBr.csv"
              ]

    vis_attention = args.vis
    ffi = FluxFormerInfer(model_path, data_dir, vis_attention=vis_attention)
    for split in splits:
        site_name = split.split(".")[0]

        df, attention_map = ffi.infer_site(split)

        # get statistic
        prediction = df["LE_predict"].values
        measurement = df["LE_measure"].values
        statistic = ffi.get_statistic(prediction, measurement)

        # vis attention
        if vis_attention:
            attention_heatmap = ffi.visualize_attention(attention_map)
            attention_heatmap = ffi.add_texts(attention_heatmap)
            heatmap_name = join(output_dir, f"{site_name}.png")
            cv2.imwrite(heatmap_name, attention_heatmap)

        # save csv
        csv_name = join(output_dir, split)
        df["R2"] = ""
        df["RMSE"] = ""
        df["MB"] = ""
        df.loc[0, "R2"] = statistic["R2"]
        df.loc[0, "RMSE"] = statistic["RMSE"]
        df.loc[0, "MB"] = statistic["MB"]
        df.to_csv(csv_name, index=False)
        print(
            f"{site_name}, R2: {statistic['R2']:.02f}, RMSE: {statistic['RMSE']:.02f}, MB: {statistic['MB']:.02f}")


if __name__ == '__main__':
    main()
