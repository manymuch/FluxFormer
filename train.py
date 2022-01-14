import argparse
from os.path import join
import torch
from loss import r2_loss, corr_coeff, MAELoss
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Fluxnet
from model import FluxFormer, RegNet_v2
import wandb
wandb.init(project="FluxNet",
           entity="jiaxin-zhang",
           name="fluxnet_transformer_mae_corrected",
           tags=["transformer", "corrected", "mae", "gauss_norm", "target_relative"])


def log_error(args, prefix, normed_output, sample, epoch):
    metrics = {}
    with torch.no_grad():
        sparse = sample['sparse']
        target = sample['target']
        sparse_max = sample["sparse_max"]
        sparse_min = sample["sparse_min"]
        output = normed_output * (sparse_max - sparse_min) + sparse_min
        normed_sparse = (sparse - sparse_min) / (sparse_max - sparse_min + 1e-2)
        normed_target = (target - sparse_min) / (sparse_max - sparse_min + 1e-2)
        naive_output = torch.mean(sparse, dim=1, keepdim=True)
        normed_naive_output = torch.mean(normed_sparse, dim=1, keepdim=True)
        metrics["{}_naive_mae".format(prefix)] = torch.mean(torch.abs(naive_output - target))
        metrics["{}_naive_r2".format(prefix)] = r2_loss(naive_output, target)
        metrics["{}_naive_corr".format(prefix)] = corr_coeff(naive_output, target)

        metrics["{}_mae".format(prefix)] = torch.mean(torch.abs(output - target))
        metrics["{}_r2".format(prefix)] = r2_loss(output, target)
        metrics["{}_corr".format(prefix)] = corr_coeff(output, target)

        metrics["{}_normed_mae".format(prefix)] = torch.mean(torch.abs(normed_output - normed_target))
        metrics["{}_normed_r2".format(prefix)] = r2_loss(normed_output, normed_target)
        metrics["{}_normed_corr".format(prefix)] = corr_coeff(normed_output, normed_target)

        metrics["{}_normed_naive_mae".format(prefix)] = torch.mean(torch.abs(normed_naive_output - normed_target))
        metrics["{}_normed_naive_r2".format(prefix)] = r2_loss(normed_naive_output, normed_target)
        metrics["{}_normed_naive_corr".format(prefix)] = corr_coeff(normed_naive_output, normed_target)
    wandb.log(metrics, step=epoch)


def train(args, model, device, train_loader, optimizer, epoch, loss_function):
    model.train()
    for sample in train_loader:
        data = sample['inputs']
        sparse = sample['sparse']
        target = sample['target']
        sparse_max = sample["sparse_max"]
        sparse_min = sample["sparse_min"]
        optimizer.zero_grad()
        # nomalization for LE
        sparse = (sparse - sparse_min) / (sparse_max - sparse_min + 1e-2)
        target = (target - sparse_min) / (sparse_max - sparse_min + 1e-2)
        normed_output = model(data, sparse)
        loss = loss_function(normed_output, target)
        loss.backward()
        optimizer.step()
    log_error(args, "train", normed_output, sample, epoch)


def test(args, model, device, test_loader, epoch):
    model.eval()
    normed_output_list = []
    sample_list = {}
    sample_list["sparse"] = []
    sample_list["target"] = []
    sample_list["sparse_max"] = []
    sample_list["sparse_min"] = []
    with torch.no_grad():
        for sample in test_loader:
            data = sample['inputs']
            sparse = sample['sparse']
            target = sample['target']
            sparse_max = sample["sparse_max"]
            sparse_min = sample["sparse_min"]
            # nomalization for LE
            sparse = (sparse - sparse_min) / (sparse_max - sparse_min + 1e-2)
            target = (target - sparse_min) / (sparse_max - sparse_min + 1e-2)
            normed_output = model(data, sparse)
            sample_list["sparse"].append(sparse)
            sample_list["target"].append(target)
            sample_list["sparse_max"].append(sparse_max)
            sample_list["sparse_min"].append(sparse_min)
            normed_output_list.append(normed_output)
    sample_list["sparse"] = torch.cat(sample_list["sparse"], dim=0)
    sample_list["target"] = torch.cat(sample_list["target"], dim=0)
    sample_list["sparse_max"] = torch.cat(sample_list["sparse_max"], dim=0)
    sample_list["sparse_min"] = torch.cat(sample_list["sparse_min"], dim=0)
    normed_output_list = torch.cat(normed_output_list, dim=0)
    log_error(args, "test", normed_output, sample, epoch)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch FLUXNET')
    parser.add_argument('--batchsize', type=int, default=128, metavar='N')
    parser.add_argument('--epochs', type=int, default=100, metavar='N')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
    parser.add_argument('--lr_gamma', type=float, default=0.2, metavar='LRG')
    parser.add_argument('--lr_milestone', nargs="+", type=int, default=[30, 60], metavar='LRM')
    parser.add_argument('--train_split', type=str, default="train.txt")
    parser.add_argument('--test_split', type=str, default="val.txt")
    args = parser.parse_args()
    wandb.config.update(args)
    device = torch.device("cpu")

    rootdir = "/home/zmj/FluxFormer/data_corrected"
    train_split = join("/home/zmj/FluxFormer/split_corrected", args.train_split)
    train_dataset = Fluxnet(root_dir=rootdir, split_file=train_split, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=2, drop_last=True)

    val_split = join("/home/zmj/FluxFormer/split_corrected", args.test_split)
    val_dataset = Fluxnet(root_dir=rootdir, split_file=val_split, is_train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=2, drop_last=True)

    model = FluxFormer()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestone, gamma=args.lr_gamma)
    loss_function = MAELoss()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr, gamma=0.95)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_dataloader, optimizer, epoch, loss_function)
        scheduler.step()
        print("epoch = {}".format(epoch))
        # scheduler.step()
        if epoch % 5 == 0:
            test(args, model, device, val_dataloader, epoch)
            ckpt_path = "/home/zmj/FluxFormer/checkpoints/regnet_{}.pth".format(epoch)
            torch.save(model.state_dict(), ckpt_path)


if __name__ == '__main__':
    main()
