import argparse
from os.path import join
import pathlib
import torch
from loss import r2_loss, corr_coeff, MAELoss
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Fluxnet
from model import FluxAttention
import wandb
from tqdm import tqdm
wandb.init(project="FluxNet",
           entity="jiaxin-zhang",
           name="FluxFormer_small_1e5",
           tags=[])


def log_error(args, prefix, normed_output, sample, target, epoch):
    metrics = {}
    with torch.no_grad():
        sparse = sample['sparse'].cpu()
        sparse_max = sample["sparse_max"].cpu()
        sparse_min = sample["sparse_min"].cpu()
        normed_output = normed_output.cpu()
        target = target.cpu()
        output = normed_output * (sparse_max - sparse_min) + sparse_min
        normed_sparse = (sparse - sparse_min) / \
            (sparse_max - sparse_min + 1e-2)
        normed_target = (target - sparse_min) / \
            (sparse_max - sparse_min + 1e-2)
        naive_output = torch.mean(sparse, dim=1, keepdim=True)
        normed_naive_output = torch.mean(normed_sparse, dim=1, keepdim=True)
        metrics["{}_naive_mae".format(prefix)] = torch.mean(
            torch.abs(naive_output - target))
        metrics["{}_naive_r2".format(prefix)] = r2_loss(naive_output, target)
        metrics["{}_naive_corr".format(prefix)] = corr_coeff(
            naive_output, target)

        metrics["{}_mae".format(prefix)] = torch.mean(
            torch.abs(output - target))
        metrics["{}_r2".format(prefix)] = r2_loss(output, target)
        metrics["{}_corr".format(prefix)] = corr_coeff(output, target)

        metrics["{}_normed_mae".format(prefix)] = torch.mean(
            torch.abs(normed_output - normed_target))
        metrics["{}_normed_r2".format(prefix)] = r2_loss(
            normed_output, normed_target)
        metrics["{}_normed_corr".format(prefix)] = corr_coeff(
            normed_output, normed_target)

        metrics["{}_normed_naive_mae".format(prefix)] = torch.mean(
            torch.abs(normed_naive_output - normed_target))
        metrics["{}_normed_naive_r2".format(prefix)] = r2_loss(
            normed_naive_output, normed_target)
        metrics["{}_normed_naive_corr".format(prefix)] = corr_coeff(
            normed_naive_output, normed_target)
    wandb.log(metrics, step=epoch)


def train(args, model, device, train_loader, optimizer, epoch, loss_function):
    model.train()
    for sample in train_loader:
        data = sample['inputs'].to(device)
        le_all = sample["le_all"].to(device)
        sparse = sample['sparse'].to(device)
        sparse_max = sample["sparse_max"].to(device)
        sparse_min = sample["sparse_min"].to(device)
        optimizer.zero_grad()
        # nomalization for LE
        sparse = (sparse - sparse_min) / (sparse_max - sparse_min + 1e-2)
        target = torch.mean(le_all, dim=1, keepdim=True)
        normed_target = (target - sparse_min) / \
            (sparse_max - sparse_min + 1e-2)
        normed_output = model(data, sparse)
        loss = loss_function(normed_output, normed_target)
        loss.backward()
        optimizer.step()
    log_error(args, "train", normed_output, sample, target, epoch)


def test(args, model, device, test_loader, epoch):
    model.eval()
    normed_output_list = []
    target_list = []
    sample_list = {}
    sample_list["sparse"] = []
    sample_list["sparse_max"] = []
    sample_list["sparse_min"] = []
    with torch.no_grad():
        for sample in test_loader:
            data = sample['inputs'].to(device)
            le_all = sample["le_all"].to(device)
            sparse = sample['sparse'].to(device)
            sparse_max = sample["sparse_max"].to(device)
            sparse_min = sample["sparse_min"].to(device)
            # nomalization for LE
            sparse = (sparse - sparse_min) / (sparse_max - sparse_min + 1e-2)
            target = torch.mean(le_all, dim=1, keepdim=True)
            normed_output = model(data, sparse)
            normed_output_list.append(normed_output)
            sample_list["sparse"].append(sparse)
            sample_list["sparse_max"].append(sparse_max)
            sample_list["sparse_min"].append(sparse_min)
            target_list.append(target)
        normed_output_list = torch.cat(normed_output_list, dim=0)
        target_list = torch.cat(target_list, dim=0)
        sample_list["sparse"] = torch.cat(sample_list["sparse"], dim=0)
        sample_list["sparse_max"] = torch.cat(sample_list["sparse_max"], dim=0)
        sample_list["sparse_min"] = torch.cat(sample_list["sparse_min"], dim=0)
    log_error(args, "test", normed_output_list,
              sample_list, target_list, epoch)


def read_split(split_file):
    with open(split_file, "r") as f:
        lines = f.readlines()
    return lines


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch FLUXNET')
    parser.add_argument('--batchsize', type=int, default=128, metavar='N')
    parser.add_argument('--epochs', type=int, default=60, metavar='N')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR')
    parser.add_argument('--lr_gamma', type=float, default=0.2, metavar='LRG')
    parser.add_argument('--lr_milestone', nargs="+",
                        type=int, default=[50, 55], metavar='LRM')
    parser.add_argument('--train_split', type=str, default="train.txt")
    parser.add_argument('--test_split', type=str, default="val.txt")
    parser.add_argument('--d_model', type=int, default=32)  # 36
    parser.add_argument('--n_head', type=int, default=4)  # 4
    parser.add_argument('--n_layer', type=int, default=2)  # 2

    args = parser.parse_args()
    wandb.config.update(args)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    rootdir = "/home/mijun/Code/jiaxin/FluxFormer"
    data_dir = join(rootdir, "data")
    train_split = read_split(join(rootdir, "split", args.train_split))
    print("Loading training data...")
    train_dataset = Fluxnet(
        root_dir=data_dir, site_csv_names=train_split, is_train=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=2, drop_last=True)

    val_split = read_split(join(rootdir, "split", args.test_split))
    print("Loading validation data...")
    val_dataset = Fluxnet(
        root_dir=data_dir, site_csv_names=val_split, is_train=False)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=2, drop_last=True)

    model = FluxAttention(d_model=args.d_model, n_head=args.n_head, n_layer=args.n_layer)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestone, gamma=args.lr_gamma)
    loss_function = MAELoss()
    print("Start training...")
    for epoch in tqdm(range(1, args.epochs + 1)):
        train(args, model, device, train_dataloader,
              optimizer, epoch, loss_function)
        scheduler.step()
        if epoch % 5 == 0:
            test(args, model, device, val_dataloader, epoch)
    # save_model
    pathlib.Path("checkpoints").mkdir(parents=True, exist_ok=True)
    ckpt_path = "checkpoints/fluxAttention.pth"
    torch.save(model.state_dict(), ckpt_path)


if __name__ == '__main__':
    main()
