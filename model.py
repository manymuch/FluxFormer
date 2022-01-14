import torch.nn as nn
import torch
import torchvision.models as models


class FluxFormer(nn.Module):
    def __init__(self):
        super(FluxFormer, self).__init__()
        self.transformer = nn.Transformer(nhead=8, num_encoder_layers=6, d_model=8, batch_first=True)

    def forward(self, data, sparse):
        up_sparse = torch.zeros((data.shape[0], 48, 1), dtype=sparse.dtype, device=sparse.device)
        up_sparse[:, [4, 22, 28, 47], 0] = sparse
        feature = torch.cat((data, up_sparse), dim=2)
        # feature = self.linear_encoder(data)
        output = self.transformer(feature, feature)
        mean_output = torch.mean(output[:, :, -1], dim=1, keepdim=True)
        return mean_output


class RegNet(nn.Module):
    def __init__(self):
        super(RegNet, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(512, 1)
        self.sparse_upsample = nn.Linear(4, 48)
        self.linear_in = nn.Linear(1, 7)

    def forward(self, data, sparse):
        data = torch.unsqueeze(torch.swapaxes(data, 1, 2), -1)
        up_sparse = torch.unsqueeze(torch.unsqueeze(self.sparse_upsample(sparse), 1), -1)
        input = torch.cat((data, up_sparse), dim=1)
        pseudo_image = self.linear_in(input)
        output = self.backbone(pseudo_image)
        return output


class RegNet_v2(nn.Module):
    def __init__(self):
        super(RegNet_v2, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(512, 1)

    def forward(self, data, sparse):
        data = torch.unsqueeze(torch.swapaxes(data, 1, 2), -1)
        up_sparse = torch.zeros((data.shape[0], 1, 48, 1), dtype=sparse.dtype, device=sparse.device)
        up_sparse[:, 0, [4, 22, 28, 47], 0] = sparse
        pseudo_image = torch.cat((data, up_sparse), dim=1)
        output = self.backbone(pseudo_image)
        return output
