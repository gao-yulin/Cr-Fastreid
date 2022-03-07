import torch
import torch.nn as nn

from .build import BACKBONE_REGISTRY


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features=None, bn=True, act=nn.ReLU):
        super().__init__()
        out_features = out_features or in_features
        self.bn = nn.BatchNorm1d(out_features) if bn is True else None
        self.fc = nn.Linear(in_features, out_features, bias=self.bn is None)
        self.act = act() if act is not None else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        print("total dims of x is {}".format(x.ndim))
        print("shape[0] is {}".format(x.shape[0]))
        print("shape[1] is {}".format(x.shape[1]))
        print("shape[2] is {}".format(x.shape[2]))
        print("shape[3] is {}".format(x.shape[3]))
        assert x.ndim == 4 and x.shape[2] == x.shape[3] == 1
        return x[:, :, 0, 0]


class Unsqueeze(nn.Module):
    def __init__(self):
        super(Unsqueeze, self).__init__()

    def forward(self, x):
        assert x.ndim == 2
        return x[:, :, None, None]


@BACKBONE_REGISTRY.register()
def build_mlp_backbone(cfg):
    feat_channels = cfg.MODEL.BACKBONE.FEAT_DIM
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN
    last_stride = cfg.MODEL.BACKBONE.LAST_STRIDE
    if cfg.MODEL.BACKBONE.NORM == 'BN':
        bn_norm = True
    else:
        raise NotImplementedError
    depth = int(cfg.MODEL.BACKBONE.DEPTH[:-1])
    assert last_stride == 1
    if pretrain is True:
        raise NotImplementedError

    model = nn.Sequential(
        Squeeze(),
        *[MLPBlock(
            feat_channels, bn=bn_norm, act=nn.ReLU if idx != depth - 1 else None
        ) for idx in range(depth)],
        Unsqueeze()
    )
    return model
