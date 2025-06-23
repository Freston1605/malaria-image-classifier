import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from unittest import mock

from malaria.cnn.model import MalariaLitModel
from malaria.yolo import model as yolo_module


class DummyClassifyBlock(nn.Module):
    def __init__(self, in_features=3 * 64 * 64, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)


class DummyInnerModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        blocks = [nn.Identity() for _ in range(10)]
        blocks.append(DummyClassifyBlock(num_classes=num_classes))
        self.model = nn.ModuleList(blocks)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model[10](x)


class DummyYOLO:
    def __init__(self, *args, **kwargs):
        self.model = DummyInnerModel()


def test_cnn_forward():
    model = MalariaLitModel(num_classes=2)
    x = torch.randn(2, 3, 64, 64)
    logits, embedding = model(x)
    assert logits.shape == (2, 2)
    assert embedding.shape[0] == 2


@mock.patch.object(yolo_module, "YOLO", DummyYOLO)
def test_yolo_forward():
    model = yolo_module.YOLOLitModel(num_classes=2, freeze_backbone=True)
    x = torch.randn(2, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (2, 2)
