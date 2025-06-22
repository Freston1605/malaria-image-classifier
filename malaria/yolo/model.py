from ultralytics import YOLO
from malaria.model import BaseLitModel
import torch.nn as nn


class YOLOLitModel(BaseLitModel):
    def __init__(self, model_path=None, lr=1e-3, num_classes=2, freeze_backbone=True):
        super().__init__(num_classes=num_classes, lr=lr)
        self.save_hyperparameters()

        # Load the pretrained YOLO model and extract only the model (nn.Module)
        yolo_model = YOLO(model_path) if model_path else YOLO("yolo11n-cls.pt")
        self.model = yolo_model.model  # discard the outer YOLO wrapper!

        # Replace classifier head
        classify_block = self.model.model[10]
        if isinstance(classify_block, nn.Module) and hasattr(classify_block, "linear"):
            in_features = classify_block.linear.in_features
            classify_block.linear = nn.Linear(in_features, num_classes)
        else:
            raise AttributeError("Expected 'linear' in final classify block.")

        # Optionally freeze all except final head
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "model.10.linear" not in name:
                    param.requires_grad = False

    def forward(self, x):
        out = self.model(x)
        # print(f"Model output type: {type(out)}, shape: {getattr(out, 'shape', 'n/a')}")
        return out[0] if isinstance(out, tuple) else out

