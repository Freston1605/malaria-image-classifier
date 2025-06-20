from ultralytics import YOLO

from malaria.model import BaseLitModel


class YOLOLitModel(BaseLitModel):
    def __init__(self, model_path=None, lr=1e-3, num_classes=2):
        super().__init__()
        self.save_hyperparameters()
        self.model = YOLO(model_path) if model_path else YOLO("yolo11n-cls.pt")
        self.num_classes = num_classes
        self.lr = lr

    def forward(self, x):
        # Pass the input through the model
        return self.model(x)
