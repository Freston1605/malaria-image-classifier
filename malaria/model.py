import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F

class MalariaLitModel(L.LightningModule):
    """
    PyTorch Lightning Module for malaria cell classification using a simple CNN.
    Handles model definition, training, validation, and test steps.
    """

    def __init__(self, num_classes=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        # CNN architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, *args, **kwargs):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Output tensor after classification layer (e.g., logits or class scores).
                - Embedding tensor representing the feature vector before classification.
        """

        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        embedding = x
        x = self.classifier(x)
        return x, embedding

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step on a given batch.
        Args:
            batch (Tuple[Tensor, Tensor]): A tuple containing input data (x) and target labels (y).
            batch_idx (int): Index of the current batch.
        Returns:
            Tensor: The computed loss for the current batch.
        Logs:
            - 'train_loss': The loss value for the current batch (logged per epoch).
            - 'train_acc': The accuracy for the current batch (logged per epoch).
        """
        
        x, y = batch
        logits, _ = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch):
        """
        Performs a single test step on a batch of data.

        Unpacks the input batch, handling cases where the batch contains either two or three elements.
        If three elements are present, the third (typically labels) is ignored. Passes the input data
        through the model to obtain logits, computes the predicted class indices, and returns a dictionary
        containing image names and their corresponding predictions.

        Args:
            batch (tuple): A tuple containing the batch data. Expected to be either (x, names) or (x, names, _).

        Returns:
            dict: A dictionary with keys:
                - 'img_names': The names/identifiers of the images in the batch.
                - 'preds': The predicted class indices for each image.

        Raises:
            ValueError: If the batch does not have 2 or 3 elements.
        """

        # Adjust unpacking based on your test dataloader's output
        if len(batch) == 2:
            x, names = batch
        elif len(batch) == 3:
            x, names, _ = batch  # ignore labels if present
        else:
            raise ValueError("Unexpected batch format in test_step")
        logits, _ = self(x)
        preds = logits.argmax(dim=1)
        return {'img_names': names, 'preds': preds}

    def configure_optimizers(self):
        """
        Configures and returns the optimizer for training the model.
        Returns:
            torch.optim.Optimizer: An Adam optimizer initialized with the model's parameters and the learning rate specified in self.hparams.lr.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
