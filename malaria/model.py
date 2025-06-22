import lightning.pytorch as L
import torch
from torch import nn


class BaseLitModel(L.LightningModule):
    """
    Base PyTorch Lightning Module for malaria cell classification and similar tasks.
    Provides common methods for loss computation, accuracy calculation, optimizer configuration,
    and logging. Subclasses should implement the `forward` method.
    """

    def __init__(self, num_classes=2, lr=1e-3):
        """
        Args:
            num_classes (int): Number of output classes.
            lr (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()


    def compute_loss(self, logits, targets):
        """
        Computes the cross-entropy loss between logits and targets.
        Args:
            logits (torch.Tensor): Model output of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth class indices of shape (batch_size,).
        Returns:
            torch.Tensor: Scalar loss tensor.
        """

        return self.criterion(logits, targets)


    def compute_accuracy(self, logits, targets):
        """
        Computes the accuracy of model predictions.
        Args:
            logits (torch.Tensor): Model output of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth class indices of shape (batch_size,).
        Returns:
            torch.Tensor: Scalar tensor with accuracy value.
        """

        preds = logits.argmax(dim=1)
        return (preds == targets).float().mean()


    def log_metrics(self, loss, acc, prefix="val"):
        """
        Logs loss and accuracy metrics.
        Args:
            loss (torch.Tensor): Loss value.
            acc (torch.Tensor): Accuracy value.
            prefix (str): Prefix for metric names ('train' or 'val').
        """

        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)


    def training_step(self, batch, batch_idx):
        """
        Training step for a batch.
        """

        images, targets = batch
        logits = self(images)
        loss = self.compute_loss(logits, targets)
        acc = self.compute_accuracy(logits, targets)
        self.log_metrics(loss, acc, prefix="train")
        return loss


    def validation_step(self, batch, batch_idx):
        """
        Validation step for a batch.
        """

        images, targets = batch
        logits = self(images)
        loss = self.compute_loss(logits, targets)
        acc = self.compute_accuracy(logits, targets)
        self.log_metrics(loss, acc, prefix="val")
        return loss


    def forward(self, x):
        """
        Abstract forward pass. Subclasses must implement this.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output logits.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for model training.
        Returns:
            dict: A dictionary containing:
                - 'optimizer': An Adam optimizer initialized with the model parameters and learning rate `self.lr`.
                - 'lr_scheduler': A dictionary specifying a ReduceLROnPlateau scheduler that monitors 'val_loss',
                  reduces the learning rate by a factor of 0.5 if no improvement is seen for 5 epochs,
                  and is updated every epoch.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}