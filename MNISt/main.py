import torch
import torch.nn as nn
import torchvision.transforms as T
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR


# Your CNN architecture
class MNISTNet(nn.Module):
    def __init__(self, seq, drop_p, drop_final):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.SiLU()
            )

        def maxpool_dropout(pool_kernel=2, drop_p=drop_p):
            return nn.Sequential(
                nn.MaxPool2d(pool_kernel),
                nn.Dropout(drop_p),
            )

        def conv_block_sequence(channels):
            layers = []
            for in_c, out_c in zip(channels, channels[1:]):
                layers.append(conv_block(in_c, out_c))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block_sequence(seq[0]),
            maxpool_dropout(),
            conv_block_sequence(seq[1]),
            maxpool_dropout(),
            conv_block_sequence(seq[2]),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(seq[-1][-1]),
            nn.Dropout(drop_final),
            nn.Linear(seq[-1][-1], 10)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Lightning wrapper
class LightningMNIST(pl.LightningModule):
    def __init__(self, model: nn.Module, lr=5e-3, weight_decay=1e-3, label_smoothing=0.1, epochs=8):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': OneCycleLR(
                optimizer,
                max_lr=self.lr,
                epochs=self.epochs,
                steps_per_epoch=self.trainer.estimated_stepping_batches // self.epochs
            ),
            'interval': 'step',
        }
        return [optimizer], [scheduler]

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)


# --- REQUIRED FUNCTIONALITY ---

# Must return plain nn.Module
def init_model() -> nn.Module:
    drop_p, drop_final = 0.15, 0.15
    seq = [
        [1, 64, 128, 64, 128, 64],
        [64, 128, 256, 128],
        [128, 256, 128, 64, 128]
    ]
    model = MNISTNet(seq, drop_p, drop_final)
    return model


# Must train and return trained nn.Module
def train_model(model: nn.Module, dev_dataset: Dataset) -> nn.Module:
    batch_size = 128
    epochs = 8

    train_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    lightning_model = LightningMNIST(model, lr=5e-3, weight_decay=1e-3, label_smoothing=0.1, epochs=epochs)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=1.0,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(lightning_model, train_dataloaders=train_loader)

    return lightning_model.model
