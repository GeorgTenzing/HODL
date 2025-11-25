import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pyloudnorm as pyln
import torchaudio.transforms as T
from audio_transform import AugmentedDataset

from blocks import Conv1d_block, SeparableConv1d, in_block, Deconv, Bottleneck, WaveUNetBlock, DCUNetBlock, TinyConvBlock, ResidualBlock, AttentionBlock, GatedConv1d
  
LAYER_KINDS = {
        "conv", "pool", "drop", "bottleneck", "out", "sepconv", # working blocks
        # "res", "attention", "gatedconv", "WaveUNetBlock", "DCUNetBlock", "TinyConvBlock" # new blocks 
    }

LAYER_FACTORY = {
    "conv":       Conv1d_block,
    "bottleneck": Conv1d_block,
    "sepconv": SeparableConv1d,
    "deconv":  lambda ic, oc: nn.ConvTranspose1d(ic, oc, kernel_size=2, stride=2),
    "out":     lambda ic, oc: nn.Conv1d(ic, oc, kernel_size=1),
    "pool":    lambda *a:     nn.MaxPool1d(2),
    "drop":    lambda *a:     nn.Dropout(a[0]),
    
    # Advanced building blocks
      "res":         ResidualBlock,
      "attention":   AttentionBlock,
      "gatedconv":   GatedConv1d,
      "waveunet":    WaveUNetBlock,
      "dcunet":      DCUNetBlock,
      "tiny":        TinyConvBlock,
}


# --- Compact User Spec ---
def init_model() -> nn.Module:
    from init_model import fun_idea, sol_6547
    user_spec = sol_6547
    # user_spec = fun_idea
    return FlexibleUNet1D(user_spec).to("cuda")
    
# --- Training Function ---
def train_model(model: nn.Module, dev_dataset: Dataset) -> nn.Module:

    train_dataset = AugmentedDataset(dev_dataset)
    train_loader = DataLoader(
      train_dataset, 
      batch_size=32, 
      shuffle=True, 
      num_workers=7, # 7
      pin_memory=True,
      # persistent_workers=True,
      # prefetch_factor=4, 
      ) 
    # tab1: ep 153, 0.000217, (158-14:43), tab2: ep  ,  , (158-14:  )
    # Last Ideas : ep=151, base = 20, num worker 8, audgmented dataset with cache
    
    
    trainer = pl.Trainer(
        max_epochs = 152, # 126, 160 war 1.6705
        gradient_clip_val=1.0,
        accelerator="gpu",
        log_every_n_steps=1,
        enable_progress_bar=False,
        precision= 16 ,  
        limit_val_batches=0,    # skip validation
        enable_checkpointing=False, 
        num_sanity_val_steps=0,
        fast_dev_run=False,
        # max_time="00:00:14:55", # This could be interesting
        # benchmark=True,         # speec boost i think
    )
    
    trainer.fit(model, train_loader)
    return model



# --- Main Model ---
class FlexibleUNet1D(pl.LightningModule):
    def __init__(self, user_spec, in_channels=1, out_channels=1, base=20):   #base 16
        super().__init__()
        self.apply(lambda m: nn.init.kaiming_normal_(m.weight) if isinstance(m, nn.Conv1d) else None)
        self.base = base
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.config_spec = translate_spec(user_spec)
        self.layers = nn.ModuleList()

        for i, (category, kind, *args) in enumerate(self.config_spec):
            if category == "layer":
                factory = LAYER_FACTORY[kind]
                if kind in ["pool", "drop"]:
                    self.layers.append(factory(*args))
                else:
                    in_mult, out_mult, *rest = args
                    in_c = in_channels if i == 0 else in_mult * base
                    out_c = out_channels if kind == "out" else out_mult * base
                    self.layers.append(factory(in_c, out_c))
                    
                    
    def forward(self, x):
        context = {}
        i = 0
        for category, kind, *args in self.config_spec:
            if category == "layer":
                x = self.layers[i](x)
                i += 1
                if args and isinstance(args[-1], str):
                    context[args[-1]] = x
                    
            elif category == "transform": 
                if kind == "concat":
                    skip_name = args[0]
                    x = torch.cat([x, context[skip_name]], dim=1)
        return x
        

    def training_step(self, batch, batch_idx):
        noisy, target = batch
        x_hat = self(noisy)

        if x_hat.size(-1) != target.size(-1):
            x_hat = F.pad(x_hat, (0, target.size(-1) - x_hat.size(-1)))
              
        loss = F.mse_loss(x_hat, target)
        self.log("train_loss", loss, prog_bar=True) 
        
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-3)  
        
    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get("train_loss")
        if avg_loss is not None:
            print(f"✅ Finished Epoch {self.current_epoch + 1} | Avg loss: {avg_loss:.6f}")


def translate_spec(user_spec: list) -> list:
    """
    Expand a user-friendly model spec into a normalized config spec.
    Converts tuples like ("deconv", in, out, "skip") into a full
    sequence of deconv + concat + conv blocks.
    """
    config = []

    for step in user_spec:
        kind = step[0]

        # --- deconv shortcut expansion ---
        if kind == "deconv" and len(step) == 4:
            _, in_mult, out_mult, skip = step
            config.extend([
                ("layer", "deconv", in_mult, out_mult),
                ("transform", "concat", skip),
                ("layer", "conv", out_mult * 2, out_mult),
            ])
            continue

        # --- regular layer kinds ---
        if kind in LAYER_KINDS:
            config.append(("layer", *step))
        else:
            raise ValueError(f"Unknown spec kind: {kind}")

    # optional pretty-print
    print("\nConfig Spec:")
    for i, (cat, kind, *args) in enumerate(config):
        print(f"{i:02d}: {cat:<9} {kind:<10} {args}")

    return config