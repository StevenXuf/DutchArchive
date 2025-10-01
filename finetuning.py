import fire
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


class CLIPFineTuner(pl.LightningModule):
    def __init__(self, model_name="openai/clip-vit-base-patch32", lr=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.lr = lr

    def forward(self, pixel_values, input_ids, attention_mask):
        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_loss=True
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def collate_fn(batch, processor):
    texts = [item['text'] for item in batch]
    images = [item['image'] for item in batch]
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    return inputs


def train_clip(
    model_name="openai/clip-vit-base-patch32",
    dataset_name="flickr8k",
    batch_size=8,
    lr=1e-5,
    max_epochs=5,
    patience=2,
    num_workers=2,
    output_dir="checkpoints"
):
    # Load dataset (images + captions)
    dataset = load_dataset(dataset_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    def preprocess(example):
        return {"image": example["image"], "text": example["caption"]}

    dataset = dataset.map(preprocess)
    train_loader = DataLoader(
        dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn(x, processor)
    )
    val_loader = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn(x, processor)
    )

    # Model
    model = CLIPFineTuner(model_name=model_name, lr=lr)

    # Callbacks
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=output_dir,
        filename="best-clip"
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min"
    )

    # Logger
    logger = TensorBoardLogger("logs", name="clip_finetune")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint, early_stop],
        logger=logger,
        accelerator="auto",
        devices="auto"
    )

    # Train
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    fire.Fire(train_clip)

