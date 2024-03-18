import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import BinaryF1Score
from torcheval.metrics import R2Score
from sklearn.metrics import r2_score

from nam.trainer.losses import penalized_loss
from nam.trainer.metrics import accuracy
from nam.trainer.metrics import mae, mse
from nam.types import Config


class LitNAM(pl.LightningModule):

    def __init__(self, config: Config, model: nn.Module) -> None:
        super().__init__()
        self.config = Config(**vars(config))  #config
        self.model = model
        self.f1_score_metric = BinaryF1Score()
        self.r2_score_metric = R2Score()
        self.criterion = lambda inputs, targets, weights, fnns_out, model: penalized_loss(
            self.config, inputs, targets, weights, fnns_out, model)

        self.metrics = lambda logits, targets: mse(logits, targets) if config.regression else accuracy(logits, targets)
        self.metrics_name = "MSE" if config.regression else "Accuracy"

        self.save_hyperparameters(vars(self.config))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits, fnns_out = self.model(inputs)
        return logits, fnns_out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.decay_rate)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        features, targets = batch
        # inputs, targets, *weights = batch
        # weights = weights.pop() if weights else torch.tensor(1)

        logits, fnns_out = self.model(features)
        loss = self.criterion(logits, targets, None, fnns_out, self.model)
        metric = self.metrics(logits, targets)

        self.log_dict(
            {
                'train_loss': loss,
                f"{self.metrics_name}_metric": metric
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss  #{'training_loss': loss}

    def validation_step(self, batch, batch_idx):
        features, targets = batch
        # inputs, targets, *weights = batch
        # weights = weights.pop() if weights else torch.tensor(1)

        logits, fnns_out = self.model(features)  #, weights)
        loss = self.criterion(logits, targets, None, fnns_out, self.model)
        metric = self.metrics(logits, targets)

        self.log_dict(
            {
                'val_loss': loss,
                f"{self.metrics_name}_metric": metric
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        features, targets = batch
        # inputs, targets, *weights = batch

        logits, fnns_out = self.model(features)
        loss = self.criterion(logits, targets, None, fnns_out, self.model)
        metric = self.metrics(logits, targets)
        f1_score = -1
        r2 = -5
        if self.config.regression is False:
            f1_score = self.f1_score_metric(logits.view(-1), targets.view(-1))
        else: 
            # self.r2_score_metric.update(logits.view(-1), targets.view(-1))
            # r2 = self.r2_score_metric.compute()
            print(logits.view(-1).tolist())
            print(targets.view(-1).tolist())
            r2 = r2_score(logits.view(-1).tolist(), targets.view(-1).tolist())
        self.log_dict(
            {
                'test_loss': loss,
                f"{self.metrics_name}_metric": metric,
                "f1_score": f1_score,
                'r2': r2
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {'test_loss': loss}
