import torch
import torch.nn as nn
import pytorch_lightning as pl

class PinballLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """
        preds: (batch_size, num_quantiles)
        target: (batch_size, 1)
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i:i+1] # Error for this quantile
            # L_q = max(q * err, (q - 1) * err)
            loss_q = torch.max(q * errors, (q - 1) * errors)
            losses.append(loss_q)
        # Sum losses over quantiles, then average across the batch.
        return torch.stack(losses, dim=1).sum(dim=1).mean()

class SalaryPredictor(pl.LightningModule):
    def __init__(self, input_dim=769, hidden_layers=2, neurons=128, dropout_rate=0.2, lr=1e-3, loss_type='MSE'):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.loss_type = loss_type
        
        # Determine output dimension
        self.out_dim = 3 if loss_type == 'Quantile' else 1
        
        # Build dynamic MLP
        layers = []
        current_dim = input_dim
        
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = neurons
            
        layers.append(nn.Linear(current_dim, self.out_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Assign objective funtion
        if self.loss_type == 'MSE':
            self.criterion = nn.MSELoss()
        elif self.loss_type == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_type == 'Quantile':
            self.criterion = PinballLoss(quantiles=[0.1, 0.5, 0.9])
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward(self, x):
        return self.mlp(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log('train_loss', loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log('val_loss', loss, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
