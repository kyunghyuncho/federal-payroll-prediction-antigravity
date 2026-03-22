import pytorch_lightning as pl
import pandas as pd

class StreamlitLiveMetrics(pl.Callback):
    def __init__(self, st_placeholder):
        super().__init__()
        self.st_placeholder = st_placeholder
        self.metrics_history = {"Epoch": [], "Train Loss": [], "Val Loss": []}

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        
        val_loss = metrics.get('val_loss')
        train_loss = metrics.get('train_loss')
        
        val_loss_val = val_loss.item() if val_loss is not None else None
        train_loss_val = train_loss.item() if train_loss is not None else None
        
        if val_loss_val is not None:
            self.metrics_history["Epoch"].append(trainer.current_epoch)
            
            if train_loss_val is None:
                if len(self.metrics_history["Train Loss"]) > 0:
                    train_loss_val = self.metrics_history["Train Loss"][-1]
                else:
                    train_loss_val = val_loss_val
                    
            self.metrics_history["Train Loss"].append(train_loss_val)
            self.metrics_history["Val Loss"].append(val_loss_val)
            
            df = pd.DataFrame({
                "Train Loss": self.metrics_history["Train Loss"],
                "Val Loss": self.metrics_history["Val Loss"]
            }, index=self.metrics_history["Epoch"])
            
            self.st_placeholder.line_chart(df)
