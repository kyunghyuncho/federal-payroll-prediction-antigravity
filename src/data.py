import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class SalaryDataModule(pl.LightningDataModule):
    def __init__(self, df: pd.DataFrame, batch_size: int = 32, test_size: float = 0.2):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.test_size = test_size
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        # 1. Separate features and target
        # Assuming the dataframe has "Year", "Target_Salary", and "dim_0" to "dim_767"
        year_data = self.df[['Year']].values
        target_data = self.df['Target_Salary'].values
        
        # 2. Extract embeddings
        emb_cols = [c for c in self.df.columns if c.startswith('dim_')]
        emb_data = self.df[emb_cols].values
        
        # 3. Scale the Year column
        self.scaler = MinMaxScaler()
        year_scaled = self.scaler.fit_transform(year_data)
        
        # 4. Concatenate scaled Year and embeddings (769-d input)
        X = np.concatenate([emb_data, year_scaled], axis=1)
        y = target_data
        
        # 5. Train / Test Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )
        
        # 6. Convert to PyTorch Tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        
        # 7. Create TensorDatasets
        self.train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
