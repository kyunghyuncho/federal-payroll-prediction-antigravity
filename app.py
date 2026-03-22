import streamlit as st
import pandas as pd
import torch
import pytorch_lightning as pl
import os

from src.data import SalaryDataModule
from src.model import SalaryPredictor
from src.callbacks import StreamlitLiveMetrics
from src.visualization import plot_residual_distribution, plot_actual_vs_predicted, plot_quantile_bands

st.set_page_config(layout="wide", page_title="Federal Salary Predictor")

st.title("Federal Salary Prediction: Comparative Analysis")
st.markdown("Explore how different loss functions (MSE, MAE, Quantile Distribution) change neural network outputs for federal job salaries using Nomic embeddings.")

# Sidebar
st.sidebar.header("Architecture")
hidden_layers = st.sidebar.slider("Hidden Layers", 1, 5, 2)
neurons = st.sidebar.slider("Neurons per Layer", 32, 512, 128, step=32)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, step=0.05)

st.sidebar.header("Optimization")
lr = st.sidebar.selectbox("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2], index=2)
batch_size = st.sidebar.select_slider("Batch Size", options=[16, 32, 64, 128, 256], value=32)
max_epochs = st.sidebar.slider("Epochs", 5, 50, 15)

st.sidebar.header("Objective")
loss_type = st.sidebar.radio("Loss Function", ["MSE", "MAE", "Quantile"], help="MSE penalizes outliers. MAE is median-seeking. Quantile plots simultaneous percentiles.")

@st.cache_data
def load_data():
    if not os.path.exists("processed_data.parquet"):
        return None
    return pd.read_parquet("processed_data.parquet")

df = load_data()

if df is None:
    st.error("`processed_data.parquet` not found. Please follow the instructions to run `scripts/acquire_data.py` and `scripts/process_data.py` first!")
    st.stop()

st.write(f"**Loaded {len(df)} samples into memory.**")

if st.button("Train Model"):
    st.subheader(f"Training MLP with {loss_type} Loss")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Live Training Metrics**")
        metrics_placeholder = st.empty()
    
    with st.spinner("Initializing Data & Model..."):
        datamodule = SalaryDataModule(df, batch_size=batch_size, test_size=0.2)
        datamodule.setup()
        
        model = SalaryPredictor(
            input_dim=769, 
            hidden_layers=hidden_layers, 
            neurons=neurons, 
            dropout_rate=dropout_rate,
            lr=lr,
            loss_type=loss_type
        )
        
        live_callback = StreamlitLiveMetrics(metrics_placeholder)
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[live_callback],
            enable_progress_bar=False,
            logger=False 
        )
        
    with st.spinner("Training in progress..."):
        trainer.fit(model, datamodule=datamodule)
        st.success("Training complete!")
        
        st.session_state["model"] = model
        st.session_state["datamodule"] = datamodule
        st.session_state["loss_type"] = loss_type

# Inference & Visualization
if "model" in st.session_state:
    st.markdown("---")
    st.subheader("Validation Set Analysis")
    
    model = st.session_state["model"]
    datamodule = st.session_state["datamodule"]
    current_loss_type = st.session_state["loss_type"]
    
    model.eval()
    val_loader = datamodule.val_dataloader()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            preds = model(x)
            all_preds.append(preds)
            all_targets.append(y)
            
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy().squeeze()
    
    if current_loss_type in ["MSE", "MAE"]:
        point_preds = all_preds.squeeze()
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.plotly_chart(plot_residual_distribution(all_targets, point_preds), use_container_width=True)
        with col_v2:
            st.plotly_chart(plot_actual_vs_predicted(all_targets, point_preds), use_container_width=True)
            
    elif current_loss_type == "Quantile":
        point_preds = all_preds[:, 1] # 50th percentile as the primary prediction
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.plotly_chart(plot_residual_distribution(all_targets, point_preds, title="Residuals (50th Percentile)"), use_container_width=True)
        with col_v2:
            st.plotly_chart(plot_actual_vs_predicted(all_targets, point_preds, title="Actual vs. 50th Percentile Prediction"), use_container_width=True)
            
        st.plotly_chart(plot_quantile_bands(all_targets, all_preds), use_container_width=True)
