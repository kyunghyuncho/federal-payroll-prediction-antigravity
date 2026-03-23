import streamlit as st
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
import os
from sentence_transformers import SentenceTransformer

from src.data import SalaryDataModule
from src.model import SalaryPredictor
from src.callbacks import StreamlitLiveMetrics
from src.visualization import plot_residual_distribution, plot_actual_vs_predicted, plot_quantile_bands, plot_pca_features

st.set_page_config(layout="wide", page_title="Federal Salary Predictor")

st.title("Federal Salary Prediction: Interactive Dashboard")
st.markdown("Explore how different loss functions (MSE, MAE, Quantile Distribution) change neural network outputs for federal job salaries using HuggingFace embeddings.")

# Sidebar
st.sidebar.header("Architecture")
hidden_layers = st.sidebar.slider("Hidden Layers", 1, 5, 2)
neurons = st.sidebar.slider("Neurons per Layer", 32, 512, 128, step=32)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, step=0.05)

st.sidebar.header("Optimization")
lr = st.sidebar.selectbox("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2], index=2)
batch_size = st.sidebar.select_slider("Batch Size", options=[16, 32, 64, 128, 256], value=32)
max_epochs = st.sidebar.slider("Epochs", 5, 500, 15)

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

# Layout columns for Train and Interactive Predictor side-by-side
col_left, col_right = st.columns([1, 1])

with col_left:
    if st.button("Train Model"):
        st.subheader(f"Training MLP with {loss_type} Loss")
        
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

with col_right:
    st.subheader("Interactive Try-It-Out")
    custom_desc = st.text_area("Job Description", "Enter the major duties and responsibilities for a hypothetical job here...", height=150)
    custom_year = st.number_input("Publication Year", min_value=1990, max_value=2050, value=2024)
    
    if st.button("Predict Salary"):
        if "model" not in st.session_state:
            st.error("Please train a model first so we have weights available for prediction!")
        else:
            with st.spinner("Loading HuggingFace Embedder..."):
                @st.cache_resource
                def get_embedder():
                    device = "mps" if torch.backends.mps.is_available() else "cpu"
                    return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", device=device, trust_remote_code=True)
                embedder = get_embedder()
                
            with st.spinner("Predicting..."):
                emb = embedder.encode(["search_document: " + custom_desc])
                
                scaler = st.session_state["datamodule"].scaler
                yr_scaled = scaler.transform([[custom_year]])
                
                x_custom = np.concatenate([emb, yr_scaled], axis=1)
                x_tensor = torch.tensor(x_custom, dtype=torch.float32)
                
                model = st.session_state["model"]
                model.eval()
                with torch.no_grad():
                    pred = model(x_tensor)
                
                loss_t = st.session_state["loss_type"]
                if loss_t in ["MSE", "MAE"]:
                    st.success(f"**Predicted Salary:** ${pred.item():,.2f}")
                elif loss_t == "Quantile":
                    qpreds = pred.squeeze().tolist()
                    st.success("**Predicted Salary Bounds:**")
                    st.write(f"- 10th Percentile: **${qpreds[0]:,.2f}**")
                    st.write(f"- 50th Percentile (Median): **${qpreds[1]:,.2f}**")
                    st.write(f"- 90th Percentile: **${qpreds[2]:,.2f}**")

# Inference & Visualization
if "model" in st.session_state:
    st.markdown("---")
    
    col_hdr, col_tog = st.columns([3, 1])
    with col_hdr:
        st.subheader("Interactive Analysis & Visualization")
    with col_tog:
        show_all_data = st.checkbox("Show All Data (Train + Val)", value=False)
    
    model = st.session_state["model"]
    datamodule = st.session_state["datamodule"]
    current_loss_type = st.session_state["loss_type"]
    
    model.eval()
    
    from torch.utils.data import DataLoader
    loaders_to_evaluate = []
    
    if show_all_data:
        train_eval_loader = DataLoader(datamodule.train_dataset, batch_size=datamodule.batch_size, shuffle=False)
        loaders_to_evaluate.append(train_eval_loader)
        plot_df = pd.concat([datamodule.df_train, datamodule.df_val], ignore_index=True)
    else:
        plot_df = datamodule.df_val
        
    loaders_to_evaluate.append(datamodule.val_dataloader())
    
    all_preds = []
    all_targets = []
    all_embeddings = []
    
    with torch.no_grad():
        for loader in loaders_to_evaluate:
            for batch in loader:
                x, y = batch
                preds = model(x)
                all_preds.append(preds)
                all_targets.append(y)
                all_embeddings.append(x[:, :768]) # extract just the 768-D embeddings
            
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy().squeeze()
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    
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
        
    from sklearn.metrics.pairwise import cosine_similarity

    st.markdown("---")
    st.subheader("2D Representation of Job Descriptions (PCA)")
    st.markdown("*(Click on any dot to see the job details card below!)*")
    
    fig_actual, fig_pred = plot_pca_features(all_embeddings, all_targets, point_preds, plot_df)
    
    col_pca1, col_pca2 = st.columns(2)
    with col_pca1:
        sel_actual = st.plotly_chart(fig_actual, use_container_width=True, on_select="rerun")
    with col_pca2:
        sel_pred = st.plotly_chart(fig_pred, use_container_width=True, on_select="rerun")
        
    def get_selected_index(sel):
        try:
            sel_dict = dict(sel)
            pts = sel_dict.get("selection", {}).get("points", [])
            if len(pts) > 0:
                p = pts[0]
                idx = p.get("point_index")
                if idx is None:
                    idx = p.get("point_number")
                return int(idx) if idx is not None else None
        except Exception:
            pass
        return None

    selected_idx = get_selected_index(sel_actual)
    if selected_idx is None:
        selected_idx = get_selected_index(sel_pred)
        
    if selected_idx is not None:
        st.markdown("---")
        st.subheader("Job Details Card")
        job_row = plot_df.iloc[selected_idx]
        
        st.info(f"**Actual Salary:** ${job_row['Target_Salary']:,.2f} | **Predicted Salary:** ${point_preds[selected_idx]:,.2f} | **Year:** {job_row['Year']}")
        st.write(f"**Description:** {job_row['Description']}")
        
        st.markdown("#### Nearest Neighbor Jobs")
        query_emb = all_embeddings[selected_idx].reshape(1, -1)
        sims = cosine_similarity(query_emb, all_embeddings)[0]
        
        top_indices = np.argsort(sims)[::-1][1:4]
        
        for idx in top_indices:
            nn_row = plot_df.iloc[idx]
            nn_sim = sims[idx]
            st.markdown(f"- **Similarity: {nn_sim:.2f}** | **Salary:** ${nn_row['Target_Salary']:,.2f} | *{str(nn_row['Description'])[:200]}...*")
