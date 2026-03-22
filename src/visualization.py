import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_residual_distribution(y_true, y_pred, title="Residual Distribution"):
    errors = y_true - y_pred
    fig = px.histogram(
        x=errors, 
        nbins=50, 
        title=title, 
        labels={'x': 'Error (Actual - Predicted)'},
        color_discrete_sequence=['#ef553b']
    )
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    return fig

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs. Predicted Salary"):
    df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    fig = px.scatter(
        df, x='Actual', y='Predicted', 
        title=title,
        opacity=0.5,
        color_discrete_sequence=['#636efa']
    )
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=min_val, y0=min_val, x1=max_val, y1=max_val
    )
    return fig

def plot_quantile_bands(y_true, q_preds, title="Quantile Predictions (10th - 50th - 90th)"):
    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]
    q10_sorted = q_preds[sort_idx, 0]
    q50_sorted = q_preds[sort_idx, 1]
    q90_sorted = q_preds[sort_idx, 2]
    
    x = np.arange(len(y_true_sorted))
    
    fig = go.Figure()

    # 90th Percentile Band (Upper)
    fig.add_trace(go.Scatter(
        x=x, y=q90_sorted,
        line=dict(width=0),
        showlegend=False,
        name='90th Percentile'
    ))
    
    # 10th Percentile Band (Lower)
    fig.add_trace(go.Scatter(
        x=x, y=q10_sorted,
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(99, 110, 250, 0.2)',
        showlegend=False,
        name='10th Percentile'
    ))
    
    # 50th Percentile
    fig.add_trace(go.Scatter(
        x=x, y=q50_sorted,
        mode='lines',
        line=dict(color='blue', width=2),
        name='50th Percentile (Prediction)'
    ))
    
    # Actual Values
    fig.add_trace(go.Scatter(
        x=x, y=y_true_sorted,
        mode='markers',
        marker=dict(color='red', size=4, opacity=0.5),
        name='Actual Salary'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Samples (Sorted by Actual Salary)",
        yaxis_title="Salary"
    )
    
    return fig

from sklearn.decomposition import PCA

def plot_pca_features(embeddings, actual, predicted, df_val, title="PCA of Job Descriptions"):
    pca = PCA(n_components=2)
    components = pca.fit_transform(embeddings)
    
    hover_desc = df_val['Description'].apply(lambda x: str(x)[:100] + '...')
    
    df = pd.DataFrame({
        'PCA1': components[:, 0],
        'PCA2': components[:, 1],
        'Actual Salary': actual,
        'Predicted Salary': predicted,
        'Description': hover_desc
    })
    
    fig_actual = px.scatter(
        df, x='PCA1', y='PCA2', color='Actual Salary',
        hover_data=['Description'],
        color_continuous_scale='Viridis',
        title=title + " (Actual Salary)",
        opacity=0.7
    )
    
    fig_pred = px.scatter(
        df, x='PCA1', y='PCA2', color='Predicted Salary',
        hover_data=['Description'],
        color_continuous_scale='Viridis',
        title=title + " (Predicted Salary)",
        opacity=0.7
    )
    
    return fig_actual, fig_pred
