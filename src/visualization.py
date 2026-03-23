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

def plot_quantile_calibration(y_true, q_preds, quantiles=[0.1, 0.5, 0.9], title="Quantile Calibration Error"):
    empirical_coverage = []
    for i, q in enumerate(quantiles):
        coverage = np.mean(y_true <= q_preds[:, i])
        empirical_coverage.append(coverage)
        
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=quantiles, y=empirical_coverage,
        mode='lines+markers+text',
        name='Model Calibration',
        text=[f"{c:.1%}" for c in empirical_coverage],
        textposition="bottom right",
        marker=dict(size=12, color='#ef553b')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Target Quantile",
        yaxis_title="Empirical Coverage (Portion of Data ≤ Pred)",
        xaxis=dict(range=[0, 1.05], tickformat=".0%"),
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig
from sklearn.decomposition import PCA

import textwrap

def plot_pca_features(embeddings, actual, predicted, df_val, title="PCA of Job Descriptions"):
    pca = PCA(n_components=2)
    components = pca.fit_transform(embeddings)
    
    def format_hover(text):
        truncated = str(text)[:180] + '...' if len(str(text)) > 180 else str(text)
        return '<br>'.join(textwrap.wrap(truncated, width=60))
        
    hover_desc = df_val['Description'].apply(format_hover)
    
    df = pd.DataFrame({
        'PCA1': components[:, 0],
        'PCA2': components[:, 1],
        'Actual Salary': actual,
        'Predicted Salary': predicted,
        'Description': hover_desc
    })
    
    fig_actual = px.scatter(
        df, x='PCA1', y='PCA2', color='Actual Salary',
        hover_data={'PCA1': False, 'PCA2': False, 'Actual Salary': ':$,.0f', 'Description': True},
        color_continuous_scale='Magma',
        title=title + " (Actual Salary)",
        opacity=0.8
    )
    
    fig_pred = px.scatter(
        df, x='PCA1', y='PCA2', color='Predicted Salary',
        hover_data={'PCA1': False, 'PCA2': False, 'Predicted Salary': ':$,.0f', 'Description': True},
        color_continuous_scale='Magma',
        title=title + " (Predicted Salary)",
        opacity=0.8
    )
    
    for fig in [fig_actual, fig_pred]:
        fig.update_traces(marker=dict(size=9, line=dict(width=0)))
        fig.update_layout(
            xaxis_title="",
            yaxis_title="",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial"),
            coloraxis_colorbar=dict(title="")
        )
    
    return fig_actual, fig_pred
