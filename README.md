# Federal Salary Prediction App

An end-to-end educational machine learning pipeline and interactive Streamlit application that predicts federal salary bands based on job descriptions using PyTorch Lightning.

## Overview
This project guides students through taking raw data from the USAJOBS API all the way to a deployed interactive PyTorch Lightning model. The core pedagogical goal is for students to visually compare the behaviors of Mean Squared Error (MSE), Mean Absolute Error (MAE), and Simultaneous Quantile Regression (Pinball Loss).

## Educational Methods & Architecture

This repository is designed to demonstrate a complete, modern machine learning pipeline. Students will learn the following core concepts:

### 1. Data Acquisition & Vectorization
- **USAJOBS API**: We pull real-world, messy tabular data and text descriptions from the official federal government job board via REST API.
- **HuggingFace `sentence-transformers`**: Instead of relying on traditional bag-of-words or TF-IDF, we utilize `nomic-ai/nomic-embed-text-v1.5` natively in Python to convert long job descriptions into rich, 768-dimensional semantic embeddings. 

### 2. PyTorch Lightning Framework
- **`LightningDataModule`**: Encapsulates all `Pandas` preprocessing, `scikit-learn` scaling (`MinMaxScaler` for the `Year` feature), and PyTorch `DataLoader` batching into a single reusable class.
- **`LightningModule`**: Defines a dynamic Multilayer Perceptron (MLP) mapping a 769-dimensional input (768 text dimensions + 1 scaled year) to either a single continuous value or a multi-output quantile regression. 

### 3. Objective Functions (Loss)
The Streamlit dashboard allows real-time swapping of the neural network's loss function to observe how it fundamentally changes the model's behavior:
- **Mean Squared Error (MSE)**: Heavily penalizes large errors. Tends to over-index on outliers, pulling predictions toward the statistical mean.
- **Mean Absolute Error (MAE)**: Penalizes errors linearly. Highly robust to outliers, pulling predictions toward the statistical median.
- **Simultaneous Quantile Regression (Pinball Loss)**: Instead of a single point prediction, the model outputs 3 separate values representing the **10th, 50th, and 90th percentiles**. The Pinball Loss formula $L_q(y,\hat{y}) = \max(q(y-\hat{y}), (q-1)(y-\hat{y}))$ uniquely penalizes overestimates vs. underestimates depending on the target quantile $q$. This visually demonstrates that ML models can output *confidence intervals* and distributions rather than absolute certainties.

### 4. Interactive Visualizations & Interpretability 
- **PCA Dimensionality Reduction**: The 768-dimensional job embeddings are reduced to 2D spatial coordinates using Principal Component Analysis (`sklearn.decomposition.PCA`). This allows students to visually inspect how the neural network clusters semantic job families. 
- **Nearest Neighbors (Cosine Similarity)**: By passing a new, hypothetical job description through the embedder and calculating the Vector Dot Product (Cosine Similarity) against the validation set, the app instantly retrieves the most semantically identical federal roles.

## Setup Instructions

1. **Prerequisites**
   - Install `uv` for python environment management.
   - Vectorization is handled natively in Python via HuggingFace `sentence-transformers`.

2. **Environment Initialization**
   ```bash
   uv sync
   ```

3. **Data Pipeline**
   - Copy `.env.example` to `.env` and insert your USAJOBS API Key and Email.
   - Run the extraction script:
     ```bash
     uv run scripts/acquire_data.py
     ```
   - Run the preprocessing and vectorization script:
     ```bash
     uv run scripts/process_data.py
     ```
   - *This will generate `processed_data.parquet` in the root directory.*

4. **Run the Streamlit Dashboard**
   ```bash
   uv run streamlit run app.py
   ```

## Repository Structure
- `PLAN.md`: The original project plan and specification.
- `scripts/`: Independent scripts for data acquisition and preprocessing.
- `src/`: Core Python modules (`data.py`, `model.py`, `callbacks.py`, `visualization.py`).
- `app.py`: The main Streamlit dashboard application.
