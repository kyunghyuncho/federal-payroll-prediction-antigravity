# Federal Salary Prediction App

An end-to-end educational machine learning pipeline and interactive Streamlit application that predicts federal salary bands based on job descriptions using PyTorch Lightning.

## Overview
This project guides students through taking raw data from the USAJOBS API all the way to a deployed interactive PyTorch Lightning model. The core pedagogical goal is for students to visually compare the behaviors of Mean Squared Error (MSE), Mean Absolute Error (MAE), and Simultaneous Quantile Regression (Pinball Loss).

## Educational Methods & Architecture

This repository is designed to demonstrate a complete, modern machine learning pipeline. Students will learn the following core concepts:

### 1. Data Acquisition & Vectorization
- **USAJOBS API**: We pull real-world, messy tabular data and text descriptions from the official federal government job board via REST API.
  - **How to get an API Key**: 
    1. Visit the [USAJOBS Developer Portal](https://developer.usajobs.gov/).
    2. Create an account and navigate to your developer dashboard to request an API Key.
    3. Once granted, copy `.env.example` to `.env` in this directory. 
    4. Set your `USAJOBS_API_KEY` and set `USAJOBS_EMAIL` to your registration email.
- **HuggingFace `sentence-transformers`**: Instead of relying on traditional bag-of-words or TF-IDF, we utilize `nomic-ai/nomic-embed-text-v1.5` natively in Python to convert long job descriptions into rich, 768-dimensional semantic embeddings. 

### 2. PyTorch Lightning Framework
- **`LightningDataModule`**: Encapsulates all `Pandas` preprocessing, `scikit-learn` scaling (`MinMaxScaler` for the `Year` feature), and PyTorch `DataLoader` batching into a single reusable class.
- **`LightningModule`**: Defines a dynamic Multilayer Perceptron (MLP) mapping a 769-dimensional input (768 text dimensions + 1 scaled year) to either a single continuous value or a multi-output quantile regression. 

### 3. Objective Functions (Loss)
The Streamlit dashboard allows real-time swapping of the neural network's loss function to observe how it fundamentally changes the model's behavior. Let $y$ be the actual salary and $\hat{y}$ be the predicted salary:

- **Mean Squared Error (MSE)**
  $$L_{\text{MSE}} = (y - \hat{y})^2$$
  Heavily penalizes large errors. Because of the squaring term, the model inherently over-indexes on outliers (like exceptionally high executive salaries), pulling predictions toward the statistical **mean**.

- **Mean Absolute Error (MAE)**
  $$L_{\text{MAE}} = |y - \hat{y}|$$
  Penalizes errors linearly. It is highly robust to outliers because an error of \$100k is exactly 10x worse than an error of \$10k, unlike MSE where it would be 100x worse. This pulls the model's predictions toward the statistical **median**.

- **Simultaneous Quantile Regression (Pinball Loss)**
  Instead of a single point prediction, the model outputs 3 separate values representing the **10th, 50th, and 90th percentiles**. The Pinball Loss formula for a given target quantile $q \in (0, 1)$ is:
  $$L_q(y,\hat{y}) = \max(q(y-\hat{y}), (q-1)(y-\hat{y}))$$
  For example, if $q=0.90$, underestimating the salary ($y > \hat{y}$) is heavily penalized by $0.90 \times \text{error}$, while overestimating is only penalized by $-0.10 \times \text{error}$. By computing this loss simultaneously for $q=\{0.10, 0.50, 0.90\}$, the model actively learns to output structured **confidence intervals** rather than absolute point certainties.

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
