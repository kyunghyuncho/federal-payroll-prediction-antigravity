# Federal Salary Prediction App

An end-to-end educational machine learning pipeline and interactive Streamlit application that predicts federal salary bands based on job descriptions using PyTorch Lightning.

## Overview
This project guides students through taking raw data from the USAJOBS API all the way to a deployed interactive PyTorch Lightning model. The core pedagogical goal is for students to visually compare the behaviors of Mean Squared Error (MSE), Mean Absolute Error (MAE), and Simultaneous Quantile Regression (Pinball Loss).

## Setup Instructions

1. **Prerequisites**
   - Install `uv` for python environment management.
   - Install Ollama and pull the embedding model: `ollama run nomic-embed-text`

2. **Environment Initialization**
   ```bash
   uv sync
   ```

3. **Data Pipeline**
   - Edit `scripts/acquire_data.py` to insert your USAJOBS API Key and Email.
   - Run the extraction script:
     ```bash
     uv run scripts/acquire_data.py
     ```
   - Run the preprocessing and vectorization script (Make sure Ollama is running):
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
