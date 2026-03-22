# PLAN.md: End-to-End Federal Salary Prediction App

## Project Objective
Develop a complete machine learning pipeline and interactive Streamlit application that predicts federal salary bands based on job descriptions. Students will extract raw data from the USAJOBS API, vectorize the text locally using Nomic embeddings, and build a PyTorch Lightning Multilayer Perceptron (MLP). The final dashboard will allow students to visually compare the behaviors of Mean Squared Error (MSE), Mean Absolute Error (MAE), and Simultaneous Quantile Regression.

## Technology Stack
* **Data Extraction:** Python `requests`, USAJOBS REST API
* **Preprocessing:** Pandas, Local Ollama (`nomic-embed-text`)
* **Deep Learning Framework:** PyTorch & PyTorch Lightning
* **Frontend & Visualization:** Streamlit, Plotly Express

---

## Phase 1: Data Acquisition Script (The USAJOBS API)

**Goal:** Write an independent Python script to gather historical job postings and raw text descriptions.

1. **Authentication:** Instruct students to request a free API key via the USAJOBS Developer Portal. Set up the required headers (`Host`, `User-Agent`, `Authorization-Key`).
2. **API Querying:** Write a loop using the `requests` library to query the `GET /api/Search` endpoint. Iterate through the pages using the `Page` and `ResultsPerPage` parameters.
3. **Data Extraction:** Parse the returned JSON payload and extract three specific attributes for each job:
   * The text description (e.g., `JobSummary` or `MajorDuties`).
   * The date (e.g., `PublicationStartDate` to extract the year).
   * The salary bounds (extract `MinimumRange` and `MaximumRange` from the `PositionRemuneration` array).
4. **Raw Storage:** Append the parsed dictionaries to a list and save it locally as `raw_usajobs_data.json` to prevent hitting API rate limits during subsequent steps.

---

## Phase 2: Data Cleaning & Vectorization Script

**Goal:** Transform the raw JSON into the final numerical tensor formats required for deep learning.

1. **Pandas Initialization:** Load `raw_usajobs_data.json` into a Pandas DataFrame.
2. **Target Variable Engineering:** * Filter out any rows where the pay frequency is not listed as "Per Year".
   * Calculate the `Target_Salary` by taking the average of the minimum and maximum ranges.
   * Drop rows with missing text descriptions or missing salary data.
3. **Local Text Vectorization (Nomic):**
   * Ensure the student has Ollama installed locally and has pulled the `nomic-embed-text` model.
   * Write a function that uses the `requests` library to send batches of job descriptions to the local Ollama endpoint (`http://localhost:11434/api/embeddings`).
   * Append the resulting 768-dimensional vectors as new columns in the DataFrame.
4. **Final Export:** Save the cleaned DataFrame (containing the Year, Target Salary, and 768 embedding columns) as `processed_data.parquet`. This file will serve as the backend database for the Streamlit app.

---

## Phase 3: LightningDataModule Setup

**Goal:** Encapsulate all data loading, scaling, and batching logic for PyTorch.

1. **Initialization:** Initialize `st.set_page_config(layout="wide")` in Streamlit.
2. **Data Ingestion:** Write a `@st.cache_data` function to load the `processed_data.parquet` file.
3. **`SalaryDataModule` Class:**
   * In the `setup()` method, apply a Scikit-learn `MinMaxScaler` to the `Year` column, concatenate it with the embeddings (creating a 769-d input), and execute `train_test_split` based on a UI slider.
   * In the `train_dataloader()` and `val_dataloader()` methods, return PyTorch DataLoaders with the batch size dictated by the Streamlit UI.

---

## Phase 4: Model Architecture & Objective Functions

**Goal:** Define the PyTorch Lightning model and the loss functions.

1. **`SalaryPredictor` Module:**
   * Create a `pl.LightningModule` that dynamically builds an `nn.Sequential` MLP based on Streamlit UI inputs (hidden layers, neurons, dropout).
   * Define the `training_step()` and `validation_step()` methods to calculate and log loss.
   * Use `configure_optimizers()` to return an Adam optimizer with a UI-defined learning rate.
2. **Loss Function Routing:**
   * Allow the module to swap between `nn.MSELoss()`, `nn.L1Loss()`, or a custom Pinball Loss function based on user selection.
   * Define the Pinball Loss module to output simultaneous losses for the 10th, 50th, and 90th percentiles using the formula: $L_q(y,\hat{y})=\max(q(y-\hat{y}),(q-1)(y-\hat{y}))$.

---

## Phase 5: Streamlit "Control Room" & Training

**Goal:** Bridge the PyTorch training loop with the web interface.

1. **Hyperparameter UI:** Build the `st.sidebar` with controls for Architecture (layers, nodes, dropout), Optimization (learning rate, batch size, epochs), and Objective (Loss Function).
2. **Live Metrics Callback:** Write a custom PyTorch Lightning Callback class (`StreamlitLiveMetrics`) that extracts `trainer.callback_metrics` at the end of each epoch and writes them directly to an `st.line_chart` placeholder.
3. **Training Execution:** Bind a "Train Model" button to instantiate the `Trainer`, disable the default terminal progress bar, and execute `trainer.fit()`. Save the final weights to `st.session_state`.

---

## Phase 6: Comparative Visualization

**Goal:** Render interactive charts post-training to demonstrate algorithm behavior.

1. **Inference:** Run the trained model from `st.session_state` on the validation set.
2. **Residual Distribution Histogram:** Use Plotly to plot actual vs. predicted errors, highlighting MSE's tighter clustering and vulnerability to outliers versus MAE.
3. **Actual vs. Predicted Scatter:** Plot predictions against actual salaries with a perfect-prediction diagonal line to expose bias across different pay grades.
4. **Quantile Bands:** If Quantile loss is selected, render a Plotly line chart with a shaded ribbon representing the 10th to 90th percentile predictions to illustrate confidence intervals for specific roles.