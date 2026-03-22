import json
import pandas as pd
import requests
import datetime
from tqdm import tqdm

OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "nomic-embed-text"

def get_nomic_embedding(text):
    """Fetch 768-D embedding from local Ollama endpoint."""
    payload = {
        "model": MODEL_NAME,
        "prompt": text
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json().get("embedding", [])
    except Exception as e:
        print(f"Error fetching embedding: {e}")
        return []

def main():
    print("Loading raw_usajobs_data.json...")
    try:
        with open("raw_usajobs_data.json", "r") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print("Error: raw_usajobs_data.json not found. Please run acquire_data.py first.")
        return

    df = pd.DataFrame(raw_data)
    initial_len = len(df)
    
    # 1. Filter: Drop missing salary or descriptions
    df.dropna(subset=["MinimumRange", "MaximumRange", "Description", "PublicationStartDate", "RateIntervalCode"], inplace=True)
    
    # 2. Filter: Only "Per Year"
    df = df[df["RateIntervalCode"] == "Per Year"].copy()
    
    # Ensure numeric bounds
    df["MinimumRange"] = pd.to_numeric(df["MinimumRange"], errors="coerce")
    df["MaximumRange"] = pd.to_numeric(df["MaximumRange"], errors="coerce")
    df.dropna(subset=["MinimumRange", "MaximumRange"], inplace=True)
    
    # 3. Target Variable Engineering
    df["Target_Salary"] = (df["MinimumRange"] + df["MaximumRange"]) / 2.0
    
    # Process Year
    df["Year"] = pd.to_datetime(df["PublicationStartDate"]).dt.year
    
    print(f"Filtered DataFrame from {initial_len} to {len(df)} rows.")
    
    if len(df) == 0:
        print("No data left after filtering.")
        return
        
    # 4. Local Text Vectorization (Nomic)
    print(f"Vectorizing texts locally using Ollama ({MODEL_NAME})...")
    embeddings = []
    
    # We use tqdm for a progress bar
    for text in tqdm(df["Description"]):
        # Truncate to avoid context limit issues (~8192 tokens for Nomic)
        # Using a simplistic character truncation for safety
        truncated_text = text[:15000] 
        emb = get_nomic_embedding(truncated_text)
        embeddings.append(emb)
        
    # Append the 768-dimensional vectors as new columns
    valid_indices = [i for i, emb in enumerate(embeddings) if len(emb) == 768]
    if len(valid_indices) < len(df):
        print(f"Warning: {len(df) - len(valid_indices)} rows failed vectorization. Dropping them.")
        
    df = df.iloc[valid_indices].copy()
    valid_embeddings = [embeddings[i] for i in valid_indices]
    
    emb_cols = [f"dim_{i}" for i in range(768)]
    emb_df = pd.DataFrame(valid_embeddings, index=df.index, columns=emb_cols)
    
    # Concat
    final_df = pd.concat([df[["Year", "Target_Salary"]], emb_df], axis=1)
    
    # 5. Final Export
    out_path = "processed_data.parquet"
    print(f"Saving to {out_path}...")
    final_df.to_parquet(out_path, engine="pyarrow")
    print("Done!")

if __name__ == "__main__":
    main()
