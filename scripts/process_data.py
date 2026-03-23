import json
import pandas as pd
import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

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
    df = df[df["RateIntervalCode"] == "PA"].copy()
    
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
        
    # 4. Local Text Vectorization (HuggingFace sentence-transformers)
    print(f"Loading HuggingFace model ({MODEL_NAME})...")
    # trust_remote_code=True is required for custom Nomic architecture
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    
    # Nomic embedding instructions: "search_document: " is the default format for clustering.
    texts = ["search_document: " + str(t)[:15000] for t in df["Description"].tolist()]
    
    print(f"Vectorizing {len(texts)} texts locally...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=16)
    embeddings = embeddings.tolist()
    
    # Append the 768-dimensional vectors as new columns
    valid_indices = [i for i, emb in enumerate(embeddings) if len(emb) == 768]
    if len(valid_indices) < len(df):
        print(f"Warning: {len(df) - len(valid_indices)} rows failed vectorization. Dropping them.")
        
    df = df.iloc[valid_indices].copy()
    valid_embeddings = [embeddings[i] for i in valid_indices]
    
    emb_cols = [f"dim_{i}" for i in range(768)]
    emb_df = pd.DataFrame(valid_embeddings, index=df.index, columns=emb_cols)
    
    # Concat
    final_df = pd.concat([df[["PublicationStartDate", "Year", "Target_Salary", "Description"]], emb_df], axis=1)
    
    # 5. Final Export
    out_path = "processed_data.parquet"
    print(f"Saving to {out_path}...")
    final_df.to_parquet(out_path, engine="pyarrow")
    print("Done!")

if __name__ == "__main__":
    main()
