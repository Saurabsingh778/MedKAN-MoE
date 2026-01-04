"""
MAXIMUM PERFORMANCE VERSION
- Memory-efficient chunked processing
- Pre-compiled regex for label cleaning
- Vectorized operations everywhere
- Smart batch processing for embeddings
"""

import pandas as pd
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from collections import defaultdict
import re

# --- CONFIG ---
DATA_DIR = "./titans_medical_data_v3_clean"
FILES = {
    "direct": "titans_1_direct_attributes.jsonl",
    "noisy": "titans_2_noisy_robustness.jsonl",
    "needle": "titans_3_needle_reasoning.jsonl"
}

PCA_DIMENSIONS = 20 
EMBEDDING_BATCH_SIZE = 512  # Large batches for efficiency

# Pre-compile regex for faster label cleaning
CODE_PATTERN = re.compile(r'(?:Code:\s*)?([A-Z]\d+\.?\d*)')

def clean_label_fast(text):
    """Optimized label cleaning with regex"""
    if not isinstance(text, str) or not text:
        return ""
    match = CODE_PATTERN.search(text)
    return match.group(1) if match else text.split()[0].split('(')[0].strip()

def load_and_balance_optimized():
    """Fully optimized data loading and balancing"""
    print("=" * 70)
    print("PHASE 1: DATA LOADING")
    print("=" * 70)
    
    # Load all files at once
    all_data = {}
    total_rows = 0
    
    for source, f_name in FILES.items():
        path = os.path.join(DATA_DIR, f_name)
        if not os.path.exists(path):
            print(f"❌ Error: {f_name} not found!")
            return None
        
        print(f"\nLoading {source}...", end=" ")
        # Read with optimized settings
        df = pd.read_json(path, lines=True, dtype={'output': str, 'input': str})
        
        # Vectorized label cleaning
        df['clean_label'] = df['output'].apply(clean_label_fast)
        
        # Filter empties efficiently
        mask = df['clean_label'].str.len() > 0
        df = df[mask].copy()
        df['source'] = source
        
        all_data[source] = df
        total_rows += len(df)
        print(f"✓ {len(df):,} rows")
    
    print(f"\nTotal rows loaded: {total_rows:,}")
    
    # ======================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: BUILDING INDEX")
    print("=" * 70)
    
    # Build index structure efficiently
    code_index = {}
    
    for source, df in all_data.items():
        print(f"Indexing {source}...", end=" ")
        # Use groupby for vectorized indexing
        for code, group_df in df.groupby('clean_label'):
            if code not in code_index:
                code_index[code] = {}
            code_index[code][source] = group_df.index.tolist()
        print(f"✓")
    
    # Filter valid codes (present in all 3 sources)
    valid_codes = [
        code for code, sources in code_index.items() 
        if len(sources) == 3
    ]
    
    print(f"\nTotal codes: {len(code_index):,}")
    print(f"Valid codes (in all 3 files): {len(valid_codes):,}")
    
    # ======================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: BALANCED SAMPLING")
    print("=" * 70)
    
    # Pre-calculate total size
    total_samples = 0
    for code in valid_codes:
        min_count = min(len(code_index[code][s]) for s in FILES.keys())
        total_samples += min_count * 3
    
    print(f"Expected output size: {total_samples:,} rows")
    
    # Pre-allocate arrays for efficiency
    all_indices_by_source = {source: [] for source in FILES.keys()}
    
    # Sample efficiently
    for code in tqdm(valid_codes, desc="Sampling", unit="code"):
        min_count = min(len(code_index[code][s]) for s in FILES.keys())
        
        for source in FILES.keys():
            indices = np.array(code_index[code][source])
            sampled = np.random.choice(indices, size=min_count, replace=False)
            all_indices_by_source[source].extend(sampled.tolist())
    
    # Concatenate by source first (faster than row-by-row)
    print("\nConcatenating...")
    balanced_dfs = []
    for source in FILES.keys():
        indices = all_indices_by_source[source]
        balanced_dfs.append(all_data[source].loc[indices])
        print(f"  {source}: {len(indices):,} rows")
    
    final_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle
    print("Shuffling...", end=" ")
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("✓")
    
    print(f"\n✓ Final dataset: {len(final_df):,} rows")
    print(f"✓ Unique codes: {final_df['clean_label'].nunique():,}")
    print(f"✓ Balance check:\n{final_df['source'].value_counts()}")
    
    return final_df

def process_embeddings_optimized(df):
    """Optimized embedding generation with smart batching"""
    print("\n" + "=" * 70)
    print("PHASE 4: EMBEDDING GENERATION")
    print("=" * 70)
    
    # Detect optimal device
    if torch.cuda.is_available():
        device = 'cuda'
        batch_size = EMBEDDING_BATCH_SIZE
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        batch_size = 128
        print(f"✓ Using CPU")
    
    print(f"✓ Batch size: {batch_size}")
    
    # Load model once
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    model.eval()  # Set to eval mode for inference
    
    # Encode all texts
    texts = df['input'].tolist()
    print(f"✓ Encoding {len(texts):,} texts...")
    
    with torch.no_grad():  # Disable gradients for faster inference
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False  # We'll normalize if needed
        )
    
    print(f"✓ Generated embeddings: {embeddings.shape}")
    
    # ======================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: DIMENSIONALITY REDUCTION")
    print("=" * 70)
    
    print(f"Reducing from {embeddings.shape[1]} to {PCA_DIMENSIONS} dimensions...")
    
    # Use regular PCA for speed (IncrementalPCA is for datasets that don't fit in memory)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIMENSIONS)
    X_reduced = pca.fit_transform(embeddings)
    
    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"✓ Reduced shape: {X_reduced.shape}")
    print(f"✓ Variance explained: {variance_explained:.4f} ({variance_explained*100:.2f}%)")
    
    return X_reduced

def save_dataset(X, y, classes, df):
    """Save processed dataset with metadata"""
    print("\n" + "=" * 70)
    print("PHASE 6: SAVING")
    print("=" * 70)
    
    output_file = 'kan_deep_data.pt'
    
    # Create save dictionary
    save_dict = {
        'inputs': torch.tensor(X, dtype=torch.float32),
        'labels': torch.tensor(y, dtype=torch.long),
        'classes': classes,
        'metadata': {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(classes),
            'sources': df['source'].value_counts().to_dict(),
            'pca_dimensions': PCA_DIMENSIONS,
            'class_distribution': pd.Series(y).value_counts().to_dict()
        }
    }
    
    print(f"Saving to {output_file}...", end=" ")
    torch.save(save_dict, output_file)
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ {file_size_mb:.2f} MB")
    
    return output_file, file_size_mb

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import time
    
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "MAXIMUM PERFORMANCE DATA PREP" + " " * 24 + "║")
    print("╚" + "=" * 68 + "╝\n")
    
    start_time = time.time()
    
    # Phase 1-3: Load and balance
    df = load_and_balance_optimized()
    
    if df is None or len(df) == 0:
        print("\n❌ Failed to load data. Check your files.")
        exit(1)
    
    # Phase 4-5: Generate embeddings and reduce dimensions
    X = process_embeddings_optimized(df)
    
    # Encode labels
    print("\nEncoding labels...", end=" ")
    le = LabelEncoder()
    y = le.fit_transform(df['clean_label'])
    print(f"✓ {len(le.classes_):,} classes")
    
    # Phase 6: Save
    output_file, file_size = save_dataset(X, y, le.classes_, df)
    
    # Final summary
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Output file: {output_file}")
    print(f"✓ Total samples: {len(X):,}")
    print(f"✓ Features per sample: {X.shape[1]}")
    print(f"✓ Number of classes: {len(le.classes_):,}")
    print(f"✓ File size: {file_size:.2f} MB")
    print(f"✓ Processing time: {minutes}m {seconds}s")
    print(f"✓ Samples per second: {len(X)/elapsed_time:.1f}")
    print("=" * 70)
    print("\n🎉 All done!")