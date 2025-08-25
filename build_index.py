import pandas as pd, numpy as np, unicodedata, re, os
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import joblib

def norm(s):
    s = str(s).lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = s.replace('"',' ')
    s = re.sub(r'[^a-z0-9\s\-\/]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def build_index(meddra_csv: str = "MedDRA_database.csv",
                cache_dir: str = "cache",
                bi_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                n_neighbors: int = 50):
    # Load MedDRA CSV
    df = pd.read_csv(meddra_csv, sep=";", encoding="utf-8", dtype=str)
    df = df[['LLT_Code','LLT_Label','PT_Code','PT_Label']].dropna(subset=['LLT_Code','LLT_Label']).drop_duplicates()
    df['LLT_N'] = df['LLT_Label'].map(norm)

    # Build variants and meta
    variants = df[['LLT_Code','LLT_N']].rename(columns={'LLT_Code':'LLT_CODE','LLT_N':'TEXT'}).drop_duplicates()
    ptv = df[['LLT_Code','PT_Label']].drop_duplicates()
    ptv['TEXT'] = ptv['PT_Label'].map(norm)
    ptv = ptv[['LLT_Code','TEXT']].rename(columns={'LLT_Code':'LLT_CODE'})
    variants = pd.concat([variants, ptv], ignore_index=True).dropna().drop_duplicates()

    meta = (df.drop_duplicates(subset=['LLT_Code'])
              .set_index('LLT_Code')[['LLT_Label','PT_Code','PT_Label']])

    # Save basic cache files
    os.makedirs(cache_dir, exist_ok=True)
    variants_path = os.path.join(cache_dir, "variants.parquet")
    meta_path = os.path.join(cache_dir, "meta.parquet")
    emb_path = os.path.join(cache_dir, "emb.npy")
    nn_path = os.path.join(cache_dir, "nn.joblib")

    variants.to_parquet(variants_path, index=False)
    meta.to_parquet(meta_path)

    # Encode variants and build NN
    bi = SentenceTransformer(bi_model)
    emb = bi.encode(variants['TEXT'].tolist(), normalize_embeddings=True, batch_size=512, show_progress_bar=True)
    np.save(emb_path, emb)

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(emb)
    joblib.dump(nn, nn_path)

    print(f"OK: variants, meta, emb, nn sauvegardés dans {cache_dir}/")
    return {
        "cache_dir": cache_dir,
        "variants": len(variants),
        "emb_shape": tuple(emb.shape),
        "n_neighbors": n_neighbors
    }

if __name__ == "__main__":
    # Allow CLI usage, default params
    build_index()
