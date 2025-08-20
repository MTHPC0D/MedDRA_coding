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

df = pd.read_csv("MedDRA_database.csv", sep=";", encoding="utf-8", dtype=str)
df = df[['LLT_Code','LLT_Label','PT_Code','PT_Label']].dropna(subset=['LLT_Code','LLT_Label']).drop_duplicates()
df['LLT_N'] = df['LLT_Label'].map(norm)

variants = df[['LLT_Code','LLT_N']].rename(columns={'LLT_Code':'LLT_CODE','LLT_N':'TEXT'}).drop_duplicates()
ptv = df[['LLT_Code','PT_Label']].drop_duplicates()
ptv['TEXT'] = ptv['PT_Label'].map(norm)
ptv = ptv[['LLT_Code','TEXT']].rename(columns={'LLT_Code':'LLT_CODE'})
variants = pd.concat([variants, ptv], ignore_index=True).dropna().drop_duplicates()

meta = (df.drop_duplicates(subset=['LLT_Code'])
          .set_index('LLT_Code')[['LLT_Label','PT_Code','PT_Label']])

os.makedirs("cache", exist_ok=True)
variants.to_parquet("cache/variants.parquet", index=False)
meta.to_parquet("cache/meta.parquet")

bi = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
emb = bi.encode(variants['TEXT'].tolist(), normalize_embeddings=True, batch_size=512, show_progress_bar=True)
np.save("cache/emb.npy", emb)

nn = NearestNeighbors(n_neighbors=50, metric='cosine').fit(emb)
joblib.dump(nn, "cache/nn.joblib")

print("OK: variants, meta, emb, nn sauvegardés dans cache/")
