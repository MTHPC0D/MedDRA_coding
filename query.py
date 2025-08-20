import pandas as pd, numpy as np, unicodedata, re, time, joblib, torch
from sentence_transformers import SentenceTransformer, CrossEncoder

def norm(s):
    s = str(s).lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = s.replace('"',' ')
    s = re.sub(r'[^a-z0-9\s\-\/]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

sev = {'mild','moderate','severe','very severe','grade i','grade ii','grade iii','grade iv','grade 1','grade 2','grade 3','grade 4'}
def strip_severity(s):
    t = norm(s).split()
    t = [x for x in t if x not in sev]
    return ' '.join(t) if t else norm(s)

variants = pd.read_parquet("cache/variants.parquet")
meta = pd.read_parquet("cache/meta.parquet")
emb = np.load("cache/emb.npy")
nn = joblib.load("cache/nn.joblib")

bi = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2',
                  device='cuda' if torch.cuda.is_available() else 'cpu')

def batch_code_llt(verbatims, top_k=5, n_neighbors=50):
    queries = [strip_severity(v) for v in verbatims]
    t0 = time.time()
    qe = bi.encode(queries, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    dist, idx = nn.kneighbors(qe, n_neighbors=n_neighbors)
    pairs, spans = [], []
    for i, q in enumerate(queries):
        texts = variants.iloc[idx[i]]['TEXT'].tolist()
        pairs.extend((q, t) for t in texts)
        spans.append(len(texts))
    t1 = time.time()
    scores = ce.predict(pairs, batch_size=64, show_progress_bar=False)
    t2 = time.time()
    out_all, pos = [], 0
    for i, q in enumerate(queries):
        qscores = scores[pos:pos+spans[i]]
        pos += spans[i]
        cands = variants.iloc[idx[i]]
        order = np.argsort(qscores)[::-1]
        seen, out = set(), []
        for j in order:
            llt = str(cands.iloc[j]['LLT_CODE'])
            if llt in seen: continue
            info = meta.loc[llt]
            out.append({'query': q,
                        'LLT_CODE': llt,
                        'LLT_LABEL': info['LLT_Label'],
                        'PT_CODE': info['PT_Code'],
                        'PT_LABEL': info['PT_Label'],
                        'score': float(qscores[j])})
            seen.add(llt)
            if len(out) == top_k: break
        out_all.append(out)
    t3 = time.time()
    print(f"Encode queries: {t1-t0:.2f}s | Cross-encoder: {t2-t1:.2f}s | Post: {t3-t2:.2f}s")
    return out_all

if __name__ == "__main__":
    q = ["GLOBULAR PELLET", "ABDOMINAL PAIN", "ABSCESS DRAINAGE SURGERY (AE SKIN INFECTION)"]
    print(batch_code_llt(q, top_k=3))
