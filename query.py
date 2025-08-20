import pandas as pd, numpy as np, unicodedata, re, time, joblib, torch, argparse
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

def process_csv_file(input_file, output_file, term_column='term', top_k=5):
    """
    Process a CSV file containing terms to be coded.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        term_column: Name of the column containing terms to code
        top_k: Number of top results to return per term
    """
    # Read input CSV
    df_input = pd.read_csv(input_file)
    
    if term_column not in df_input.columns:
        raise ValueError(f"Column '{term_column}' not found in input file. Available columns: {list(df_input.columns)}")
    
    # Get terms to process
    terms = df_input[term_column].dropna().tolist()
    
    print(f"Processing {len(terms)} terms...")
    
    # Process terms using existing function
    results = batch_code_llt(terms, top_k=top_k)
    
    # Flatten results into DataFrame
    output_rows = []
    for i, term_results in enumerate(results):
        original_term = terms[i]
        for rank, result in enumerate(term_results, 1):
            output_rows.append({
                'original_term': original_term,
                'rank': rank,
                'query': result['query'],
                'LLT_CODE': result['LLT_CODE'],
                'LLT_LABEL': result['LLT_LABEL'],
                'PT_CODE': result['PT_CODE'],
                'PT_LABEL': result['PT_LABEL'],
                'score': result['score']
            })
    
    # Create output DataFrame
    df_output = pd.DataFrame(output_rows)
    
    # Save to CSV
    df_output.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Results saved to {output_file}")
    
    return df_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code medical terms from CSV file')
    parser.add_argument('--input', '-i', help='Input CSV file path')
    parser.add_argument('--output', '-o', help='Output CSV file path') 
    parser.add_argument('--column', '-c', default='term', help='Column name containing terms to code (default: term)')
    parser.add_argument('--top_k', '-k', type=int, default=5, help='Number of top results per term (default: 5)')
    
    args = parser.parse_args()
    
    if args.input and args.output:
        process_csv_file(args.input, args.output, args.column, args.top_k)
    else:
        # Original example code
        q = ["GLOBULAR PELLET", "ABDOMINAL PAIN", "ABSCESS DRAINAGE SURGERY (AE SKIN INFECTION)"]
        print(batch_code_llt(q, top_k=3))
