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

SYN_MAP = {
    'diarrhea': 'diarrhoea', 'haemorrhage': 'hemorrhage', 'hyponatraemia': 'hyponatremia',
    'hemorrhage': 'haemorrhage', 'hyponatremia': 'hyponatraemia',
    'cells': 'cell', 'counts': 'count', 'levels': 'level', 'numbers': 'number'
}
DROP_TOK = {'the','a','an','patient','developed','type','ii','2','with','and','et','avec'}

def toks(s: str):
    return [w for w in re.split(r'\s+', s) if w]

def map_syn(tokens):
    return [SYN_MAP.get(t, t) for t in tokens if t not in DROP_TOK]

def level_and_penalty(query_norm: str, cand_text: str):
    if cand_text == query_norm:
        return 0, 0, True
    qset = set(map_syn(toks(query_norm)))
    if not qset:
        return 3, 999, False
    cset = set(map_syn(toks(cand_text)))
    if qset.issubset(cset):
        penalty = max(0, len(cset - qset))
        return 2, penalty, False
    if qset & cset:
        penalty = max(0, len(cset - qset))
        return 3, penalty, False
    return 3, 999, False

def rule_confidence(level: int, penalty: int) -> float:
    if level == 0:
        return 1.0
    if level == 2:
        return max(0.6, 0.9 - 0.05 * min(penalty, 6))
    return max(0.3, 0.7 - 0.05 * min(penalty, 8))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def levenshtein_sim(a: str, b: str):
    a, b = norm(a), norm(b)
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    dp = np.zeros((la+1, lb+1), dtype=np.int32)
    dp[:,0] = np.arange(la+1)
    dp[0,:] = np.arange(lb+1)
    for i in range(1, la+1):
        ai = a[i-1]
        for j in range(1, lb+1):
            cost = 0 if ai == b[j-1] else 1
            dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1, dp[i-1,j-1]+cost)
    dist = dp[la, lb]
    return 1.0 - dist / max(la, lb)

SCORING_ALPHA = 1.0
SCORING_BETA  = 1.0
SCORING_MU    = 0.6
SCORING_GAMMA = 1.8
SCORING_TAU   = 0.1

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

    pairs, spans, cand_rows_per_q = [], [], []
    all_lltnorms = set()
    for i, q in enumerate(queries):
        cands = variants.iloc[idx[i]].copy().reset_index(drop=True)
        cand_rows_per_q.append(cands)
        pairs.extend((q, t) for t in cands['TEXT'].tolist())
        spans.append(len(cands))
        for llt_code in cands['LLT_CODE'].unique():
            info = meta.loc[llt_code]
            all_lltnorms.add(norm(info['LLT_Label']))

    t1 = time.time()
    scores = ce.predict(pairs, batch_size=64, show_progress_bar=False)
    t2 = time.time()

    lltnorm_list = list(all_lltnorms)
    if lltnorm_list:
        llt_embs = bi.encode(lltnorm_list, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
        llt_emb_dict = {lltnorm_list[k]: llt_embs[k] for k in range(len(lltnorm_list))}
    else:
        llt_emb_dict = {}

    out_all, pos = [], 0
    for i, q in enumerate(queries):
        cands = cand_rows_per_q[i].copy()
        qscores = scores[pos:pos+spans[i]]; pos += spans[i]
        cands["ce_logit"] = qscores
        cands["ce_norm"] = sigmoid(qscores)

        llt_norm_map = {}
        for llt_code in cands['LLT_CODE'].unique():
            info = meta.loc[llt_code]
            llt_norm_map[llt_code] = norm(info['LLT_Label'])

        ranking_rows = []
        for llt_code in cands['LLT_CODE'].unique():
            lltn = llt_norm_map[llt_code]
            sub = cands[cands['LLT_CODE'] == llt_code]
            pt_sub = sub[sub['TEXT'] != lltn]
            if len(pt_sub):
                s_pt = float(pt_sub['ce_norm'].max())
                s_pt_logit = float(pt_sub.loc[pt_sub['ce_norm'].idxmax(), 'ce_logit'])
            else:
                s_pt = float(sub['ce_norm'].max()) if len(sub) else 0.0
                s_pt_logit = float(sub.loc[sub['ce_norm'].idxmax(), 'ce_logit']) if len(sub) else 0.0

            E = levenshtein_sim(q, lltn)
            E_gamma = E ** SCORING_GAMMA
            if lltn in llt_emb_dict:
                C = 0.5 * (1.0 + float(np.dot(qe[i], llt_emb_dict[lltn])))
            else:
                C = 0.0
            S_llt = SCORING_MU * E_gamma + (1.0 - SCORING_MU) * C
            exact = 1 if q == lltn else 0
            combined = (s_pt ** SCORING_ALPHA) * (((1.0 - SCORING_TAU) * S_llt + SCORING_TAU * exact) ** SCORING_BETA)
            if exact == 1:
                combined = 1.01

            level, penalty, _ = level_and_penalty(q, lltn)
            ranking_rows.append((combined, s_pt, s_pt_logit, E, C, exact, level, penalty, llt_code))

        order = sorted(ranking_rows, key=lambda t: (t[0], t[1], t[5]), reverse=True)

        out, k = [], 0
        for combined, s_pt, s_pt_logit, E, C, exact, level, penalty, llt_code in order:
            info = meta.loc[llt_code]
            out.append({
                'query': q,
                'LLT_CODE': str(llt_code),
                'LLT_LABEL': info['LLT_Label'],
                'PT_CODE': info['PT_Code'],
                'PT_LABEL': info['PT_Label'],
                'score': float(combined),
                'combined_score': float(combined),
                'score_ce': float(s_pt_logit),
                'score_ce_sigmoid': float(s_pt),
                'sim_edit': float(E),
                'sim_cosine': float(C),
                'level': int(level),
                'penalty': int(penalty if penalty != 999 else -1),
                'exact_match': bool(exact == 1),
                'src': 'AGG'
            })
            k += 1
            if k == top_k:
                break
        out_all.append(out)

    t3 = time.time()
    print(f"Encode queries: {t1-t0:.2f}s | Cross-encoder: {t2-t1:.2f}s | Post: {t3-t2:.2f}s")
    return out_all

def process_csv_file(input_file, output_file, term_column='term', top_k=5):
    df_input = pd.read_csv(input_file)
    if term_column not in df_input.columns:
        raise ValueError(f"Column '{term_column}' not found in input file. Available columns: {list(df_input.columns)}")
    terms = df_input[term_column].fillna('').astype(str).tolist()
    print(f"Processing {len(terms)} terms...")
    results = batch_code_llt(terms, top_k=top_k)
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
                'score': result['score'],
                'combined_score': result['combined_score'],
                'score_ce': result['score_ce'],
                'score_ce_sigmoid': result['score_ce_sigmoid'],
                'sim_edit': result['sim_edit'],
                'sim_cosine': result['sim_cosine'],
                'level': result['level'],
                'penalty': result['penalty'],
                'exact_match': result['exact_match'],
                'src': result['src']
            })
    df_output = pd.DataFrame(output_rows)
    df_output.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Results saved to {output_file}")
    return df_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code medical terms (single or CSV).")
    parser.add_argument("--input", "-i", help="Input CSV path")
    parser.add_argument("--output", "-o", help="Output CSV path")
    parser.add_argument("--column", "-c", default="term", help="Column containing terms (default: term)")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Top K per term (default: 5)")
    args = parser.parse_args()
    if args.input and args.output:
        df = process_csv_file(args.input, args.output, args.column, args.top_k)
        print(f"Saved {len(df)} rows to {args.output}")
    else:
        q = ["GLOBULAR PELLET", "ABDOMINAL PAIN", "ABSCESS DRAINAGE SURGERY (AE SKIN INFECTION)"]
        print(batch_code_llt(q, top_k=3))
