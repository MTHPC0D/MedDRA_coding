import streamlit as st
import pandas as pd
from io import StringIO
from query import batch_code_llt  # imports models & resources once
# --- added imports for cache status and rebuilding ---
import os, numpy as np, importlib
import build_index as build_mod
# --- end added imports ---

st.set_page_config(page_title="MedDRA Coding", layout="wide")

st.markdown("""
<style>
body { font-family: system-ui, sans-serif; }
.block-container { padding-top: 1.5rem; }
/* Hide Streamlit Deploy and toolbar/menu */
[data-testid="stDeployButton"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
#MainMenu { visibility: hidden; }
header [data-testid="baseButton-header"] { display: none !important; }
/* Hide Deploy modal/popup if it appears */
[data-testid="stModal"] { display: none !important; }
div[data-testid="stModal"] { display: none !important; }
.stDeployButton { display: none !important; }
/* Hide any Deploy-related dialogs */
div[role="dialog"][aria-label*="Deploy"] { display: none !important; }
div[role="dialog"] .stMarkdown:has-text("Deploy this app") { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.title("MedDRA Auto-Coding")

# --- Cache status helpers and UI ---
def get_cache_status(cache_dir: str = "cache"):
    expected = ["variants.parquet", "meta.parquet", "emb.npy", "nn.joblib"]
    paths = {name: os.path.join(cache_dir, name) for name in expected}
    exists = {k: os.path.exists(v) for k, v in paths.items()}
    ok = all(exists.values())
    details = []
    if not ok:
        missing = [k for k, v in exists.items() if not v]
        if missing:
            details.append(f"Fichiers manquants: {', '.join(missing)}")
        return ok, details, paths

    try:
        variants = pd.read_parquet(paths["variants.parquet"])
        emb = np.load(paths["emb.npy"])
        if len(variants) != emb.shape[0]:
            details.append(f"Incohérence: variants={len(variants)} vs emb={emb.shape[0]}")
            ok = False
        else:
            details.append(f"variants={len(variants)} | emb={emb.shape} | cache_dir={cache_dir}")
    except Exception as e:
        ok = False
        details.append(f"Erreur lecture cache: {e}")

    return ok, details, paths

cache_ok, cache_details, cache_paths = get_cache_status()
if cache_ok:
    st.success("Cache prêt.")
    if cache_details:
        with st.expander("Détails du cache"):
            for line in cache_details:
                st.write(line)
else:
    st.warning("Cache manquant ou incomplet.")
    if cache_details:
        with st.expander("Détails"):
            for line in cache_details:
                st.write(line)

csv_present = os.path.exists("MedDRA_database.csv")
if not csv_present:
    st.error("Fichier source 'MedDRA_database.csv' introuvable.")

col_a, _ = st.columns([1, 3])
with col_a:
    build_btn = st.button(
        "Construire / Mettre à jour le cache",
        disabled=not csv_present,
        help="Télécharge les modèles et encode les variantes (peut prendre plusieurs minutes)."
    )

if build_btn:
    with st.spinner("Construction de l'index en cours..."):
        info = build_mod.build_index(meddra_csv="MedDRA_database.csv", cache_dir="cache")
    # Re-évalue le statut
    cache_ok, cache_details, cache_paths = get_cache_status()
    if cache_ok:
        st.success("Cache reconstruit.")
        # Recharge le module query pour prendre en compte le nouveau cache
        try:
            import query as query_module
            importlib.reload(query_module)
            globals()['batch_code_llt'] = query_module.batch_code_llt
            st.caption("Moteur rechargé.")
        except Exception as e:
            st.warning(f"Cache reconstruit, rechargement du moteur impossible: {e}")
    else:
        st.error("La reconstruction du cache n'a pas abouti. Consultez les logs de la console.")

# --- end cache status UI ---

tabs = st.tabs(["Coder un terme", "Coder un fichier CSV"])

with tabs[0]:
    st.subheader("Terme unique")
    term = st.text_input("Entrez un terme médical à coder", "")
    top_k_single = st.slider("Nombre de propositions (Top K)", 1, 20, 5)
    if st.button("Coder", disabled=not term.strip()):
        results = batch_code_llt([term], top_k=top_k_single)[0]
        if results:
            df_single = pd.DataFrame(results)
            hidden_cols = ['score_ce', 'score_ce_sigmoid', 'combined_score', 'level', 'penalty']
            df_single = df_single.drop(columns=[c for c in hidden_cols if c in df_single.columns])
            st.dataframe(df_single, use_container_width=True)
        else:
            st.info("Aucun résultat.")

with tabs[1]:
    st.subheader("Batch CSV")
    uploaded = st.file_uploader("Charger un fichier CSV", type=["csv"])
    if uploaded:
        try:
            raw = uploaded.read().decode("utf-8", errors="ignore")
            df_in = pd.read_csv(StringIO(raw))
        except Exception as e:
            st.error(f"Erreur lecture CSV: {e}")
            df_in = None
        if df_in is not None:
            st.write(f"Lignes: {len(df_in)}")
            col = st.selectbox("Colonne contenant les termes", options=df_in.columns.tolist())
            top_k_batch = st.slider("Top K par terme", 1, 20, 5, key="batch_topk")
            run = st.button("Lancer le codage")
            if run:
                terms = df_in[col].fillna('').astype(str).tolist()
                results = batch_code_llt(terms, top_k=top_k_batch)
                rows = []
                for i, term_results in enumerate(results):
                    orig = terms[i]
                    for rank, r in enumerate(term_results, 1):
                        rows.append({
                            'original_term': orig,
                            'rank': rank,
                            'query': r['query'],
                            'LLT_CODE': r['LLT_CODE'],
                            'LLT_LABEL': r['LLT_LABEL'],
                            'PT_CODE': r['PT_CODE'],
                            'PT_LABEL': r['PT_LABEL'],
                            'score': r['score']
                        })
                df_out = pd.DataFrame(rows)
                hidden_cols = ['score_ce', 'score_ce_sigmoid', 'combined_score', 'level', 'penalty']
                df_out = df_out.drop(columns=[c for c in hidden_cols if c in df_out.columns])
                st.success(f"Codage terminé: {len(df_out)} lignes (Top {top_k_batch}).")
                st.dataframe(df_out, use_container_width=True)
                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("Télécharger le résultat CSV", data=csv_bytes,
                                   file_name="coded_terms.csv", mime="text/csv")
    else:
        st.info("Chargez un fichier CSV pour commencer.")
