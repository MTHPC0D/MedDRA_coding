import streamlit as st
import pandas as pd
from io import StringIO
from query import batch_code_llt  # imports models & resources once

st.set_page_config(page_title="MedDRA Coding", layout="wide")

st.markdown("""
<style>
body { font-family: system-ui, sans-serif; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("MedDRA Auto-Coding")

st.caption("Assurez-vous d'avoir exécuté build_index.py (génère cache/) avant d'utiliser cette interface.")

tabs = st.tabs(["Coder un terme", "Coder un fichier CSV"])

with tabs[0]:
    st.subheader("Terme unique")
    term = st.text_input("Entrez un terme médical à coder", "")
    top_k_single = st.slider("Nombre de propositions (Top K)", 1, 20, 5)
    if st.button("Coder", disabled=not term.strip()):
        results = batch_code_llt([term], top_k=top_k_single)[0]
        if results:
            df_single = pd.DataFrame(results)
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
                st.success(f"Codage terminé: {len(df_out)} lignes (Top {top_k_batch}).")
                st.dataframe(df_out, use_container_width=True)
                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("Télécharger le résultat CSV", data=csv_bytes,
                                   file_name="coded_terms.csv", mime="text/csv")
    else:
        st.info("Chargez un fichier CSV pour commencer.")
