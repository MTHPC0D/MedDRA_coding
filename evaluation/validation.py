import pandas as pd
import unicodedata, re

# Fichiers
ref_file = "evaluation/output_50.csv"        # contient Term, LLT Name
prop_file = "evaluation/coded_terms.csv"     # contient Term, Prop1, Prop2, Prop3

out_file = "evaluation/validation_details_50.csv"

# --- Normalisation pour comparaisons robustes ---
def normalize(s):
    s = str(s).strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'\s+', ' ', s)
    return s

# --- 1) Charger la référence et limiter à 50 termes ---
ref = pd.read_csv(ref_file, usecols=["Term", "LLT Name"])
ref_50 = ref.head(50).copy()
ref_50["LLT_norm"] = ref_50["LLT Name"].apply(normalize)

# --- 2) Charger les propositions (chaque ligne = une proposition) ---
cols = ["Term","Rank","norm_term","LLT_Code","LLT_Name","PT_Code","PT_Name","Score"]
props = pd.read_csv(prop_file, header=None, names=cols)

# Sécurités
props["Rank"] = pd.to_numeric(props["Rank"], errors="coerce")
props = props.sort_values(["Term","Rank"], kind="mergesort")

# Garder top-20 propositions par Term (même s'il y en a plus)
top3 = props.groupby("Term", as_index=False).head(20)

# On garde aussi l’info de rang pour stats par position
cand_lists = (top3
              .assign(LLT_norm=lambda d: d["LLT_Name"].apply(normalize))
              .groupby("Term")
              .agg(
                  Candidates=("LLT_norm", list),
                  RankList=("Rank", list),
                  RawCandidates=("LLT_Name", list)  # pour export lisible
              )
              .reset_index())

# --- 3) Joindre ref_50 avec les candidates (left pour compter 0 quand il manque) ---
merged = ref_50.merge(cand_lists, on="Term", how="left")
merged["Candidates"] = merged["Candidates"].apply(lambda x: x if isinstance(x, list) else [])
merged["RawCandidates"] = merged["RawCandidates"].apply(lambda x: x if isinstance(x, list) else [])
merged["RankList"] = merged["RankList"].apply(lambda x: x if isinstance(x, list) else [])

# --- 4) Calculer Validation (1 si LLT_norm ∈ Candidates, sinon 0) + position trouvée ---
def check_hit(row):
    try:
        idx = row["Candidates"].index(row["LLT_norm"])
        return 1, int(row["RankList"][idx]) if idx < len(row["RankList"]) else None
    except ValueError:
        return 0, None

merged[["Validation","FoundAtRank"]] = merged.apply(
    lambda r: pd.Series(check_hit(r)), axis=1
)

# --- 5) Résultats globaux ---
total_terms = len(ref_50)  # toujours 50 par ta contrainte
hits = int(merged["Validation"].sum())
acc = hits / total_terms if total_terms else 0.0

# Stats par rang (1,2,3)
by_rank = merged["FoundAtRank"].value_counts(dropna=True).sort_index()

print(f"Validations sur 50 termes : {hits}/{total_terms}  (Accuracy = {acc:.2%})")
for rk, cnt in by_rank.items():
    print(f"  - Trouvé en proposition #{int(rk)} : {int(cnt)}")

# --- 6) Export détaillé ---
merged_out = merged[[
    "Term","LLT Name","RawCandidates","Validation","FoundAtRank"
]]
merged_out.to_csv(out_file, index=False)
print(f"Détails enregistrés dans {out_file}")