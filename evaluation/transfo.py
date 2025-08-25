import pandas as pd

# Fichier d'entrée et de sortie
input_csv = "evaluation/testfranck.csv"
output_csv = "evaluation/output_200.csv"

# Lecture du CSV
df = pd.read_csv(input_csv)

# Sélection des colonnes désirées et des 200 premières lignes
df_filtered = df.loc[:200, ["Term", "LLT Name"]]

# Sauvegarde dans un nouveau CSV
df_filtered.to_csv(output_csv, index=False)
