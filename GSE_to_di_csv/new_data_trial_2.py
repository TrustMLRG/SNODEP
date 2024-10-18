import os
import pandas as pd
import numpy as np
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

input_folder = "./extracted_data/GSE180697_BCells"
output_folder = "./GSE180697_BCells_mtx"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

all_data = []
all_genes = set()
all_barcodes = set()

# Process each CSV file
for file in sorted(os.listdir(input_folder)):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(input_folder, file), index_col=0)
        # Filter for AP1 and Allergen
        mask = (df.loc['human_name'] == 'P3') & (df.loc['condition'] == 'Allergen')
        df = df.loc[:, mask]

        # Remove unwanted rows
        df = df.drop(index=['cell_type', 'human_name', 'time', 'condition', 'allergic'])

        all_data.append(df)
        all_genes.update(df.index)
        all_barcodes.update(df.columns)

# Combine all data
combined_df = pd.concat(all_data, axis=1)
combined_df = combined_df.fillna(0)

# Sort genes and barcodes
all_genes = sorted(list(all_genes))
all_barcodes = sorted(list(all_barcodes))

# Create the sparse matrix
matrix = csr_matrix(combined_df.values)

# Write .mtx file
mmwrite(os.path.join(output_folder, "matrix.mtx"), matrix)

# Write genes.txt
with open(os.path.join(output_folder, "genes.txt"), "w") as f:
    for gene in all_genes:
        f.write(f"{gene}\n")

# Write barcodes.txt
with open(os.path.join(output_folder, "barcodes.txt"), "w") as f:
    for barcode in all_barcodes:
        f.write(f"{barcode}\n")

print("Processing complete. Files saved in", output_folder)