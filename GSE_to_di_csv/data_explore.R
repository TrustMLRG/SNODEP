Sys.setenv(LANGUAGE = "en")

library(Matrix)
library(Seurat)

# Path to the .mtx file and accompanying files
mtx_file <- "/Users/rssantanu/Desktop/codebase/scRNA_datasets/rstudio/pluripotent_stem_cells/GSE167011_IPS_differentiation_raw_counts.mtx"
genes_file <- "/Users/rssantanu/Desktop/codebase/scRNA_datasets/rstudio/pluripotent_stem_cells/GSE167011_IPS_differentiation_genes.txt"
barcodes_file <- "/Users/rssantanu/Desktop/codebase/scRNA_datasets/rstudio/pluripotent_stem_cells/GSE167011_IPS_differentiation_barcodes.txt"

# Read the matrix file
matrix <- readMM(mtx_file)

# Read the genes/features and barcodes
genes <- read.table(genes_file, header = FALSE, stringsAsFactors = FALSE)
barcodes <- read.table(barcodes_file, header = FALSE, stringsAsFactors = FALSE)

# Create a sparse matrix
rownames(matrix) <- barcodes$V1
colnames(matrix) <- genes$V1

# Create a Seurat object (optional, if you want to use Seurat functionalities)
seurat_object <- CreateSeuratObject(counts = matrix)

# Print dimensions
print(dim(matrix))

# Access the 1000th column name (cell name)
print((matrix)[1:10,1:10])

