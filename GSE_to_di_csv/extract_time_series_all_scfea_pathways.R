# Set the environment to English
Sys.setenv(LANGUAGE = "en")

# Load necessary libraries
library(Matrix)
library(Seurat)

# Define the GSE number as a variable
gse_number <- "GSE167011"

# File paths
mtx_file <- file.path("./GEO_files", gse_number, "GSE167011_IPS_differentiation_raw_counts.mtx")
genes_file <- file.path("./GEO_files", gse_number, "GSE167011_IPS_differentiation_genes.txt")
barcodes_file <- file.path("./GEO_files", gse_number, "GSE167011_IPS_differentiation_barcodes.txt")
scFEA_data_dir <- "./scFEA_data"
output_root_dir <- file.path("./extracted_data", gse_number)


# Read the data
matrix <- readMM(mtx_file)
genes <- read.table(genes_file, header = FALSE, stringsAsFactors = FALSE)
barcodes <- read.table(barcodes_file, header = FALSE, stringsAsFactors = FALSE)

# Transpose the matrix
matrix <- t(matrix)

# Set the row and column names
rownames(matrix) <- genes$V1
colnames(matrix) <- barcodes$V1

# Create a Seurat object
seurat_object <- CreateSeuratObject(counts = matrix)

# Normalize the data
seurat_object <- NormalizeData(seurat_object, normalization.method = "LogNormalize", scale.factor = 10000)

# Print dimensions of the matrix
print(dim(GetAssayData(seurat_object, slot = "counts")))

# Check first 10 rows and 10 columns of the matrix
print(GetAssayData(seurat_object, slot = "counts")[1:10, 1:10])


# This part may not be similar for each case
# Get unique time series annotations and sort them
time_points <- sort(unique(gsub("_.*", "", colnames(GetAssayData(seurat_object, slot = "counts")))))
print(time_points)

# Function to filter and write the matrix for each time point
write_timepoint_matrix <- function(time_point, seurat_object, relevant_genes, output_dir = ".") {
  # Filter the columns for the current time point
  columns_of_interest <- grepl(time_point, colnames(GetAssayData(seurat_object, slot = "counts")))
  filtered_matrix <- GetAssayData(seurat_object, slot = "counts")[, columns_of_interest, drop = FALSE]
  
  # Filter the rows for the relevant genes
  rows_of_interest <- rownames(filtered_matrix) %in% relevant_genes
  filtered_matrix <- filtered_matrix[rows_of_interest, , drop = FALSE]
  
  # Convert to data frame for writing to CSV
  df <- as.data.frame(as.matrix(filtered_matrix))
  
  # Write to CSV
  output_file <- file.path(output_dir, paste0(time_point, ".csv"))
  write.csv(df, output_file, row.names = TRUE)
}

# Function to read relevant genes from a .gmt file
read_genes_from_gmt <- function(gmt_file_path) {
  genes <- c()
  con <- file(gmt_file_path, open = "r")
  while (length(line <- readLines(con, n = 1, warn = FALSE)) > 0) {
    parts <- strsplit(line, "\t")[[1]]
    genes <- c(genes, parts[-(1:2)])  # Skip the first two columns (reaction and metabolite names)
  }
  close(con)
  return(unique(genes))
}

# Traverse each folder in the scFEA_data directory
for (folder in list.files(scFEA_data_dir)) {
  folder_path <- file.path(scFEA_data_dir, folder)
  if (dir.exists(folder_path)) {
    for (file in list.files(folder_path, pattern = "*.gmt")) {
      gmt_file_path <- file.path(folder_path, file)
      relevant_genes <- read_genes_from_gmt(gmt_file_path)
      
      # Directory to save the CSV files
      output_dir <- file.path(output_root_dir, folder)
      
      # Create the output directory if it doesn't exist
      if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
      }
      
      # Write CSV files for each time point
      for (i in seq_along(time_points)) {
        time_point <- time_points[i]
        output_file_name <- paste0("d", i - 1, ".csv")
        write_timepoint_matrix(time_point, seurat_object, relevant_genes, output_dir)
        file.rename(file.path(output_dir, paste0(time_point, ".csv")), file.path(output_dir, output_file_name))
      }
      
      print(paste("CSV files for each time point have been created for", folder, "with", file))
    }
  }
}

print("CSV files for each time point and folder have been created.")

