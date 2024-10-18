import pandas as pd
import os
import re
import gzip
from tqdm import tqdm


def process_csv(file_path, is_allergic):
    print(f'Reading data from: {file_path}')
    with gzip.open(file_path, 'rt') as f:
        data = pd.read_csv(f, index_col=0)
    
    time_mapping = {
        '0h': 'd0', '12h': 'd0.5', 'day1': 'd1', 'day2': 'd2',
        'day3': 'd3', 'day5': 'd5', 'day7': 'd7',
        '1D':'d1', '2D':'d2', '3D': 'd3', '4D':'d4', '5D':'d5', '6D':'d6', '7D':'d7'
    }
    print('Processing column names')
    new_rows = {'cell_type':[], 'human_name':[], 'time':[], 'condition':[], 'allergic':[]}
    for col in data.columns:
        parts = col.split('_')
        new_rows['cell_type'].append(parts[0])
        new_rows['human_name'].append(parts[1])
        new_rows['time'].append(time_mapping[parts[2]])
        new_rows['condition'].append(parts[3])
        new_rows['allergic'].append(bool(is_allergic))
    
    addon_df= pd.DataFrame(new_rows).transpose()
    addon_df.columns= data.columns

    combined_df = pd.concat([data, addon_df], axis=0)

    print('Done')
    return combined_df

# # Process both CSV files
# allergic_data = process_csv("./GEO_files/GSE180697/GSE180697_allergic.csv", True)
# healthy_data = process_csv("./GEO_files/GSE180697/GSE180697_healthy.csv", False)

# Process both CSV files
allergic_data = process_csv("./GEO_files/GSE180697/GSE180697_SAR_patients_expression_matrix.csv.gz", True)
healthy_data = process_csv("./GEO_files/GSE180697/GSE180697_Healthy_controls_expression_matrix.csv.gz", False)
# Combine the data, filling missing values with zero
all_data = pd.concat([allergic_data, healthy_data], axis=1, join='outer').fillna(0)
# import pdb; pdb.set_trace()


# Get unique cell types
cell_types = all_data.loc['cell_type'].unique()
# import pdb; pdb.set_trace()
# Create output directory
output_dir = "extracted_data"
os.makedirs(output_dir, exist_ok=True)

# Process each cell type
for cell_type in cell_types:
    # Create directory for cell type
    cell_type_dir = os.path.join(output_dir, f"GSE180697_{cell_type}")
    os.makedirs(cell_type_dir, exist_ok=True)
    
    # Get data for this cell type
    # cell_type_data = all_data[all_data.loc['cell_type'] == cell_type]
    cell_type_data = all_data.loc[:, all_data.loc['cell_type'] == cell_type] 
    # import pdb; pdb.set_trace()   
    # Get unique di values
    di_values = cell_type_data.loc['time'].unique()
    
    # Process each di value
    for di in di_values:
        # Get data for this di value
        di_data = cell_type_data.loc[:, cell_type_data.loc['time'] == di]
        
        # # Pivot the data
        # pivoted_data = di_data.pivot(index='gene', columns='cell', values='expression')
        
        # Write to CSV
        output_file = os.path.join(cell_type_dir, f"{di}.csv")
        # pivoted_data.to_csv(output_file)
        di_data.to_csv(output_file)
    print(f"Processed {cell_type}")

print("All data processed and saved.")