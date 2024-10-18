# M171
# python -m main.new_2d_regression --model ndp --exp_name m171_geneexpr_snodep --epochs 500 --ordered_time True --pathway m171 --encoder_type lstm_last --all_poisson True
# # python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type lstm_last
# # python -m main.new_2d_regression --model ndp --exp_name m171_metabolites_snodep --epochs 500 --ordered_time True --data metabolites --pathway m171 --encoder_type lstm_last
# # # python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_snodep --epochs 500 --ordered_time True --data flux_knockout --pathway m171 --encoder_type lstm_last
# # python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type lstm_last
# # # python -m main.new_2d_regression --model ndp --exp_name m171_metabolites_knockout_snodep --epochs 500 --ordered_time True --data metabolites_knockout --pathway m171 --encoder_type lstm_last
# # python -m main.new_2d_regression --model ndp --exp_name m171_metabolites_knockout_gene_added_snodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway m171 --encoder_type lstm_last

# # python -m main.new_2d_regression --model ndp --exp_name m171_geneexpr_nodep --epochs 500 --ordered_time True --pathway m171
# # python -m main.new_2d_regression --model ndp --exp_name m171_flux_nodep --epochs 500 --ordered_time True --data flux --pathway m171
# # python -m main.new_2d_regression --model ndp --exp_name m171_metabolites_nodep --epochs 500 --ordered_time True --data metabolites --pathway m171
# # # python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_nodep --epochs 500 --ordered_time True --data flux_knockout --pathway m171
# # python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_nodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171
# # # python -m main.new_2d_regression --model ndp --exp_name m171_metabolites_knockout_nodep --epochs 500 --ordered_time True --data metabolites_knockout --pathway m171
# # python -m main.new_2d_regression --model ndp --exp_name m171_metabolites_knockout_gene_added_nodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway m171

# python -m main.new_2d_regression --model np --exp_name m171_geneexpr_np --epochs 500 --ordered_time True --pathway m171
# python -m main.new_2d_regression --model np --exp_name m171_flux_np --epochs 500 --ordered_time True --data flux --pathway m171
# python -m main.new_2d_regression --model np --exp_name m171_metabolites_np --epochs 500 --ordered_time True --data metabolites --pathway m171
# python -m main.new_2d_regression --model np --exp_name m171_flux_knockout_np --epochs 500 --ordered_time True --data flux_knockout --pathway m171
# python -m main.new_2d_regression --model np --exp_name m171_flux_knockout_gene_added_np --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model np --exp_name m171_metabolites_knockout_np --epochs 500 --ordered_time True --data metabolites_knockout --pathway m171
# python -m main.new_2d_regression --model np --exp_name m171_metabolites_knockout_gene_added_np --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway m171 --knockout_kind multiple_gene

# ####################################################################################################################################################################################################

# BCAA
# python -m main.new_2d_regression --model ndp --exp_name bcaa_geneexpr_snodep --epochs 500 --ordered_time True --pathway bcaa --encoder_type lstm_last --all_poisson True
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type lstm_last
# python -m main.new_2d_regression --model ndp --exp_name bcaa_metabolites_snodep --epochs 500 --ordered_time True --data metabolites --pathway bcaa --encoder_type lstm_last
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_snodep --epochs 500 --ordered_time True --data flux_knockout --pathway bcaa --encoder_type lstm_last
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type lstm_last --knockout_kind multiple_gene
# # python -m main.new_2d_regression --model ndp --exp_name bcaa_metabolites_knockout_snodep --epochs 500 --ordered_time True --data metabolites_knockout --pathway bcaa --encoder_type lstm_last
# python -m main.new_2d_regression --model ndp --exp_name bcaa_metabolites_knockout_gene_added_snodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway bcaa --encoder_type lstm_last --knockout_kind multiple_gene

# python -m main.new_2d_regression --model ndp --exp_name bcaa_geneexpr_nodep --epochs 500 --ordered_time True --pathway bcaa
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_nodep --epochs 500 --ordered_time True --data flux --pathway bcaa
# python -m main.new_2d_regression --model ndp --exp_name bcaa_metabolites_nodep --epochs 500 --ordered_time True --data metabolites --pathway bcaa
# # python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_nodep --epochs 500 --ordered_time True --data flux_knockout --pathway bcaa
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_nodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --knockout_kind multiple_gene
# # python -m main.new_2d_regression --model ndp --exp_name bcaa_metabolites_knockout_nodep --epochs 500 --ordered_time True --data metabolites_knockout --pathway bcaa
# python -m main.new_2d_regression --model ndp --exp_name bcaa_metabolites_knockout_gene_added_nodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway bcaa --knockout_kind multiple_gene

# python -m main.new_2d_regression --model np --exp_name bcaa_geneexpr_np --epochs 500 --ordered_time True --pathway bcaa
# python -m main.new_2d_regression --model np --exp_name bcaa_flux_np --epochs 500 --ordered_time True --data flux --pathway bcaa
# python -m main.new_2d_regression --model np --exp_name bcaa_metabolites_np --epochs 500 --ordered_time True --data metabolites --pathway bcaa
# python -m main.new_2d_regression --model np --exp_name bcaa_flux_knockout_np --epochs 500 --ordered_time True --data flux_knockout --pathway bcaa
# python -m main.new_2d_regression --model np --exp_name bcaa_flux_knockout_gene_added_np --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --knockout_kind multiple_gene
# python -m main.new_2d_regression --model np --exp_name bcaa_metabolites_knockout_np --epochs 500 --ordered_time True --data metabolites_knockout --pathway bcaa
# python -m main.new_2d_regression --model np --exp_name bcaa_metabolites_knockout_gene_added_np --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway bcaa --knockout_kind multiple_gene

# ###################################################################################################################################################################################################

# # MHC-I Pathway
# # python -m main.new_2d_regression --model ndp --exp_name mhc-i_geneexpr_snodep --epochs 500 --ordered_time True --pathway mhc-i --encoder_type lstm_last --all_poisson True
# # python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type lstm_last
# # python -m main.new_2d_regression --model ndp --exp_name mhc-i_metabolites_snodep --epochs 500 --ordered_time True --data metabolites --pathway mhc-i --encoder_type lstm_last
# # # python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_snodep --epochs 500 --ordered_time True --data flux_knockout --pathway mhc-i --encoder_type lstm_last
# # python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type lstm_last
# # # python -m main.new_2d_regression --model ndp --exp_name mhc-i_metabolites_knockout_snodep --epochs 500 --ordered_time True --data metabolites_knockout --pathway mhc-i --encoder_type lstm_last
# # python -m main.new_2d_regression --model ndp --exp_name mhc-i_metabolites_knockout_gene_added_snodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway mhc-i --encoder_type lstm_last

# # python -m main.new_2d_regression --model ndp --exp_name mhc-i_geneexpr_nodep --epochs 500 --ordered_time True --pathway mhc-i
# # python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_nodep --epochs 500 --ordered_time True --data flux --pathway mhc-i
# # python -m main.new_2d_regression --model ndp --exp_name mhc-i_metabolites_nodep --epochs 500 --ordered_time True --data metabolites --pathway mhc-i
# # # python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_nodep --epochs 500 --ordered_time True --data flux_knockout --pathway mhc-i
# # python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_nodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i
# # # python -m main.new_2d_regression --model ndp --exp_name mhc-i_metabolites_knockout_nodep --epochs 500 --ordered_time True --data metabolites_knockout --pathway mhc-i
# # python -m main.new_2d_regression --model ndp --exp_name mhc-i_metabolites_knockout_gene_added_nodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway mhc-i

# python -m main.new_2d_regression --model np --exp_name mhc-i_geneexpr_np --epochs 500 --ordered_time True --pathway mhc-i
# python -m main.new_2d_regression --model np --exp_name mhc-i_flux_np --epochs 500 --ordered_time True --data flux --pathway mhc-i
# python -m main.new_2d_regression --model np --exp_name mhc-i_metabolites_np --epochs 500 --ordered_time True --data metabolites --pathway mhc-i
# python -m main.new_2d_regression --model np --exp_name mhc-i_flux_knockout_np --epochs 500 --ordered_time True --data flux_knockout --pathway mhc-i
# python -m main.new_2d_regression --model np --exp_name mhc-i_flux_knockout_gene_added_np --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --knockout_kind multiple_gene
# python -m main.new_2d_regression --model np --exp_name mhc-i_metabolites_knockout_np --epochs 500 --ordered_time True --data metabolites_knockout --pathway mhc-i
# python -m main.new_2d_regression --model np --exp_name mhc-i_metabolites_knockout_gene_added_np --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway mhc-i --knockout_kind multiple_gene

# ####################################################################################################################################################################################################

# # Glucose-TCA Cycle Pathway
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_geneexpr_snodep --epochs 500 --ordered_time True --pathway glucose-tcacycle --encoder_type lstm_last --all_poisson True
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type lstm_last
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_metabolites_snodep --epochs 500 --ordered_time True --data metabolites --pathway glucose-tcacycle --encoder_type lstm_last
# # python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_snodep --epochs 500 --ordered_time True --data flux_knockout --pathway glucose-tcacycle --encoder_type lstm_last
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type lstm_last
# # python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_metabolites_knockout_snodep --epochs 500 --ordered_time True --data metabolites_knockout --pathway glucose-tcacycle --encoder_type lstm_last
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_metabolites_knockout_gene_added_snodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway glucose-tcacycle --encoder_type lstm_last

# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_geneexpr_nodep --epochs 500 --ordered_time True --pathway glucose-tcacycle
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_nodep --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_metabolites_nodep --epochs 500 --ordered_time True --data metabolites --pathway glucose-tcacycle
# # python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_nodep --epochs 500 --ordered_time True --data flux_knockout --pathway glucose-tcacycle
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_nodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle
# # python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_metabolites_knockout_nodep --epochs 500 --ordered_time True --data metabolites_knockout --pathway glucose-tcacycle
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_metabolites_knockout_gene_added_nodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway glucose-tcacycle

# python -m main.new_2d_regression --model np --exp_name glucose-tcacycle_geneexpr_np --epochs 500 --ordered_time True --pathway glucose-tcacycle
# python -m main.new_2d_regression --model np --exp_name glucose-tcacycle_flux_np --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle
# python -m main.new_2d_regression --model np --exp_name glucose-tcacycle_metabolites_np --epochs 500 --ordered_time True --data metabolites --pathway glucose-tcacycle
# python -m main.new_2d_regression --model np --exp_name glucose-tcacycle_flux_knockout_np --epochs 500 --ordered_time True --data flux_knockout --pathway glucose-tcacycle
# python -m main.new_2d_regression --model np --exp_name glucose-tcacycle_flux_knockout_gene_added_np --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --knockout_kind multiple_gene
# python -m main.new_2d_regression --model np --exp_name glucose-tcacycle_metabolites_knockout_np --epochs 500 --ordered_time True --data metabolites_knockout --pathway glucose-tcacycle
# python -m main.new_2d_regression --model np --exp_name glucose-tcacycle_metabolites_knockout_gene_added_np --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway glucose-tcacycle --knockout_kind multiple_gene

# ####################################################################################################################################################################################################
# # Histamine Pathway
# python -m main.new_2d_regression --model ndp --exp_name histamine_geneexpr_snodep --epochs 500 --ordered_time True --pathway histamine --encoder_type lstm_last --all_poisson True
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type lstm_last
# python -m main.new_2d_regression --model ndp --exp_name histamine_metabolites_snodep --epochs 500 --ordered_time True --data metabolites --pathway histamine --encoder_type lstm_last
# # python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_snodep --epochs 500 --ordered_time True --data flux_knockout --pathway histamine --encoder_type lstm_last
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type lstm_last
# # python -m main.new_2d_regression --model ndp --exp_name histamine_metabolites_knockout_snodep --epochs 500 --ordered_time True --data metabolites_knockout --pathway histamine --encoder_type lstm_last
# python -m main.new_2d_regression --model ndp --exp_name histamine_metabolites_knockout_gene_added_snodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway histamine --encoder_type lstm_last

# python -m main.new_2d_regression --model ndp --exp_name histamine_geneexpr_nodep --epochs 500 --ordered_time True --pathway histamine
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_nodep --epochs 500 --ordered_time True --data flux --pathway histamine
# python -m main.new_2d_regression --model ndp --exp_name histamine_metabolites_nodep --epochs 500 --ordered_time True --data metabolites --pathway histamine
# # python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_nodep --epochs 500 --ordered_time True --data flux_knockout --pathway histamine
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_nodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine
# # python -m main.new_2d_regression --model ndp --exp_name histamine_metabolites_knockout_nodep --epochs 500 --ordered_time True --data metabolites_knockout --pathway histamine
# python -m main.new_2d_regression --model ndp --exp_name histamine_metabolites_knockout_gene_added_nodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway histamine

# python -m main.new_2d_regression --model np --exp_name histamine_geneexpr_np --epochs 500 --ordered_time True --pathway histamine
# ####################################################################################################################################################################################################
# # # Iron Ion Pathway
# # python -m main.new_2d_regression --model ndp --exp_name ironion_geneexpr_snodep --epochs 500 --ordered_time True --pathway ironion --encoder_type lstm_last --all_poisson True
# # python -m main.new_2d_regression --model ndp --exp_name ironion_flux_snodep --epochs 500 --ordered_time True --data flux --pathway ironion --encoder_type lstm_last
# # python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_snodep --epochs 500 --ordered_time True --data metabolites --pathway ironion --encoder_type lstm_last
# # # python -m main.new_2d_regression --model ndp --exp_name ironion_flux_knockout_snodep --epochs 500 --ordered_time True --data flux_knockout --pathway ironion --encoder_type lstm_last
# # python -m main.new_2d_regression --model ndp --exp_name ironion_flux_knockout_gene_added_snodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway ironion --encoder_type lstm_last
# # # python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_knockout_snodep --epochs 500 --ordered_time True --data metabolites_knockout --pathway ironion --encoder_type lstm_last
# # python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_knockout_gene_added_snodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway ironion --encoder_type lstm_last

# # python -m main.new_2d_regression --model ndp --exp_name ironion_geneexpr_nodep --epochs 500 --ordered_time True --pathway ironion
# # python -m main.new_2d_regression --model ndp --exp_name ironion_flux_nodep --epochs 500 --ordered_time True --data flux --pathway ironion
# # python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_nodep --epochs 500 --ordered_time True --data metabolites --pathway ironion
# # # python -m main.new_2d_regression --model ndp --exp_name ironion_flux_knockout_nodep --epochs 500 --ordered_time True --data flux_knockout --pathway ironion
# # python -m main.new_2d_regression --model ndp --exp_name ironion_flux_knockout_gene_added_nodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway ironion
# # # python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_knockout_nodep --epochs 500 --ordered_time True --data metabolites_knockout --pathway ironion
# # python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_knockout_gene_added_nodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway ironion

# python -m main.new_2d_regression --model np --exp_name ironion_geneexpr_np --epochs 500 --ordered_time True --pathway ironion
# python -m main.new_2d_regression --model np --exp_name ironion_flux_np --epochs 500 --ordered_time True --data flux --pathway ironion
# python -m main.new_2d_regression --model np --exp_name ironion_metabolites_np --epochs 500 --ordered_time True --data metabolites --pathway ironion
# python -m main.new_2d_regression --model np --exp_name ironion_flux_knockout_np --epochs 500 --ordered_time True --data flux_knockout --pathway ironion
# python -m main.new_2d_regression --model np --exp_name ironion_flux_knockout_gene_added_np --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway ironion --knockout_kind multiple_gene
# python -m main.new_2d_regression --model np --exp_name ironion_metabolites_knockout_np --epochs 500 --ordered_time True --data metabolites_knockout --pathway ironion
# python -m main.new_2d_regression --model np --exp_name ironion_metabolites_knockout_gene_added_np --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway ironion --knockout_kind multiple_gene
# ####################################################################################################################################################################################################
# # Serotonin Pathway
# python -m main.new_2d_regression --model ndp --exp_name serotonin_geneexpr_snodep --epochs 500 --ordered_time True --pathway serotonin --encoder_type lstm_last --all_poisson True
# python -m main.new_2d_regression --model ndp --exp_name serotonin_flux_snodep --epochs 500 --ordered_time True --data flux --pathway serotonin --encoder_type lstm_last
# python -m main.new_2d_regression --model ndp --exp_name serotonin_metabolites_snodep --epochs 500 --ordered_time True --data metabolites --pathway serotonin --encoder_type lstm_last
# # python -m main.new_2d_regression --model ndp --exp_name serotonin_flux_knockout_snodep --epochs 500 --ordered_time True --data flux_knockout --pathway serotonin --encoder_type lstm_last
# python -m main.new_2d_regression --model ndp --exp_name serotonin_flux_knockout_gene_added_snodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway serotonin --encoder_type lstm_last
# # python -m main.new_2d_regression --model ndp --exp_name serotonin_metabolites_knockout_snodep --epochs 500 --ordered_time True --data metabolites_knockout --pathway serotonin --encoder_type lstm_last
# python -m main.new_2d_regression --model ndp --exp_name serotonin_metabolites_knockout_gene_added_snodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway serotonin --encoder_type lstm_last

# python -m main.new_2d_regression --model ndp --exp_name serotonin_geneexpr_nodep --epochs 500 --ordered_time True --pathway serotonin
# python -m main.new_2d_regression --model ndp --exp_name serotonin_flux_nodep --epochs 500 --ordered_time True --data flux --pathway serotonin
# python -m main.new_2d_regression --model ndp --exp_name serotonin_metabolites_nodep --epochs 500 --ordered_time True --data metabolites --pathway serotonin
# # python -m main.new_2d_regression --model ndp --exp_name serotonin_flux_knockout_nodep --epochs 500 --ordered_time True --data flux_knockout --pathway serotonin
# python -m main.new_2d_regression --model ndp --exp_name serotonin_flux_knockout_gene_added_nodep --epochs 500 --ordered_time True --data

# # python -m main.new_2d_regression --model ndp --exp_name serotonin_metabolites_knockout_nodep --epochs 500 --ordered_time True --data metabolites_knockout --pathway serotonin
# python -m main.new_2d_regression --model ndp --exp_name serotonin_metabolites_knockout_gene_added_nodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway serotonin



####################### reaction knockout

# nodep
# flux_knockout_gene_added
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_reaction_knockout_gene_added_nodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_reaction_knockout_gene_added_nodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_reaction_knockout_gene_added_nodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway ironion --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_reaction_knockout_gene_added_nodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_reaction_knockout_gene_added_nodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --knockout_kind multiple_gene

# metabolites_knockout_gene_added
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_metabolites_reaction_knockout_gene_added_nodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway mhc-i --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name m171_metabolites_reaction_knockout_gene_added_nodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway m171 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_reaction_knockout_gene_added_nodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway ironion --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name bcaa_metabolites_reaction_knockout_gene_added_nodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway bcaa --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_metabolites_reaction_knockout_gene_added_nodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway glucose-tcacycle --knockout_kind multiple_gene



# snodep
# flux_knockout_gene_added
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_reaction_knockout_gene_added_snodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type lstm_last --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_reaction_knockout_gene_added_snodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type lstm_last --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_reaction_knockout_gene_added_snodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway ironion --encoder_type lstm_last --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_reaction_knockout_gene_added_snodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type lstm_last --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_reaction_knockout_gene_added_snodep --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type lstm_last --knockout_kind multiple_gene

# metabolites_knockout_gene_added
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_metabolites_reaction_knockout_gene_added_snodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway mhc-i --encoder_type lstm_last --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name m171_metabolites_reaction_knockout_gene_added_snodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway m171 --encoder_type lstm_last --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_reaction_knockout_gene_added_snodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway ironion --encoder_type lstm_last --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name bcaa_metabolites_reaction_knockout_gene_added_snodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway bcaa --encoder_type lstm_last --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_metabolites_reaction_knockout_gene_added_snodep --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway glucose-tcacycle --encoder_type lstm_last --knockout_kind multiple_gene




########################################### synthetic data

# python -m main.new_2d_regression --model ndp --exp_name synthetic_geneexpr_nodep --epochs 1000 --ordered_time True --pathway synthetic --GSE_number GSE_synthetic --lr 1e-5
# python -m main.new_2d_regression --model ndp --exp_name synthetic_geneexpr_snodep --epochs 1000 --ordered_time True --pathway synthetic --encoder_type lstm_last --all_poisson True --GSE_number GSE_synthetic --lr 1e-5
# python -m main.new_2d_regression --model ndp --exp_name synthetic_geneexpr_snodep_ode_encoder --epochs 1000 --ordered_time True --pathway synthetic --encoder_type ode_encoder --all_poisson True --GSE_number GSE_synthetic --lr 1e-5
# python -m main.new_2d_regression --model np --exp_name synthetic_geneexpr_np --epochs 1000 --ordered_time True --pathway synthetic --GSE_number GSE_synthetic --lr 1e-5

# Experiments with --irregular_freq 0.8
# python -m main.new_2d_regression --model ndp --exp_name synthetic_geneexpr_nodep_irregular_0.8 --epochs 1000 --ordered_time True --pathway synthetic --GSE_number GSE_synthetic --lr 1e-5 --irregular_freq 0.8
# python -m main.new_2d_regression --model ndp --exp_name synthetic_geneexpr_snodep_irregular_0.8 --epochs 1000 --ordered_time True --pathway synthetic --encoder_type lstm_last --all_poisson True --GSE_number GSE_synthetic --lr 1e-5 --irregular_freq 0.8
# python -m main.new_2d_regression --model ndp --exp_name synthetic_geneexpr_snodep_ode_encoder_irregular_0.8 --epochs 1000 --ordered_time True --pathway synthetic --encoder_type ode_encoder --all_poisson True --GSE_number GSE_synthetic --lr 1e-5 --irregular_freq 0.8
# python -m main.new_2d_regression --model np --exp_name synthetic_geneexpr_np_irregular_0.8 --epochs 1000 --ordered_time True --pathway synthetic --GSE_number GSE_synthetic --lr 1e-5 --irregular_freq 0.8

# Experiments with --irregular_freq 0.6
# python -m main.new_2d_regression --model ndp --exp_name synthetic_geneexpr_nodep_irregular_0.6 --epochs 1000 --ordered_time True --pathway synthetic --GSE_number GSE_synthetic --lr 1e-5 --irregular_freq 0.6
# python -m main.new_2d_regression --model ndp --exp_name synthetic_geneexpr_snodep_irregular_0.6 --epochs 1000 --ordered_time True --pathway synthetic --encoder_type lstm_last --all_poisson True --GSE_number GSE_synthetic --lr 1e-5 --irregular_freq 0.6
# python -m main.new_2d_regression --model ndp --exp_name synthetic_geneexpr_snodep_ode_encoder_irregular_0.6 --epochs 1000 --ordered_time True --pathway synthetic --encoder_type ode_encoder --all_poisson True --GSE_number GSE_synthetic --lr 1e-5 --irregular_freq 0.6
# python -m main.new_2d_regression --model np --exp_name synthetic_geneexpr_np_irregular_0.6 --epochs 1000 --ordered_time True --pathway synthetic --GSE_number GSE_synthetic --lr 1e-5 --irregular_freq 0.6

# Experiments with --irregular_freq 0.4
# python -m main.new_2d_regression --model ndp --exp_name synthetic_geneexpr_nodep_irregular_0.4 --epochs 1000 --ordered_time True --pathway synthetic --GSE_number GSE_synthetic --lr 1e-5 --irregular_freq 0.4
# python -m main.new_2d_regression --model ndp --exp_name synthetic_geneexpr_snodep_irregular_0.4 --epochs 1000 --ordered_time True --pathway synthetic --encoder_type lstm_last --all_poisson True --GSE_number GSE_synthetic --lr 1e-5 --irregular_freq 0.4
# python -m main.new_2d_regression --model ndp --exp_name synthetic_geneexpr_snodep_ode_encoder_irregular_0.4 --epochs 1000 --ordered_time True --pathway synthetic --encoder_type ode_encoder --all_poisson True --GSE_number GSE_synthetic --lr 1e-5 --irregular_freq 0.4
# python -m main.new_2d_regression --model np --exp_name synthetic_geneexpr_np_irregular_0.4 --epochs 1000 --ordered_time True --pathway synthetic --GSE_number GSE_synthetic --lr 1e-5 --irregular_freq 0.4

# Experiments with --irregular_freq 0.2
# python -m main.new_2d_regression --model ndp --exp_name synthetic_geneexpr_nodep_irregular_0.2 --epochs 1000 --ordered_time True --pathway synthetic --GSE_number GSE_synthetic --lr 1e-5 --irregular_freq 0.2
# python -m main.new_2d_regression --model ndp --exp_name synthetic_geneexpr_snodep_irregular_0.2 --epochs 1000 --ordered_time True --pathway synthetic --encoder_type lstm_last --all_poisson True --GSE_number GSE_synthetic --lr 1e-5 --irregular_freq 0.2
# python -m main.new_2d_regression --model ndp --exp_name synthetic_geneexpr_snodep_ode_encoder_irregular_0.2 --epochs 1000 --ordered_time True --pathway synthetic --encoder_type ode_encoder --all_poisson True --GSE_number GSE_synthetic --lr 1e-5 --irregular_freq 0.2
# python -m main.new_2d_regression --model np --exp_name synthetic_geneexpr_np_irregular_0.2 --epochs 1000 --ordered_time True --pathway synthetic --GSE_number GSE_synthetic --lr 1e-5 --irregular_freq 0.2


########################################### trial sanity check for loss
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_sanity_check_ode_encoder --epochs 500 --ordered_time True --pathway m171 --data flux --encoder_type ode_encoder
python -m main.new_2d_regression --model ndp --exp_name m171_geneexpr_snodep_sanity_check_ode_encoder --epochs 5 --ordered_time True --pathway m171 --encoder_type lstm_last --all_poisson True
# python -m main.new_2d_regression --model ndp --exp_name m171_geneexpr_snodep_sanity_check_general_poisson_lstmode_encoder_0005 --epochs 500 --ordered_time True --pathway m171 --encoder_type ode_encoder --all_poisson True --lr 0.0005

# python -m main.new_2d_regression --model ndp --exp_name m171_geneexpr_snodep_sanity_check_lstm_last_nbinomial --epochs 500 --ordered_time True --pathway m171 --encoder_type lstm_last --all_poisson True --lr 1e-3
# python -m main.new_2d_regression --model ndp --exp_name m171_geneexpr_snodep_sanity_check_ode_encoder_nbinomial_1e4 --epochs 500 --ordered_time True --pathway m171 --encoder_type ode_encoder --all_poisson True --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name m171_geneexpr_snodep_irregular_trial --epochs 500 --ordered_time True --pathway m171 --encoder_type lstm_last --all_poisson True --irregular_freq 0.5
# python -m main.new_2d_regression --model ndp --exp_name m171_geneexpr_snodep_ode_encoder_irregular_trial --epochs 500 --ordered_time True --pathway m171 --encoder_type ode_encoder --all_poisson True --irregular_freq 0.5 --lr 1e-5

# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_irregular_trial --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type lstm_last --irregular_freq 0.5
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_ode_encoder_irregular_trial --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type ode_encoder --irregular_freq 0.5 --lr 1e-5

# python -m main.new_2d_regression --model ndp --exp_name m171_metabolites_snodep_irregular_trial --epochs 500 --ordered_time True --data metabolites --pathway m171 --encoder_type lstm_last --irregular_freq 0.5
# python -m main.new_2d_regression --model ndp --exp_name m171_metabolites_snodep_ode_encoder_irregular_trial --epochs 500 --ordered_time True --data metabolites --pathway m171 --encoder_type ode_encoder --irregular_freq 0.5 --lr 1e-5

# python -m main.new_2d_regression --model ndp --exp_name m171_metabolites_snodep_ode_encoder_irregular_trial --epochs 5 --ordered_time True --data metabolites --pathway m171 --encoder_type ode_encoder --irregular_freq 0.7
# python -m main.new_2d_regression --model ndp --exp_name m171_metabolites_snodep_ode_encoder_irregular_trial --epochs 5 --ordered_time True --data flux --pathway m171 --encoder_type ode_encoder --irregular_freq 0.7
# python -m main.new_2d_regression --model ndp --exp_name m171_metabolites_snodep_ode_encoder_irregular_trial --epochs 5 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type ode_encoder --irregular_freq 0.7


##### new experiments

### redo for knockout experiments

### sanity check for the folder structure
# python -m main.new_2d_regression --model ndp --exp_name m171_geneexpr_nodep_sanity_check --epochs 2 --ordered_time True --pathway m171 --encoder_type lstm_last
# python -m main.new_2d_regression --model ndp --exp_name m171_geneexpr_snodep_sanity_check --epochs 2 --ordered_time True --pathway m171 --encoder_type lstm_last --all_poisson True


# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_sanity_check --epochs 2 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type lstm_last --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name m171_metabolites_knockout_gene_added_snodep_sanity_check --epochs 2 --ordered_time True --data metabolites_knockout_gene_added --pathway m171 --encoder_type lstm_last --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_nodep_sanity_check --epochs 2 --ordered_time True --data flux_knockout_gene_added --pathway m171 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name m171_metabolites_knockout_gene_added_nodep_sanity_check --epochs 2 --ordered_time True --data metabolites_knockout_gene_added --pathway m171 --knockout_kind multiple_gene

# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_sanity_check --epochs 2 --ordered_time True --data flux --pathway m171 --encoder_type lstm_last
# python -m main.new_2d_regression --model ndp --exp_name m171_metabolites_snodep_sanity_check --epochs 2 --ordered_time True --data metabolites --pathway m171 --encoder_type lstm_last
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_nodep_sanity_check --epochs 2 --ordered_time True --data flux --pathway m171
# python -m main.new_2d_regression --model ndp --exp_name m171_metabolites_nodep_sanity_check --epochs 2 --ordered_time True --data metabolites --pathway m171
# python -m main.new_2d_regression --model ndp --exp_name synthetic_geneexpr_snodep_ode_encoder_irregular_freq_0.6_sanity_check --epochs 30 --ordered_time True --pathway synthetic --encoder_type ode_encoder --all_poisson True --GSE_number GSE_synthetic --lr 1e-5 --irregular_freq 0.6

## neural process sanity check
# python -m main.new_2d_regression --model np --exp_name np_synthetic_geneexpr_sanity_check --epochs 30 --ordered_time True --pathway synthetic --GSE_number GSE_synthetic --lr 1e-5


## copasi   

# python -m main.new_2d_regression --model ndp --exp_name copasi_nodep_10_10 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000705 --context_range 10 10 --extra_target_range 10 10
# python -m main.new_2d_regression --model ndp --exp_name copasi_snodep_10_10 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000705 --context_range 10 10 --extra_target_range 10 10 --encoder_type lstm_last
# python -m main.new_2d_regression --model ndp --exp_name copasi_snodep_ode_encoder_10_10 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000705 --context_range 10 10 --extra_target_range 10 10 --encoder_type ode_encoder
# python -m main.new_2d_regression --model np --exp_name copasi_np_10_10 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000705 --context_range 10 10 --extra_target_range 10 10

# python -m main.new_2d_regression --model ndp --exp_name copasi_nodep_10_5_irregular_freq_0.8 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000705 --context_range 10 10 --extra_target_range 5 5 --irregular_freq 0.8
# python -m main.new_2d_regression --model ndp --exp_name copasi_snodep_10_5_irregular_freq_0.8 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000705 --context_range 10 10 --extra_target_range 5 5 --encoder_type lstm_last --irregular_freq 0.8
# python -m main.new_2d_regression --model ndp --exp_name copasi_snodep_ode_encoder_10_5_irregular_freq_0.8 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000705 --context_range 10 10 --extra_target_range 5 5 --encoder_type ode_encoder --irregular_freq 0.8
# python -m main.new_2d_regression --model np --exp_name copasi_np_10_5_irregular_freq_0.8 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000705 --context_range 10 10 --extra_target_range 5 5 --irregular_freq 0.8

# python -m main.new_2d_regression --model ndp --exp_name copasi_nodep_10_5_irregular_freq_0.6 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000705 --context_range 10 10 --extra_target_range 5 5 --irregular_freq 0.6
# python -m main.new_2d_regression --model ndp --exp_name copasi_snodep_10_5_irregular_freq_0.6 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000705 --context_range 10 10 --extra_target_range 5 5 --encoder_type lstm_last --irregular_freq 0.6
# python -m main.new_2d_regression --model ndp --exp_name copasi_snodep_ode_encoder_10_5_irregular_freq_0.6 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000705 --context_range 10 10 --extra_target_range 5 5 --encoder_type ode_encoder --irregular_freq 0.6
# python -m main.new_2d_regression --model np --exp_name copasi_np_10_5_irregular_freq_0.6 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000705 --context_range 10 10 --extra_target_range 5 5 --irregular_freq 0.6

# python -m main.new_2d_regression --model ndp --exp_name copasi_nodep_10_5_irregular_freq_0.4 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000705 --context_range 10 10 --extra_target_range 5 5 --irregular_freq 0.4
# python -m main.new_2d_regression --model ndp --exp_name copasi_snodep_10_5_irregular_freq_0.4 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000705 --context_range 10 10 --extra_target_range 5 5 --encoder_type lstm_last --irregular_freq 0.4
# python -m main.new_2d_regression --model ndp --exp_name copasi_snodep_ode_encoder_10_5_irregular_freq_0.4 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000705 --context_range 10 10 --extra_target_range 5 5 --encoder_type ode_encoder --irregular_freq 0.4
# python -m main.new_2d_regression --model np --exp_name copasi_np_10_5_irregular_freq_0.4 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000705 --context_range 10 10 --extra_target_range 5 5 --irregular_freq 0.4

# python -m main.new_2d_regression --model ndp --exp_name copasi_nodep_40_20 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000105 --context_range 40 40 --extra_target_range 20 20
# python -m main.new_2d_regression --model ndp --exp_name copasi_snodep_40_20 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000105 --context_range 40 40 --extra_target_range 20 20 --encoder_type lstm_last
# python -m main.new_2d_regression --model ndp --exp_name copasi_snodep_ode_encoder_40_20 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000105 --context_range 40 40 --extra_target_range 20 20 --encoder_type ode_encoder
# python -m main.new_2d_regression --model np --exp_name copasi_np_40_20 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000105 --context_range 40 40 --extra_target_range 20 20

# python -m main.new_2d_regression --model ndp --exp_name copasi_nodep_40_20_irregular_freq_0.8 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000105 --context_range 40 40 --extra_target_range 20 20 --irregular_freq 0.8
# python -m main.new_2d_regression --model ndp --exp_name copasi_snodep_40_20_irregular_freq_0.8 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000105 --context_range 40 40 --extra_target_range 20 20 --encoder_type lstm_last --irregular_freq 0.8
# python -m main.new_2d_regression --model ndp --exp_name copasi_snodep_ode_encoder_40_20_irregular_freq_0.8 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000105 --context_range 40 40 --extra_target_range 20 20 --encoder_type ode_encoder --irregular_freq 0.8
# python -m main.new_2d_regression --model np --exp_name copasi_np_40_20_irregular_freq_0.8 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000105 --context_range 40 40 --extra_target_range 20 20 --irregular_freq 0.8

# python -m main.new_2d_regression --model ndp --exp_name copasi_nodep_40_20_irregular_freq_0.6 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000105 --context_range 40 40 --extra_target_range 20 20 --irregular_freq 0.6
# python -m main.new_2d_regression --model ndp --exp_name copasi_snodep_40_20_irregular_freq_0.6 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000105 --context_range 40 40 --extra_target_range 20 20 --encoder_type lstm_last --irregular_freq 0.6
# python -m main.new_2d_regression --model ndp --exp_name copasi_snodep_ode_encoder_40_20_irregular_freq_0.6 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000105 --context_range 40 40 --extra_target_range 20 20 --encoder_type ode_encoder --irregular_freq 0.6
# python -m main.new_2d_regression --model np --exp_name copasi_np_40_20_irregular_freq_0.6 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000105 --context_range 40 40 --extra_target_range 20 20 --irregular_freq 0.6

# python -m main.new_2d_regression --model ndp --exp_name copasi_nodep_40_20_irregular_freq_0.4 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000105 --context_range 40 40 --extra_target_range 20 20 --irregular_freq 0.4
# python -m main.new_2d_regression --model ndp --exp_name copasi_snodep_40_20_irregular_freq_0.4 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000105 --context_range 40 40 --extra_target_range 20 20 --encoder_type lstm_last --irregular_freq 0.4
# python -m main.new_2d_regression --model ndp --exp_name copasi_snodep_ode_encoder_40_20_irregular_freq_0.4 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000105 --context_range 40 40 --extra_target_range 20 20 --encoder_type ode_encoder --irregular_freq 0.4
# python -m main.new_2d_regression --model np --exp_name copasi_np_40_20_irregular_freq_0.4 --epochs 500 --ordered_time True --data copasi --copasi_model_id BIOMD0000000105 --context_range 40 40 --extra_target_range 20 20 --irregular_freq 0.4