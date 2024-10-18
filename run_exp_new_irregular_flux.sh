# Case 1 Irregularity

# # flux; irregular

# # m171
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_nodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway m171 --irregular_freq 0.8
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type lstm_last --irregular_freq 0.8
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_ode_encoder_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type ode_encoder --irregular_freq 0.8

# python -m main.new_2d_regression --model ndp --exp_name m171_flux_nodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway m171 --irregular_freq 0.6
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type lstm_last --irregular_freq 0.6
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_ode_encoder_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type ode_encoder --irregular_freq 0.6

# python -m main.new_2d_regression --model ndp --exp_name m171_flux_nodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway m171 --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type lstm_last --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_ode_encoder_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type ode_encoder --irregular_freq 0.4 --lr 1e-4

# # bcaa
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_nodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway bcaa --irregular_freq 0.8
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type lstm_last --irregular_freq 0.8
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep_ode_encoder_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type ode_encoder --irregular_freq 0.8

# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_nodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway bcaa --irregular_freq 0.6
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type lstm_last --irregular_freq 0.6
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep_ode_encoder_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type ode_encoder --irregular_freq 0.6

# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_nodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway bcaa --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type lstm_last --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep_ode_encoder_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type ode_encoder --irregular_freq 0.4 --lr 1e-4

# # mhc-i
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_nodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway mhc-i --irregular_freq 0.8
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type lstm_last --irregular_freq 0.8
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep_ode_encoder_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type ode_encoder --irregular_freq 0.8

# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_nodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway mhc-i --irregular_freq 0.6
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type lstm_last --irregular_freq 0.6
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep_ode_encoder_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type ode_encoder --irregular_freq 0.6

# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_nodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway mhc-i --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type lstm_last --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep_ode_encoder_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type ode_encoder --irregular_freq 0.4 --lr 1e-4

# # glucose-tcacycle
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_nodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --irregular_freq 0.8
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type lstm_last --irregular_freq 0.8
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep_ode_encoder_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type ode_encoder --irregular_freq 0.8

# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_nodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --irregular_freq 0.6
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type lstm_last --irregular_freq 0.6
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep_ode_encoder_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type ode_encoder --irregular_freq 0.6

# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_nodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type lstm_last --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep_ode_encoder_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type ode_encoder --irregular_freq 0.4 --lr 1e-4

# # histamine
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_nodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway histamine --irregular_freq 0.8
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type lstm_last --irregular_freq 0.8
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep_ode_encoder_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type ode_encoder --irregular_freq 0.8

# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_nodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway histamine --irregular_freq 0.6
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type lstm_last --irregular_freq 0.6
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep_ode_encoder_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type ode_encoder --irregular_freq 0.6

# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_nodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway histamine --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type lstm_last --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep_ode_encoder_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type ode_encoder --irregular_freq 0.4 --lr 1e-4

# # flux_knockout_gene_added; irregular

# # m171
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_nodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --irregular_freq 0.8 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type lstm_last --irregular_freq 0.8 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_ode_encoder_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type ode_encoder --irregular_freq 0.8 --knockout_kind multiple_gene

# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_nodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --irregular_freq 0.6 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type lstm_last --irregular_freq 0.6 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_ode_encoder_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type ode_encoder --irregular_freq 0.6 --knockout_kind multiple_gene

# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_nodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type lstm_last --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_ode_encoder_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type ode_encoder --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4


# # bcaa
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_nodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --irregular_freq 0.8 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type lstm_last --irregular_freq 0.8 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep_ode_encoder_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type ode_encoder --irregular_freq 0.8 --knockout_kind multiple_gene

# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_nodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --irregular_freq 0.6 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type lstm_last --irregular_freq 0.6 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep_ode_encoder_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type ode_encoder --irregular_freq 0.6 --knockout_kind multiple_gene

# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_nodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type lstm_last --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep_ode_encoder_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type ode_encoder --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4

# # mhc-i
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_nodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --irregular_freq 0.8 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type lstm_last --irregular_freq 0.8 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep_ode_encoder_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type ode_encoder --irregular_freq 0.8 --knockout_kind multiple_gene

# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_nodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --irregular_freq 0.6 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type lstm_last --irregular_freq 0.6 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep_ode_encoder_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type ode_encoder --irregular_freq 0.6 --knockout_kind multiple_gene

# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_nodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type lstm_last --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep_ode_encoder_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type ode_encoder --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4

# # glucose-tcacycle
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_nodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --irregular_freq 0.8 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type lstm_last --irregular_freq 0.8 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep_ode_encoder_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type ode_encoder --irregular_freq 0.8 --knockout_kind multiple_gene

# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_nodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --irregular_freq 0.6 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type lstm_last --irregular_freq 0.6 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep_ode_encoder_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type ode_encoder --irregular_freq 0.6 --knockout_kind multiple_gene

# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_nodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type lstm_last --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep_ode_encoder_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type ode_encoder --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4

# # histamine
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_nodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --irregular_freq 0.8 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type lstm_last --irregular_freq 0.8 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep_ode_encoder_irregular_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type ode_encoder --irregular_freq 0.8 --knockout_kind multiple_gene

# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_nodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --irregular_freq 0.6 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type lstm_last --irregular_freq 0.6 --knockout_kind multiple_gene
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep_ode_encoder_irregular_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type ode_encoder --irregular_freq 0.6 --knockout_kind multiple_gene

# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_nodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type lstm_last --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep_ode_encoder_irregular_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type ode_encoder --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4



# Case 2 Irregularity;; mse was still not masked

# flux_knockout_gene_added; irregular

# m171
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_nodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type lstm_last --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_ode_encoder_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type ode_encoder --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_nodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type lstm_last --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_ode_encoder_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type ode_encoder --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_nodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type lstm_last --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_ode_encoder_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type ode_encoder --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4

# # bcaa
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_nodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type lstm_last --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep_ode_encoder_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type ode_encoder --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_nodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type lstm_last --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep_ode_encoder_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type ode_encoder --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_nodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type lstm_last --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep_ode_encoder_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type ode_encoder --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4

# # mhc-i
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_nodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type lstm_last --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep_ode_encoder_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type ode_encoder --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_nodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type lstm_last --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep_ode_encoder_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type ode_encoder --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_nodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type lstm_last --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep_ode_encoder_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type ode_encoder --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4

# # glucose-tcacycle
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_nodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type lstm_last --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep_ode_encoder_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type ode_encoder --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_nodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type lstm_last --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep_ode_encoder_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type ode_encoder --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_nodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type lstm_last --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep_ode_encoder_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type ode_encoder --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4

# # histamine
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_nodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type lstm_last --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep_ode_encoder_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type ode_encoder --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_nodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type lstm_last --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep_ode_encoder_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type ode_encoder --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_nodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type lstm_last --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep_ode_encoder_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type ode_encoder --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4


# # Flux

# # m171
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_nodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway m171 --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type lstm_last --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_ode_encoder_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type ode_encoder --irregular_freq 0.8 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name m171_flux_nodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway m171 --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type lstm_last --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_ode_encoder_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type ode_encoder --irregular_freq 0.6 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name m171_flux_nodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway m171 --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type lstm_last --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_ode_encoder_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type ode_encoder --irregular_freq 0.4 --lr 1e-4

# # bcaa
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_nodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway bcaa --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type lstm_last --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep_ode_encoder_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type ode_encoder --irregular_freq 0.8 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_nodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway bcaa --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type lstm_last --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep_ode_encoder_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type ode_encoder --irregular_freq 0.6 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_nodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway bcaa --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type lstm_last --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep_ode_encoder_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type ode_encoder --irregular_freq 0.4 --lr 1e-4

# # mhc-i
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_nodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway mhc-i --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type lstm_last --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep_ode_encoder_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type ode_encoder --irregular_freq 0.8 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_nodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway mhc-i --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type lstm_last --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep_ode_encoder_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type ode_encoder --irregular_freq 0.6 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_nodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway mhc-i --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type lstm_last --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep_ode_encoder_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type ode_encoder --irregular_freq 0.4 --lr 1e-4

# # glucose-tcacycle
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_nodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type lstm_last --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep_ode_encoder_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type ode_encoder --irregular_freq 0.8 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_nodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type lstm_last --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep_ode_encoder_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type ode_encoder --irregular_freq 0.6 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_nodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type lstm_last --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep_ode_encoder_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type ode_encoder --irregular_freq 0.4 --lr 1e-4

# # histamine
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_nodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway histamine --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type lstm_last --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep_ode_encoder_irregular2_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type ode_encoder --irregular_freq 0.8 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_nodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway histamine --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type lstm_last --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep_ode_encoder_irregular2_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type ode_encoder --irregular_freq 0.6 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_nodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway histamine --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type lstm_last --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep_ode_encoder_irregular2_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type ode_encoder --irregular_freq 0.4 --lr 1e-4


# Case 3 Irregularity;; mse is also masked

# flux_knockout_gene_added; irregular

# m171
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_nodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type lstm_last --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type ode_encoder --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_nodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type lstm_last --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type ode_encoder --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_nodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type lstm_last --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway m171 --encoder_type ode_encoder --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4

# # bcaa
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_nodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type lstm_last --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type ode_encoder --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_nodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type lstm_last --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type ode_encoder --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_nodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type lstm_last --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway bcaa --encoder_type ode_encoder --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4

# # mhc-i
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_nodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type lstm_last --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type ode_encoder --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_nodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type lstm_last --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type ode_encoder --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_nodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type lstm_last --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway mhc-i --encoder_type ode_encoder --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4

# # glucose-tcacycle
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_nodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type lstm_last --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type ode_encoder --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_nodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type lstm_last --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type ode_encoder --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_nodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type lstm_last --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway glucose-tcacycle --encoder_type ode_encoder --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4

# # histamine
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_nodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type lstm_last --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type ode_encoder --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_nodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type lstm_last --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type ode_encoder --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_nodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type lstm_last --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway histamine --encoder_type ode_encoder --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4

# ironion
# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_knockout_gene_added_nodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway ironion --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_knockout_gene_added_snodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway ironion --encoder_type lstm_last --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway ironion --encoder_type ode_encoder --irregular_freq 0.8 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_knockout_gene_added_nodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway ironion --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_knockout_gene_added_snodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway ironion --encoder_type lstm_last --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway ironion --encoder_type ode_encoder --irregular_freq 0.6 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_knockout_gene_added_nodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway ironion --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_knockout_gene_added_snodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway ironion --encoder_type lstm_last --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux_knockout_gene_added --pathway ironion --encoder_type ode_encoder --irregular_freq 0.4 --lr 1e-4
# # Flux

# # # m171
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_nodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway m171 --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type lstm_last --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_ode_encoder_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type ode_encoder --irregular_freq 0.8 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name m171_flux_nodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway m171 --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type lstm_last --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_ode_encoder_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type ode_encoder --irregular_freq 0.6 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name m171_flux_nodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway m171 --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type lstm_last --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name m171_flux_snodep_ode_encoder_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway m171 --encoder_type ode_encoder --irregular_freq 0.4 --lr 1e-4

# # # bcaa
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_nodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway bcaa --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type lstm_last --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep_ode_encoder_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type ode_encoder --irregular_freq 0.8 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_nodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway bcaa --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type lstm_last --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep_ode_encoder_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type ode_encoder --irregular_freq 0.6 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_nodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway bcaa --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type lstm_last --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name bcaa_flux_snodep_ode_encoder_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway bcaa --encoder_type ode_encoder --irregular_freq 0.4 --lr 1e-4

# # # mhc-i
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_nodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway mhc-i --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type lstm_last --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep_ode_encoder_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type ode_encoder --irregular_freq 0.8 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_nodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway mhc-i --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type lstm_last --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep_ode_encoder_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type ode_encoder --irregular_freq 0.6 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_nodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway mhc-i --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type lstm_last --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name mhc-i_flux_snodep_ode_encoder_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway mhc-i --encoder_type ode_encoder --irregular_freq 0.4 --lr 1e-4

# # # glucose-tcacycle
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_nodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type lstm_last --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep_ode_encoder_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type ode_encoder --irregular_freq 0.8 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_nodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type lstm_last --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep_ode_encoder_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type ode_encoder --irregular_freq 0.6 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_nodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type lstm_last --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name glucose-tcacycle_flux_snodep_ode_encoder_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway glucose-tcacycle --encoder_type ode_encoder --irregular_freq 0.4 --lr 1e-4

# # # histamine
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_nodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway histamine --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type lstm_last --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep_ode_encoder_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type ode_encoder --irregular_freq 0.8 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_nodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway histamine --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type lstm_last --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep_ode_encoder_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type ode_encoder --irregular_freq 0.6 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_nodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway histamine --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type lstm_last --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name histamine_flux_snodep_ode_encoder_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway histamine --encoder_type ode_encoder --irregular_freq 0.4 --lr 1e-4

# ironion
# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_nodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway ironion --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_snodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway ironion --encoder_type lstm_last --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_snodep_ode_encoder_irregular3_freq_0.8 --epochs 500 --ordered_time True --data flux --pathway ironion --encoder_type ode_encoder --irregular_freq 0.8 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_nodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway ironion --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_snodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway ironion --encoder_type lstm_last --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_snodep_ode_encoder_irregular3_freq_0.6 --epochs 500 --ordered_time True --data flux --pathway ironion --encoder_type ode_encoder --irregular_freq 0.6 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_nodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway ironion --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_snodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway ironion --encoder_type lstm_last --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_flux_snodep_ode_encoder_irregular3_freq_0.4 --epochs 500 --ordered_time True --data flux --pathway ironion --encoder_type ode_encoder --irregular_freq 0.4 --lr 1e-4




# ad-hoc
# iron-ion
# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_knockout_gene_added_nodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway ironion --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_knockout_gene_added_snodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway ironion --encoder_type lstm_last --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.8 --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway ironion --encoder_type ode_encoder --irregular_freq 0.8 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_knockout_gene_added_nodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway ironion --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_knockout_gene_added_snodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway ironion --encoder_type lstm_last --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.6 --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway ironion --encoder_type ode_encoder --irregular_freq 0.6 --knockout_kind multiple_gene --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_knockout_gene_added_nodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway ironion --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_knockout_gene_added_snodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway ironion --encoder_type lstm_last --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_knockout_gene_added_snodep_ode_encoder_irregular3_freq_0.4 --epochs 500 --ordered_time True --data metabolites_knockout_gene_added --pathway ironion --encoder_type ode_encoder --irregular_freq 0.4 --knockout_kind multiple_gene --lr 1e-4


# # iron-ion
# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_nodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data metabolites --pathway ironion --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_snodep_irregular3_freq_0.8 --epochs 500 --ordered_time True --data metabolites --pathway ironion --encoder_type lstm_last --irregular_freq 0.8 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_snodep_ode_encoder_irregular3_freq_0.8 --epochs 500 --ordered_time True --data metabolites --pathway ironion --encoder_type ode_encoder --irregular_freq 0.8 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_nodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data metabolites --pathway ironion --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_snodep_irregular3_freq_0.6 --epochs 500 --ordered_time True --data metabolites --pathway ironion --encoder_type lstm_last --irregular_freq 0.6 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_snodep_ode_encoder_irregular3_freq_0.6 --epochs 500 --ordered_time True --data metabolites --pathway ironion --encoder_type ode_encoder --irregular_freq 0.6 --lr 1e-4

# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_nodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data metabolites --pathway ironion --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_snodep_irregular3_freq_0.4 --epochs 500 --ordered_time True --data metabolites --pathway ironion --encoder_type lstm_last --irregular_freq 0.4 --lr 1e-4
# python -m main.new_2d_regression --model ndp --exp_name ironion_metabolites_snodep_ode_encoder_irregular3_freq_0.4 --epochs 500 --ordered_time True --data metabolites --pathway ironion --encoder_type ode_encoder --irregular_freq 0.4 --lr 1e-4