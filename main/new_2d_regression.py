"""
Run the regression tasks on 2d data, this will train the models
"""

import argparse
import os
import os.path as osp
import time
import random
import pandas as pd

# seed = 2024
# np.random.seed(seed)
# torch.manual_seed(seed)

from data.datasets import DeterministicLotkaVolteraData, CharacterTrajectoriesDataset, Gene_Expression, Metabolites, Flux, Flux_Knockout, Flux_Knockout_Gene_Added, Metabolites_Knockout, Metabolites_Knockout_Gene_Added, Copasi
from models.neural_process import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model', type=str, choices=['np', 'ndp'],
    default='ndp')
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--data', type=str, choices=['deterministic_lv', 'handwriting', 'gene_expression', 'flux', 'metabolites', 'flux_knockout', 'metabolites_knockout', 'flux_knockout_gene_added', 'metabolites_knockout_gene_added', 'copasi'],
    default='gene_expression')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--load', type=eval, choices=[True, False], default=False)
parser.add_argument('--all_poisson', type=bool, default=False)
parser.add_argument('--ordered_time', type=bool, default=True)
parser.add_argument('--encoder_type', type=str, choices=['mlp', 'lstm_last', 'bilstm', 'lstm_all', 'ode_encoder'], default='mlp')
parser.add_argument('--decoder_type', type=str, choices=['ode', 'cde'], default='ode')
parser.add_argument('--pathway', type=str, choices=['m171', 'bcaa', 'mhc-i', 'acetylcholine', 'dopamine', 'ggsl', 'glucose-glutamine', 'glucose-glutaminolysis', 'glucose-tcacycle', 'histamine', 'ironion', 'm171_nad', 'serotonin', 'synthetic'], default='m171')
parser.add_argument('--GSE_number', type=str, choices=['GSE167011', 'GSE_synthetic'], default='GSE167011')
parser.add_argument('--context_range', type=int, nargs=2,  default=[8,8])
parser.add_argument('--extra_target_range', type=int, nargs=2, default=[4,4])
parser.add_argument('--knockout_kind', type=str, choices=['single_gene', 'multiple_gene', 'reaction_knockout'], default='multiple_gene')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--irregular_freq', type=float, default=1, help='percentage of irregularly samples data points')
parser.add_argument('--copasi_model_id', type=str, default='BIOMD0000000705')
args = parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def adjust_path(relative_path):
    """
    Adjusts a relative path to work from the script's location.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, relative_path))


def get_dimensions(pathway, GSE_number):
    out_dict = {}
    num_index= 5

    base_addr = adjust_path(f'./../GSE_to_di_csv/extracted_data/{GSE_number}/{pathway}')
    gene_expr_addr = os.path.join(base_addr, 'd0.csv')
    gene_expr_df = pd.read_csv(gene_expr_addr, index_col=0)
    num_genes = len(gene_expr_df.index)

    # Get indices of 10 most expressed genes
    gene_means = gene_expr_df.mean(axis=1)
    top_10_genes = gene_means.nlargest(num_index).index
    out_dict['top_10_genes_indices'] = [list(gene_expr_df.index).index(gene) for gene in top_10_genes]

    if pathway !='synthetic':
        flux_metabolite_base_addr = adjust_path(f'./../scFEA_repo/knockout_data/{GSE_number}/{pathway}')
        flux_addr = os.path.join(flux_metabolite_base_addr, 'all_genes', 'flux', 'd0.csv')
        flux_df = pd.read_csv(flux_addr, index_col=0)
        total_fluxes = len(flux_df.index) - num_genes

        # Get indices of most impactful fluxes
        flux_impact = flux_df.iloc[:total_fluxes].mean(axis=1)
        top_10_fluxes = flux_impact.nlargest(num_index).index
        out_dict['top_10_fluxes_indices'] = [list(flux_df.index[:total_fluxes]).index(flux) for flux in top_10_fluxes]

        metabolites_addr = os.path.join(flux_metabolite_base_addr, 'all_genes', 'metabolites', 'd0.csv')
        metabolites_df = pd.read_csv(metabolites_addr, index_col=0)
        total_metabolites = len(metabolites_df.index) - num_genes

        # Get indices of most impactful metabolites
        metabolite_impact = metabolites_df.iloc[:total_metabolites].mean(axis=1)
        top_10_metabolites = metabolite_impact.nlargest(num_index).index
        out_dict['top_10_metabolites_indices'] = [list(metabolites_df.index[:total_metabolites]).index(metabolite) for metabolite in top_10_metabolites]

        out_dict['num_fluxes'] = total_fluxes
        out_dict['num_metabolites'] = total_metabolites
    out_dict['num_genes'] = num_genes

    return out_dict

def get_copasi_dimensions():
    df_0= pd.read_csv('./copasi_data/BIOMD0000000105/d0.csv', index_col=0)
    metabolites = df_0.index.values
    out_dict = {}
    out_dict['num_metabolites'] = len(metabolites)
    return out_dict

def run():
    # set device
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(device)

    # need some design choice here
    # Make folder
    # folder = osp.join('results/2d', args.pathway, args.data, args.model, args.exp_name)
    # folder = osp.join('results/2d', args.pathway, args.data, args.knockout_kind, args.exp_name)
    if 'knockout' in args.data:
        folder = osp.join('results_new', args.pathway, args.data, args.knockout_kind, args.exp_name)
    elif args.data == 'copasi':
        folder = osp.join('results_new', args.data, args.copasi_model_id, args.exp_name)
    else:
        folder = osp.join('results_new', args.pathway, args.data, args.exp_name)
    if not osp.exists(folder):
        os.makedirs(folder)

    if args.data == 'copasi':
        y_dims = get_copasi_dimensions()
    else:   
        y_dims= get_dimensions(args.pathway, args.GSE_number)

    viz_index= []

    if args.data== 'gene_expression':
        y_dim = y_dims['num_genes']
    elif args.data== 'metabolites':
        y_dim = y_dims['num_metabolites']
    elif args.data== 'flux':
        y_dim = y_dims['num_fluxes']
    elif args.data== 'metabolites_knockout':
        y_dim = y_dims['num_metabolites']
    elif args.data== 'flux_knockout':
        y_dim = y_dims['num_fluxes']
    elif args.data== 'metabolites_knockout_gene_added':
        y_dim = y_dims['num_genes']+y_dims['num_metabolites']
    elif args.data== 'flux_knockout_gene_added':
        y_dim = y_dims['num_genes']+y_dims['num_fluxes']
    elif args.data== 'copasi':
        y_dim = y_dims['num_metabolites']

    if args.irregular_freq != 1:
        y_dim = y_dim*2

    if args.knockout_kind=='single_gene':
        knockout_folder= 'knockout_data'
    elif args.knockout_kind=='reaction_knockout':
        knockout_folder= 'knockout_reaction_data'
    elif args.knockout_kind=='multiple_gene':
        knockout_folder= 'knockout_data_multiple_genes'

    # Create dataset
    print('Downloading Data')
    if args.data == 'deterministic_lv':
        dataset = DeterministicLotkaVolteraData(alpha=2. / 3, beta=4. / 3, gamma=1., delta=1.,
            num_samples=50)
        initial_x = -0.1
    elif args.data == 'gene_expression':
        data_dir= adjust_path('./../GSE_to_di_csv/extracted_data/{}/{}'.format(args.GSE_number, args.pathway))
        # data_dir= ('./GSE_to_di_csv/extracted_data/{}/{}'.format(args.GSE_number, args.pathway))
        dataset = Gene_Expression(num_samples=1000, data_dir= data_dir)
        initial_x = -0.1
        # [santanu]: what is initial_x
        viz_index= y_dims['top_10_genes_indices']
    elif args.data == 'metabolites':
        data_dir = adjust_path('./../scFEA_repo/{}/{}/{}/{}/{}'.format(knockout_folder, args.GSE_number, args.pathway, 'all_genes', 'metabolites'))
        # data_dir = ('./scFEA_repo/knockout_data/{}/{}/{}/{}'.format(args.GSE_number, args.pathway, 'all_genes', 'metabolites'))
        dataset = Metabolites(num_samples=1000, data_dir=data_dir, num_genes= y_dims['num_genes'])
        initial_x = -0.1
        # [santanu]: what is initial_x
        viz_index= y_dims['top_10_metabolites_indices']
    elif args.data == 'flux':
        data_dir = adjust_path('./../scFEA_repo/{}/{}/{}/{}/{}'.format(knockout_folder, args.GSE_number, args.pathway, 'all_genes', 'flux'))
        # data_dir = ('./scFEA_repo/knockout_data/{}/{}/{}/{}'.format(args.GSE_number, args.pathway, 'all_genes', 'flux'))
        dataset = Flux(num_samples=1000, data_dir=data_dir, num_genes= y_dims['num_genes'])
        initial_x = -0.1
        # [santanu]: what is initial_x
        viz_index= y_dims['top_10_fluxes_indices']

    elif args.data == 'flux_knockout':
        initial_x = -0.1
        # Define the base directory for data
        data_dir = adjust_path('./../scFEA_repo/{}/{}/{}'.format(knockout_folder, args.GSE_number, args.pathway))
        # data_dir = ('./scFEA_repo/knockout_data/{}/{}'.format(args.GSE_number, args.pathway))

        # List all gene folders and ensure 'original' is included in the training set
        all_genes = [gene for gene in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, gene)) and gene != 'all_genes']
        random.shuffle(all_genes)

        # Split the genes into train and test sets (80% train, 20% test)
        split_idx = int(0.8 * len(all_genes))
        train_genes = all_genes[:split_idx] + ['all_genes']
        test_genes = all_genes[split_idx:]

        # Create the dataset and data loaders
        num_samples = 1000
        batch_size = 100
        test_set_size = 200

        train_dataset = Flux_Knockout(num_samples=num_samples, data_dir=data_dir, gene_list=train_genes, num_genes= y_dims['num_genes'])
        test_dataset = Flux_Knockout(num_samples=test_set_size, data_dir=data_dir, gene_list=test_genes, num_genes= y_dims['num_genes'])
        viz_index= y_dims['top_10_fluxes_indices']
    
    elif args.data == 'flux_knockout_gene_added':
        initial_x = -0.1
        # Define the base directory for data
        data_dir = adjust_path('./../scFEA_repo/{}/{}/{}'.format(knockout_folder, args.GSE_number, args.pathway))
        # data_dir = ('./scFEA_repo/knockout_data/{}/{}'.format(args.GSE_number, args.pathway))

        # List all gene folders and ensure 'original' is included in the training set
        all_genes = [gene for gene in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, gene)) and gene != 'all_genes']
        random.shuffle(all_genes)

        # Split the genes into train and test sets (80% train, 20% test)
        split_idx = int(0.8 * len(all_genes))
        train_genes = all_genes[:split_idx] + ['all_genes']
        test_genes = all_genes[split_idx:]

        # Create the dataset and data loaders
        num_samples = 1000
        batch_size = 100
        test_set_size = 200

        train_dataset = Flux_Knockout_Gene_Added(num_samples=num_samples, data_dir=data_dir, gene_list=train_genes, num_genes= y_dims['num_genes'])
        test_dataset = Flux_Knockout_Gene_Added(num_samples=test_set_size, data_dir=data_dir, gene_list=test_genes, num_genes= y_dims['num_genes'])
        viz_index= y_dims['top_10_fluxes_indices']

    elif args.data == 'metabolites_knockout':
        initial_x = -0.1
        # Define the base directory for data
        data_dir = adjust_path('./../scFEA_repo/{}/{}/{}'.format(knockout_folder, args.GSE_number, args.pathway))
        # data_dir = ('./scFEA_repo/knockout_data/{}/{}'.format(args.GSE_number, args.pathway))

        # List all gene folders and ensure 'original' is included in the training set
        all_genes = [gene for gene in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, gene)) and gene != 'all_genes']
        random.shuffle(all_genes)

        # Split the genes into train and test sets (80% train, 20% test)
        split_idx = int(0.8 * len(all_genes))
        train_genes = all_genes[:split_idx] + ['all_genes']
        test_genes = all_genes[split_idx:]

        # Create the dataset and data loaders
        num_samples = 1000
        batch_size = 100
        test_set_size = 200

        train_dataset = Metabolites_Knockout(num_samples=num_samples, data_dir=data_dir, gene_list=train_genes, num_genes= y_dims['num_genes'])
        test_dataset = Metabolites_Knockout(num_samples=test_set_size, data_dir=data_dir, gene_list=test_genes, num_genes= y_dims['num_genes'])
        viz_index= y_dims['top_10_metabolites_indices']

    elif args.data == 'metabolites_knockout_gene_added':
        initial_x = -0.1
        # Define the base directory for data
        data_dir = adjust_path('./../scFEA_repo/{}/{}/{}'.format(knockout_folder, args.GSE_number, args.pathway))
        # data_dir = ('./scFEA_repo/knockout_data/{}/{}'.format(args.GSE_number, args.pathway))

        # List all gene folders and ensure 'original' is included in the training set
        all_genes = [gene for gene in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, gene)) and gene != 'all_genes']
        random.shuffle(all_genes)

        # Split the genes into train and test sets (80% train, 20% test)
        split_idx = int(0.8 * len(all_genes))
        train_genes = all_genes[:split_idx] + ['all_genes']
        test_genes = all_genes[split_idx:]

        # Create the dataset and data loaders
        num_samples = 1000
        batch_size = 100
        test_set_size = 200

        train_dataset = Metabolites_Knockout_Gene_Added(num_samples=num_samples, data_dir=data_dir, gene_list=train_genes, num_genes= y_dims['num_genes'])
        test_dataset = Metabolites_Knockout_Gene_Added(num_samples=test_set_size, data_dir=data_dir, gene_list=test_genes, num_genes= y_dims['num_genes'])
        viz_index= y_dims['top_10_metabolites_indices']        
    elif args.data == 'copasi':
        data_dir = adjust_path('./../copasi_data/{}'.format(args.copasi_model_id))
        dataset = Copasi(num_samples=1000, data_dir=data_dir)
        initial_x = -0.1
        viz_index= []
    else:
        dataset = CharacterTrajectoriesDataset(root_dir='./data/', position=True, velocity=False,
            include_length=False)
        initial_x = -0.1
    initial_x = torch.tensor(initial_x).view(1, 1, 1).to(device)

########################
# [santanu]: shifting it here
    # training
    from torch.utils.data import DataLoader
    from models.training import TimeNeuralProcessTrainer

    if args.data == 'handwriting':
        batch_size = 200
        test_set_size = 400
        context_range = (1, 100)
        extra_target_range = (0, 100)
    
    elif args.data == 'deterministic_lv':
        batch_size = 5
        test_set_size = 10
        context_range = (1, 100)
        extra_target_range = (0, 45)
    elif args.data in ['gene_expression', 'flux', 'metabolites', 'flux_knockout', 'metabolites_knockout', 'flux_knockout_gene_added', 'metabolites_knockout_gene_added', 'copasi']:
        batch_size = 100
        test_set_size = 200
        # context_range = (1, 100)
        # extra_target_range = (0, 45)
        # [santanu]: I'm hardcoding this as of now
        # context_range = (8, 8)
        # extra_target_range = (4, 4)
        context_range = tuple(args.context_range)
        extra_target_range = tuple(args.extra_target_range)
        # import pdb; pdb.set_trace()
########################
    nprocess = None
    irregular_freq = args.irregular_freq
    if args.model == 'np':
        r_dim = 60  # Dimension of representation of context points
        z_dim = 60  # Dimension of sampled latent variable
        h_dim = 60  # Dimension of hidden layers in encoder and decoder
        nprocess = MlpNeuralProcess(y_dim, r_dim, z_dim, h_dim, irregular_freq).to(device)
    elif args.model == 'ndp':
        r_dim = 50  # Dimension of representation of context points
        z_dim = 50  # Dimension of sampled latent variable
        h_dim = 50  # Dimension of hidden layers in encoder and decoder
        L_dim = 10
        if args.all_poisson:
            if args.encoder_type == 'mlp':
                if args.decoder_type == 'ode':
                    nprocess = MlpNeuralODEProcessPoisson(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x, irregular_freq).to(device)
                elif args.decoder_type == 'cde':
                    nprocess = MlpNeuralCDEProcessPoisson(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x, irregular_freq).to(device)
            elif args.encoder_type == 'lstm_last':
                nprocess = MlpNeuralODEProcessPoissonLSTM_last_op(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x, batch_size, irregular_freq).to(device)
            elif args.encoder_type == 'bilstm':
                nprocess = MlpNeuralODEProcessPoissonBILSTM(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x, batch_size, irregular_freq).to(device)
            elif args.encoder_type == 'lstm_all':
                nprocess = MlpNeuralODEProcessPoissonLSTM_all_op(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x, batch_size, irregular_freq).to(device)
            elif args.encoder_type == 'ode_encoder':
                nprocess = MlpNeuralODEProcessPoissonODE_Encoder(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x, batch_size, irregular_freq).to(device)
        else:
            if args.encoder_type == 'mlp':
                if args.decoder_type == 'ode':
                    nprocess = MlpNeuralODEProcess(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x, irregular_freq).to(device)
                elif args.decoder_type == 'cde':
                    nprocess = MlpNeuralCDEProcess(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x, irregular_freq).to(device)
            elif args.encoder_type == 'lstm_last':
                nprocess = MlpNeuralODEProcessLSTM_last_op(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x, batch_size, irregular_freq).to(device)
            elif args.encoder_type == 'bilstm':
                nprocess = MlpNeuralODEProcessBILSTM(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x, batch_size, irregular_freq).to(device)
            elif args.encoder_type == 'lstm_all':
                nprocess = MlpNeuralODEProcessLSTM_all_op(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x, batch_size, irregular_freq).to(device)
            elif args.encoder_type == 'ode_encoder':
                nprocess = MlpNeuralODEProcessODE_Encoder(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x, batch_size, irregular_freq).to(device)


    if args.load:
        nprocess = torch.load(osp.join(folder, 'trained_model.pth')).to(device)
    else:
        torch.save(nprocess, osp.join(folder, 'untrained_model.pth'))

    nparams = np.array([count_parameters(nprocess)])
    print('Parameters = ' + str(nparams))
    np.save(osp.join(folder, 'parameter_count.npy'), nparams)

    if args.data in ['flux_knockout', 'flux_knockout_gene_added', 'metabolites_knockout', 'metabolites_knockout_gene_added']:
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_size=test_set_size, shuffle=False)

        print("Data loaders are ready.")
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset[int(len(dataset) - test_set_size):],
            batch_size=test_set_size, shuffle=False)
    
    optimizer = torch.optim.RMSprop(nprocess.parameters(), lr=args.lr)
    np_trainer = TimeNeuralProcessTrainer(device, nprocess, optimizer,
        num_context_range=context_range, num_extra_target_range=extra_target_range, exp_name=args.exp_name, data_type= args.data, ordered_time=args.ordered_time, irregular_freq=args.irregular_freq, viz_index=viz_index)

    print('Training')
    start_time = time.time()
    np_trainer.train(data_loader, test_data_loader, args.epochs)
    end_time = time.time()
    print('Total time = ' + str(end_time - start_time))

    """
    epoch_neg_ll+=neg_ll.cpu().item()
    epoch_kl+=kl.cpu().item()
    """

    np.save(osp.join(folder, 'training_time.npy'), np.array([end_time - start_time]))
    np.save(osp.join(folder, 'loss_history.npy'), np.array(np_trainer.epoch_loss_history))
    np.save(osp.join(folder, 'test_mse_history.npy'), np.array(np_trainer.epoch_mse_history))
    np.save(osp.join(folder, 'test_logp_history.npy'), np.array(np_trainer.epoch_logp_history))
    np.save(osp.join(folder, 'test_neg_ll.npy'), np.array(np_trainer.epoch_neg_ll_history))
    np.save(osp.join(folder, 'test_kl.npy'), np.array(np_trainer.epoch_kl_history))
    torch.save(nprocess, osp.join(folder, 'trained_model.pth'))


if __name__ == "__main__":
    run()