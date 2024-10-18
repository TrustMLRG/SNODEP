"""
Run the regression tasks on 2d data, this will train the models
"""

import argparse
import os
import os.path as osp
import time
import random

# seed = 2024
# np.random.seed(seed)
# torch.manual_seed(seed)

from data.datasets import DeterministicLotkaVolteraData, CharacterTrajectoriesDataset, Gene_Expression, Metabolites, Flux, Flux_Knockout, Flux_Knockout_Gene_Added
from models.neural_process import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model', type=str, choices=['np', 'ndp', 'nd2p', 'vndp', 'vnd2p'],
    default='ndp')
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--data', type=str, choices=['deterministic_lv', 'handwriting', 'gene_expression', 'flux', 'metabolites', 'flux_knockout', 'metabolites_knockout', 'flux_knockout_gene_added', 'metabolites_knockout_gene_added'],
    default='gene_expression')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--load', type=eval, choices=[True, False], default=False)
parser.add_argument('--all_poisson', type=bool, default=False)
parser.add_argument('--ordered_time', type=bool, default=False)
parser.add_argument('--lstm_encoder', type=bool, default=False)
args = parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run():
    # set device
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(device)

    # Make folder
    folder = osp.join('results/2d', args.data, args.model, args.exp_name)
    if not osp.exists(folder):
        os.makedirs(folder)

    # Create dataset
    print('Downloading Data')
    if args.data == 'deterministic_lv':
        dataset = DeterministicLotkaVolteraData(alpha=2. / 3, beta=4. / 3, gamma=1., delta=1.,
            num_samples=50)
        initial_x = -0.1
    elif args.data == 'gene_expression':
        dataset = Gene_Expression(num_samples=1000)
        initial_x = -0.1
        # [santanu]: what is initial_x
    elif args.data == 'metabolites':
        dataset = Metabolites(num_samples=1000)
        initial_x = -0.1
        # [santanu]: what is initial_x
    elif args.data == 'flux':
        dataset = Flux(num_samples=1000)
        initial_x = -0.1
        # [santanu]: what is initial_x
    elif args.data == 'flux_knockout':
        initial_x = -0.1
        # Define the base directory for data
        data_dir = '/Users/rssantanu/Desktop/codebase/scFEA/m171_knockout/knockout_simplified'

        # List all gene folders and ensure 'original' is included in the training set
        all_genes = [gene for gene in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, gene)) and gene != 'original']
        random.shuffle(all_genes)

        # Split the genes into train and test sets (80% train, 20% test)
        split_idx = int(0.8 * len(all_genes))
        train_genes = all_genes[:split_idx] + ['original']
        test_genes = all_genes[split_idx:]

        # Create the dataset and data loaders
        num_samples = 1000
        batch_size = 100
        test_set_size = 200

        train_dataset = Flux_Knockout(num_samples=num_samples, data_dir=data_dir, gene_list=train_genes)
        test_dataset = Flux_Knockout(num_samples=test_set_size, data_dir=data_dir, gene_list=test_genes)
    elif args.data == 'flux_knockout_gene_added':
        initial_x = -0.1
        # Define the base directory for data
        data_dir = '/Users/rssantanu/Desktop/codebase/scFEA/m171_knockout/knockout_simplified'

        # List all gene folders and ensure 'original' is included in the training set
        all_genes = [gene for gene in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, gene)) and gene != 'original']
        random.shuffle(all_genes)

        # Split the genes into train and test sets (80% train, 20% test)
        split_idx = int(0.8 * len(all_genes))
        train_genes = all_genes[:split_idx] + ['original']
        test_genes = all_genes[split_idx:]

        # Create the dataset and data loaders
        num_samples = 1000
        batch_size = 100
        test_set_size = 200

        train_dataset = Flux_Knockout_Gene_Added(num_samples=num_samples, data_dir=data_dir, gene_list=train_genes)
        test_dataset = Flux_Knockout_Gene_Added(num_samples=test_set_size, data_dir=data_dir, gene_list=test_genes)
        
    else:
        dataset = CharacterTrajectoriesDataset(root_dir='./data/', position=True, velocity=False,
            include_length=False)
        initial_x = -0.1
    initial_x = torch.tensor(initial_x).view(1, 1, 1).to(device)
    # [santanu]: Change this, it should be dynamic

    if args.data== 'gene_expression':
        y_dim = 623
    elif args.data== 'metabolites':
        y_dim = 70
    elif args.data== 'flux':
        y_dim = 168
    elif args.data== 'metabolites_knockout':
        y_dim = 70
    elif args.data== 'flux_knockout':
        y_dim = 168
    elif args.data== 'metabolites_knockout_gene_added':
        y_dim = 693
    elif args.data== 'flux_knockout_gene_added':
        y_dim = 791

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
    elif args.data in ['gene_expression', 'flux', 'metabolites', 'flux_knockout', 'metabolites_knockout', 'flux_knockout_gene_added', 'metabolites_knockout_gene_added']:
        batch_size = 100
        test_set_size = 200
        # context_range = (1, 100)
        # extra_target_range = (0, 45)
        # [santanu]: I'm hardcoding this as of now
        context_range = (8, 8)
        extra_target_range = (4, 4)
########################
    nprocess = None
    if args.model == 'np':
        r_dim = 60  # Dimension of representation of context points
        z_dim = 60  # Dimension of sampled latent variable
        h_dim = 60  # Dimension of hidden layers in encoder and decoder
        nprocess = MlpNeuralProcess(y_dim, r_dim, z_dim, h_dim).to(device)
    elif args.model == 'ndp':
        r_dim = 50  # Dimension of representation of context points
        z_dim = 50  # Dimension of sampled latent variable
        h_dim = 50  # Dimension of hidden layers in encoder and decoder
        L_dim = 10
        if args.all_poisson:
            if args.lstm_encoder:
                nprocess = MlpNeuralODEProcessPoissonLSTM(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x, batch_size).to(device)
            else:
                nprocess = MlpNeuralODEProcessPoisson(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x).to(device)
        else:
            if args.lstm_encoder:
                nprocess = MlpNeuralODEProcessLSTM(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x, batch_size).to(device)
            else:
                nprocess = MlpNeuralODEProcess(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x).to(device)
    elif args.model == 'nd2p':
        r_dim = 50  # Dimension of representation of context points
        z_dim = 50  # Dimension of sampled latent variable
        h_dim = 50  # Dimension of hidden layers in encoder and decoder
        L_dim = 14
        nprocess = MlpNeuralODE2Process(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x).to(device)
    elif args.model == 'vndp':
        r_dim = 60  # Dimension of representation of context points
        z_dim = 60  # Dimension of sampled latent variable
        h_dim = 60  # Dimension of hidden layers in encoder and decoder
        L_dim = 10
        nprocess = VanillaNeuralODEProcess(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x).to(device)
    elif args.model == 'vnd2p':
        r_dim = 60  # Dimension of representation of context points
        z_dim = 60  # Dimension of sampled latent variable
        h_dim = 60  # Dimension of hidden layers in encoder and decoder
        L_dim = 14
        nprocess = VanillaNeuralODE2Process(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x).to(device)

    if args.load:
        nprocess = torch.load(osp.join(folder, 'trained_model.pth')).to(device)
    else:
        torch.save(nprocess, osp.join(folder, 'untrained_model.pth'))

    # # training
    # from torch.utils.data import DataLoader
    # from models.training import TimeNeuralProcessTrainer

    # if args.data == 'handwriting':
    #     batch_size = 200
    #     test_set_size = 400
    #     context_range = (1, 100)
    #     extra_target_range = (0, 100)
    # else:
    #     batch_size = 100
    #     test_set_size = 200
    #     # context_range = (1, 100)
    #     # extra_target_range = (0, 45)
    #     # [santanu]: I'm hardcoding this as of now
    #     context_range = (8, 8)
    #     extra_target_range = (4, 4)


    nparams = np.array([count_parameters(nprocess)])
    print('Parameters = ' + str(nparams))
    np.save(osp.join(folder, 'parameter_count.npy'), nparams)

    if args.data in ['flux_knockout', 'flux_knockout_gene_added']:
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_size=test_set_size, shuffle=False)

        print("Data loaders are ready.")
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset[int(len(dataset) - test_set_size):],
            batch_size=test_set_size, shuffle=False)
    

    optimizer = torch.optim.RMSprop(nprocess.parameters(), lr=1e-3)
    np_trainer = TimeNeuralProcessTrainer(device, nprocess, optimizer,
        num_context_range=context_range, num_extra_target_range=extra_target_range, exp_name=args.exp_name, data_type= args.data, ordered_time=args.ordered_time)

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
