import torch

from tqdm import tqdm
from typing import Tuple
from random import randint
from torch.distributions.kl import kl_divergence
from torch.distributions import Bernoulli, Normal
from models.utils import context_target_split as cts
from models.utils import create_plots_directory, visualize_gene_expression_histograms
from models.neural_process import TimeNeuralProcess
from torch.utils.data import DataLoader
import random


random.seed(42)
torch.manual_seed(42)


import os

"""
gene_indices based on expression levels: [6, 74, 405, 204, 461, 310, 378, 460, 481, 53, 412, 88, 424, 516, 246, 560, 16, 48, 292, 42, 427, 297, 394, 51, 593, 247, 344, 224, 289, 474, 100, 223, 357, 356, 482, 75, 616, 320, 393, 567, 212, 47, 335, 607, 471, 423, 278, 454, 365, 592, 403, 445, 486, 475, 77, 406, 370, 562, 620, 312, 373, 226, 267]
"""

class TimeNeuralProcessTrainer:
    """
    Class to handle training of Neural Processes.
    Code adapted from https://github.com/EmilienDupont/neural-processes

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.TimeNeuralProcess, neural_process.NeuralODEProcess instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    """
    def __init__(self,
                 device: torch.device,
                 neural_process: TimeNeuralProcess,
                 optimizer: torch.optim.Optimizer,
                 num_context_range: Tuple[int, int],
                 num_extra_target_range: Tuple[int, int],
                 exp_name: str,
                 data_type: str,
                 ordered_time=False,
                 max_context=None,
                 use_all_targets=False,
                 use_y0=True, 
                 irregular_freq=1,
                 viz_index=[]):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.max_context = max_context
        self.use_all_targets = use_all_targets
        self.use_y0 = use_y0
        self.current_epoch= 0
        self.exp_name= exp_name
        self.data_type= data_type
        self.ordered_time=ordered_time
        self.viz_index= viz_index
        self.irregular_freq= irregular_freq

        self.epoch_loss_history = []
        self.epoch_nfe_history = []
        self.epoch_mse_history = []
        self.epoch_logp_history = []
        self.epoch_neg_ll_history= []
        self.epoch_kl_history= []

        self.num_context = randint(*self.num_context_range)
        self.num_extra_target = randint(*self.num_extra_target_range)

        # Adding a random seed for reproducibility; for irregular1 or irregular
        torch.manual_seed(2022)
        if self.irregular_freq != 1:
            num_irregular_points = int((self.num_context+self.num_extra_target) * (1-self.irregular_freq))
            self.irregular_indices = (torch.randperm(self.num_context+self.num_extra_target-1)+1)[:num_irregular_points] # [santanu]: +1 because we don't want to consider the 0 as irregular
            print('indices:', self.irregular_indices)

    def train(self, train_data_loader: DataLoader, val_data_loader: DataLoader, epochs: int):
        """
        Trains Neural (ODE) Process.

        Parameters
        ----------
        train_data_loader : Data loader to use for training
        val_data_loader: Data loader to use for validation
        epochs: Number of epochs to train for
        """
        self.neural_process.train()
        self.plots_dir = create_plots_directory('./../plots', exp_name= self.exp_name)

        for epoch in range(epochs):
            self.current_epoch= epoch
            print(f'Epoch {epoch}')
            self.epoch= epoch
            epoch_loss = self.train_epoch(train_data_loader)
            # some bit about the self.epoch_nfe_history.append if we want to track nfe in training
            self.epoch_loss_history.append(epoch_loss)
            self.eval_epoch(val_data_loader)

    def process_irregular_data(self, y):
        # Two cases
        # 1. The irregularity is fixed always, so it's very sensitive to the indices you choose
        # 2. The irregularity is only fixed for a batch, so you learn different kinds of irregularities for different batches

        # case 1        
        # y[:, self.irregular_indices, :] = 0
        
        # # Create a mask with the same shape as y and set the corresponding mask elements to zero
        # mask = torch.ones_like(y)
        # mask[:, self.irregular_indices, :] = 0
        
        # # Concatenate y and mask along the last dimension
        # y_with_mask = torch.cat((y, mask), dim=-1)
        # return y_with_mask, mask

        # case 2
        num_irregular_points = int((self.num_context+self.num_extra_target) * (1-self.irregular_freq))
        irregular_indices = (torch.randperm(self.num_context+self.num_extra_target-1)+1)[:num_irregular_points] # [santanu]: +1 because we don't want to consider the 0 as irregular
        # print('indices:', irregular_indices)
        y[:, irregular_indices, :] = 0

        # Create a mask with the same shape as y and set the corresponding mask elements to zero
        mask = torch.ones_like(y)
        mask[:, irregular_indices, :] = 0

        # Concatenate y and mask along the last dimension
        y_with_mask = torch.cat((y, mask), dim=-1)
        return y_with_mask, mask

    def train_epoch(self, data_loader):
        epoch_loss = 0.
        self.neural_process.train()
        for i, data in enumerate(tqdm(data_loader)):
            self.optimizer.zero_grad()

            # Extract data
            x, y = data
            points = x.size(1)

            # *********************[santanu]: I'm disabling the random context and target points for NODEP #####********************
            # Sample number of context and target points
            # num_context = randint(*self.num_context_range)
            # num_extra_target = randint(*self.num_extra_target_range)
            num_context = self.num_context
            num_extra_target = self.num_extra_target
            if self.use_all_targets:
                num_extra_target = points - num_context

            # irregular data
            if self.irregular_freq != 1:
                # only for case 2
                random_seed = random.randint(0, 10000)
                torch.manual_seed(random_seed) # [santanu]: for irregular2, so that train irregularity is different for all epochs
                ##################
                y, mask = self.process_irregular_data(y)

            # Create context and target points and apply neural process
            x_context, y_context, x_target, y_target, y0 = (
                cts(x, y, num_context, num_extra_target, use_y0=self.use_y0, ordered_time=self.ordered_time))
           
            y0 = y0.to(self.device)
            x_context = x_context.to(self.device)
            y_context = y_context.to(self.device)
            x_target = x_target.to(self.device)
            y_target = y_target.to(self.device)
            
            p_y_pred, q_target, q_context = (
                self.neural_process(x_context, y_context, x_target, y_target, y0))
            
            if self.irregular_freq != 1:
                loss = self._loss_masked(p_y_pred, y_target, q_target, q_context, mask)
            else:
                loss = self._loss(p_y_pred, y_target, q_target, q_context)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.cpu().item()

        return epoch_loss / len(data_loader)

    def eval_epoch(self, data_loader, context_size=None):
        """Runs in eval mode on the given data loader and uses the whole time series as target."""
        epoch_mse = 0
        epoch_nll = 0
        if context_size is None:
            context_size = randint(*self.num_context_range)

        self.neural_process.eval()
        for i, data in enumerate( tqdm(data_loader)):
            with torch.no_grad():
                x, y = data

                # irregular data
                if self.irregular_freq != 1:
                    # only for case 2
                    torch.manual_seed(2023) # [santanu]: so that the irregular indices are always the same for test set
                    ##################
                    y, mask = self.process_irregular_data(y)

                x_context, y_context, _, _, y0 = cts(x, y, context_size, 0, use_y0=self.use_y0, ordered_time=self.ordered_time)

                y0 = y0.to(self.device)
                x_context = x_context.to(self.device)
                y_context = y_context.to(self.device)

                # Use the whole time series as target.
                x_target = x.to(self.device)
                y_target = y.to(self.device)
                p_y_pred = self.neural_process(x_context, y_context, x_target, y_target, y0)

                # if self.current_epoch % 240==0:
                #     time_points= [13, 14, 15]
                #     # considered_genes = [0, 1, 2]  # Replace with the actual indices of genes you want to consider
                #     # if self.data_type=='gene_expression':
                #     #     #[santanu]: these should be 20 or 10 most expressed genes
                #     #     considered_genes =[406, 370, 562, 620, 312, 373, 226, 267]
                #     # elif self.data_type in ['metabolites', 'flux', 'flux_knockout', 'metabolites_knockout']:
                #     #     considered_genes= [i for i in range(y_target.shape[2])]
                #     # elif self.data_type in ['flux_knockout_gene_added', 'metabolites_knockout_gene_added']:
                #     #     considered_genes= [i for i in range(y_target.shape[2])][:200] # [santanu]: because it'll also give binary values I think, we don't want that but this is for now
                #     # if self.data_type not in ['flux_knockout', 'flux_knockout_gene_added']:
                #     visualize_gene_expression_histograms(y_target, p_y_pred, self.viz_index, data_type=self.data_type, epoch=self.current_epoch, base_save_dir=self.plots_dir, time_points=time_points)
            
                if self.irregular_freq != 1:
                    y_target = y_target[:, :, :mask.shape[2]]
                    nll = self._loss_masked(p_y_pred, y_target, mask= mask)
                    mse = self.masked_mse(y_target, p_y_pred, mask)
                    # mse = ((y_target-p_y_pred.mean)**2).mean()
                    epoch_mse += mse.item()
                else:
                    nll = self._loss(p_y_pred, y_target)
                    mse = ((y_target-p_y_pred.mean)**2).mean()
                    epoch_mse += mse.item()

                epoch_nll += nll.cpu().item()

                # epoch_neg_ll+=neg_ll.cpu().item()
                # epoch_kl+=kl.cpu().item()
                # import pdb; pdb.set_trace()

        epoch_mse = epoch_mse / len(data_loader)
        epoch_nll = epoch_nll / len(data_loader)
        self.epoch_mse_history.append(epoch_mse)
        self.epoch_logp_history.append(epoch_nll)

        return epoch_mse, epoch_nll
    def masked_mse(self, y_target, p_y_pred, mask):
        return (((y_target-p_y_pred.mean)*mask)**2).mean()

    def _loss(self, p_y_pred, y_target, q_target=None, q_context=None):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        if isinstance(p_y_pred, Bernoulli):
            # Pixels might be in (0, 1), but we still treat them as binary
            # so this is a bit of a hack. This is needed because pytorch checks the argument
            # to log_prob is in the support of the Bernoulli distribution (i.e. it is 0 or 1).
            pred = p_y_pred.logits
            loss = torch.nn.BCEWithLogitsLoss(reduction='none')
            nll = loss(pred, y_target).mean(dim=0).sum()
        else:
            # if self.current_epoch==250:
            #     import pdb; pdb.set_trace()
            nll = -p_y_pred.log_prob(y_target).mean(dim=0).sum()

        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        if q_target is None and q_context is None:
            return nll


        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()

        self.epoch_neg_ll_history.append(nll.detach().numpy())
        self.epoch_kl_history.append(kl.detach().numpy())

        return nll + kl
    
    def _loss_masked(self, p_y_pred, y_target, q_target=None, q_context=None, mask=None):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        mask: torch.Tensor
            Shape (batch_size, num_target, y_dim)
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y

        if isinstance(p_y_pred, Bernoulli):
            pred = p_y_pred.logits
            loss = torch.nn.BCEWithLogitsLoss(reduction='none')
            nll = loss(pred, y_target).mean(dim=0).sum()
        else:
            # bring everything to the same, valid, shape
            y_target= y_target[:, :, :mask.shape[2]]
            mask= mask[:, :y_target.shape[1],:]
            nll= -(p_y_pred.log_prob(y_target)*mask).mean(dim=0).sum()
            

            # import pdb; pdb.set_trace()

        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        if q_target is None and q_context is None:
            return nll


        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()

        self.epoch_neg_ll_history.append(nll.detach().numpy())
        self.epoch_kl_history.append(kl.detach().numpy())

        return nll + kl

