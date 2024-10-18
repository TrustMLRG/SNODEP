import torch
from torch import nn
# import torch.nn as nn

from torch.nn import functional as F
from torchdiffeq import odeint
import torchcde
import numpy as np
from models.utils import init_network_weights, create_net, get_device, split_last_dim, check_mask, linspace_vector
from torch.distributions import Bernoulli, Normal, Poisson, LogNormal, Geometric, NegativeBinomial

class SingleContextNDPEncoder(nn.Module):
    """Use Context and the same conv network to infer both L(0) and D"""
    def __init__(self, context_encoder):
        super(SingleContextNDPEncoder, self).__init__()
        self.context_encoder = context_encoder

    def forward(self, x, y, _):
        """
        x : torch.Tensor
            Shape (batch_size * num_points, x_dim)
        y : torch.Tensor
            Shape (batch_size * num_points, y_dim)
        """
        output = self.context_encoder(x, y)
        return output, output
    
class SingleContextNDPEncoder_for_ode_encoder(nn.Module):
    """Use Context and the same conv network to infer both L(0) and D"""
    def __init__(self, context_encoder):
        super(SingleContextNDPEncoder_for_ode_encoder, self).__init__()
        self.context_encoder = context_encoder

    def forward(self, x, y, _):
        """
        x : torch.Tensor
            Shape (batch_size * num_points, x_dim)
        y : torch.Tensor
            Shape (batch_size * num_points, y_dim)
        """
        output = self.context_encoder(x, y)
        return output


class Y0ContextNDPEncoder(nn.Module):
    """Use y0 to infer L(0) and Context to infer D"""
    def __init__(self, y0_encoder, context_encoder):
        super(Y0ContextNDPEncoder, self).__init__()
        self.y0_encoder = y0_encoder
        self.context_encoder = context_encoder

    def forward(self, x, y, y0):
        """
        x : torch.Tensor
            Shape (batch_size * num_points, x_dim)
        y : torch.Tensor
            Shape (batch_size * num_points, y_dim)
        y0 : torch.Tensor
            Shape (batch_size, y_dim)
        """
        L_output = self.y0_encoder(y0)
        D_output = self.context_encoder(x, y)
        return L_output, D_output


class Encoder(nn.Module):
    """Maps an (x_i, y_i) pair to a representation r_i.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dim : int
        Dimension of hidden layer.

    r_dim : int
        Dimension of output representation r.
    """

    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, r_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size*T, x_dim)

        y : torch.Tensor
            Shape (batch_size*T, y_dim)

        output: torch.Tensor
                Shape (batch_size*T, r_dim)
        """
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)

class LSTM_Encoder_all_op(nn.Module):
    """
    Maps an (x_i, y_i) pair to a representation r_i using an LSTM.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dim : int
        Dimension of hidden layer.

    r_dim : int
        Dimension of output representation r.
    """

    def __init__(self, x_dim, y_dim, h_dim, r_dim, batch_size, num_layers=1):
        super(LSTM_Encoder_all_op, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim
        self.num_layers = num_layers
        self.batch_size= batch_size

        self.lstm = nn.LSTM(input_size=x_dim + y_dim, hidden_size=h_dim, num_layers=num_layers, batch_first=True)
        # self.lstm = nn.LSTM(input_size=y_dim, hidden_size=h_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(h_dim, r_dim)  # Assuming T=12 for the concatenation

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size*T, x_dim)

        y : torch.Tensor
            Shape (batch_size*T, y_dim)

        output: torch.Tensor
                Shape (batch_size*T, r_dim)
        """
        # Reshape input to (batch_size, T, x_dim + y_dim)
        batch_size= self.batch_size
        T= x.size(0)//batch_size
        # batch_size = x.size(0) // 12  # Assuming T=12 as per the given example
        # T = 12
        x = x.view(batch_size, T, self.x_dim)
        y = y.view(batch_size, T, self.y_dim)

        input_pairs = torch.cat((x, y), dim=2)
        # input_pairs= y
        # LSTM forward pass
        lstm_out, _ = self.lstm(input_pairs)
        
        # Concatenate the output of each timestep
        lstm_out_concat = lstm_out.contiguous().view(batch_size*T, -1)
        
        # Pass through the fully connected layer
        output = self.fc(lstm_out_concat)
        return output


class LSTM_Encoder_last_op(nn.Module):
    """
    Maps an (x_i, y_i) pair to a representation r_i using an LSTM.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dim : int
        Dimension of hidden layer.

    r_dim : int
        Dimension of output representation r.
    """

    def __init__(self, x_dim, y_dim, h_dim, r_dim, batch_size, num_layers=1):
        super(LSTM_Encoder_last_op, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim
        self.num_layers = num_layers
        self.batch_size= batch_size

        self.lstm = nn.LSTM(input_size=y_dim, hidden_size=h_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(h_dim, r_dim)

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size, T, x_dim)

        y : torch.Tensor
            Shape (batch_size, T, y_dim)

        output: torch.Tensor
                Shape (batch_size*T, r_dim)
        """
        # for backward #
        y = torch.flip(y, dims=[1])
        ###########################
        input_pairs= y
        # LSTM forward pass
        lstm_out, _ = self.lstm(input_pairs)
        
        # We are only interested in the output of the last time step
        lstm_out_last = lstm_out[:, -1, :]
        
        # Pass through the fully connected layer
        output = self.fc(lstm_out_last) # 100 x 50
        return output


class BILSTM_Encoder_last_op(nn.Module):
    """
    Maps an (x_i, y_i) pair to a representation r_i using a bidirectional LSTM.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dim : int
        Dimension of hidden layer.

    r_dim : int
        Dimension of output representation r.

    batch_size : int
        Size of the batch.

    num_layers : int, optional
        Number of layers in the LSTM. Default is 1.
    """

    def __init__(self, x_dim, y_dim, h_dim, r_dim, batch_size, num_layers=1):
        super(BILSTM_Encoder_last_op, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=y_dim, hidden_size=h_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # Fully connected layer
        self.fc = nn.Linear(h_dim * 2, r_dim)  # Multiply by 2 for bidirectional

    def forward(self, x, y):
        """
        Forward pass of the bidirectional LSTM encoder.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, T, x_dim)

        y : torch.Tensor
            Shape (batch_size, T, y_dim)

        Returns
        -------
        output : torch.Tensor
            Shape (batch_size, r_dim)
        """
        input_pairs = y
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(input_pairs)
        
        # We are only interested in the output of the last time step
        # Concatenate the hidden states from the forward and backward LSTMs
        lstm_out_last = torch.cat((lstm_out[:, -1, :self.h_dim], lstm_out[:, 0, self.h_dim:]), dim=-1)
        
        # Pass through the fully connected layer
        output = self.fc(lstm_out_last)
        
        return output

class ODEFunc(nn.Module):
    def __init__(self, r_dim, h_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(r_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, r_dim)
        )

    def forward(self, t, y):
        return self.net(y)

class NeuralODEEncoder(nn.Module):
    """
    Maps an (x_i, y_i) pair to a representation r_i using a neural ODE.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.
    y_dim : int
        Dimension of y values.
    h_dim : int
        Dimension of hidden layer.
    r_dim : int
        Dimension of output representation r.
    """

    def __init__(self, x_dim, y_dim, h_dim, r_dim, batch_size, initial_t=0.0):
        super(NeuralODEEncoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim
        self.initial_t = initial_t
        self.batch_size = batch_size

        self.input_to_hidden = nn.Sequential(
            nn.Linear(x_dim + y_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, r_dim)
        )

        self.odefunc = ODEFunc(r_dim, h_dim)

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)
        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        output: torch.Tensor
                Shape (batch_size, num_points, r_dim)
        """
        batch_size, num_points, _ = x.size()
        
        # Reverse the order of x and y along the time dimension
        x = torch.flip(x, dims=[1])
        y = torch.flip(y, dims=[1])
        
        # Concatenate x and y along the last dimension
        input_pairs = torch.cat((x, y), dim=2)
        
        # Pass through the initial feedforward network
        hidden = self.input_to_hidden(input_pairs)

        # Integrate the ODE
        t = torch.linspace(0, 1, num_points).to(x.device)# actually x is the time
        hidden = odeint(self.odefunc, hidden[:,0,:], t, method='dopri5')

        # Return the final hidden state
        # import pdb; pdb.set_trace()
        return hidden[-1]




class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.
    """

    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, r_dim)
        """
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return mu, sigma


class NPMlpDecoder(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer.

    y_dim : int
        Dimension of y values.
    """

    def __init__(self, x_dim, z_dim, h_dim, y_dim):
        super(NPMlpDecoder, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim

        layers = [nn.Linear(x_dim + z_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu = nn.Linear(h_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim, y_dim)

    def forward(self, x, z, y):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        z : torch.Tensor
            Shape (batch_size, z_dim)

        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (batch_size, num_points, y_dim).
        """
        batch_size, num_points, _ = x.size()
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        z_flat = z.view(batch_size * num_points, self.z_dim)
        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x_flat, z_flat), dim=1)
        hidden = self.xz_to_hidden(input_pairs)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return Normal(mu, sigma)


class AbstractODEDecoder(nn.Module):
    """
    An Abstract Decoder using a Neural ODE. Child classes must implement decode_latent.

    Parameters
    ----------
    x_dim : int
        Dimension of x values. Currently only works for dimension of 1.
    z_dim : int
        Dimension of latent variable z. Contains [L0, z_].

    h_dim : int
        Dimension of hidden layer in odefunc.
    y_dim : int
        Dimension of y values.

    L_dim : int
        Dimension of latent state L.
    """

    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim=None, exclude_time=False):
        super(AbstractODEDecoder, self).__init__()
        # The input is always time.
        assert x_dim == 1

        self.exclude_time = exclude_time
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.L_dim = L_dim
        if L_out_dim is None:
            L_out_dim = L_dim

        inp_dim = z_dim if exclude_time else z_dim + x_dim
        ode_layers = [nn.Linear(inp_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, L_out_dim)]
        # z = [L0, z_] so dim([L, z_, x]) = dim(z)+1
        self.latent_odefunc = nn.Sequential(*ode_layers)

        self.decode_layers = [nn.Linear(x_dim + z_dim, h_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(h_dim, h_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(h_dim, h_dim),
                              nn.ReLU(inplace=True)]
        self.xlz_to_hidden = nn.Sequential(*self.decode_layers)

        self.initial_t = initial_t
        self.nfe = 0

    def integrate_ode(self, t, v):  # v = (L(x), z_)
        self.nfe += 1
        z_ = v[:, self.L_dim:]
        batch_size = v.size()[0]
        vt = v
        if not self.exclude_time:
            time = t.view(1, 1).repeat(batch_size, 1)
            vt = torch.cat((vt, time), dim=1)

        dL = self.latent_odefunc(vt)
        dz_ = torch.zeros_like(z_)
        return torch.cat((dL, dz_), dim=1)

    def decode_latent(self, x, z, latent) -> torch.distributions.Distribution:
        raise NotImplementedError('The decoding of the latent ODE state is not implemented')

    def forward(self, x, z, y): # [santanu]: we don't need this y here anywhere, but for CDE
        """
        x : torch.Tensor
            Shape (batch_size, num_points, 1)
        z : torch.Tensor
            Shape (batch_size, z_dim)
        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (batch_size, num_points, y_dim).
        """
        self.nfe = 0
        batch_size, num_points, _ = x.size()

        # [santanu]: what is this doing exactly?
        # Append the initial time to the set of supplied times.
        x0 = self.initial_t.repeat(batch_size, 1, 1)
        x_sort = torch.cat((x0, x), dim=1)

        # ind specifies where each element in x ended up in times.
        times, ind = torch.unique(x_sort, sorted=True, return_inverse=True)
        # Remove the initial position index since we don't care about it.
        ind = ind[:, 1:, :]

        # Integrate forward from the batch of initial positions z.
        v = odeint(self.integrate_ode, z, times, method='dopri5')

        # Make shape (batch_size, unique_times, z_dim).
        permuted_v = v.permute(1, 0, 2)
        latent = permuted_v[:, :, :self.L_dim]

        # Extract the relevant (latent, time) pairs for each batch.
        tiled_ind = ind.repeat(1, 1, self.L_dim)
        latent = torch.gather(latent, dim=1, index=tiled_ind)
        return self.decode_latent(x, z, latent)


class MlpNormalODEDecoder(AbstractODEDecoder):
    """
    Maps target times x_target (which we call x for consistency with NPs)
    and samples z (encoding information about the context points)
    to predictions y_target. The decoder is an ODEsolve, using torchdiffeq.
    This version contains no control.
    Models inheriting from MlpNormalODEDecoder *must* either set self.xlz_to_hidden
    in constructor or override decode_latent(). Optionally, integrate_ode
    and forward can also be overridden.

    Parameters
    ----------
    x_dim : int
        Dimension of x values. Currently only works for dimension of 1.
    z_dim : int
        Dimension of latent variable z. Contains [L0, z_].

    h_dim : int
        Dimension of hidden layer in odefunc.
    y_dim : int
        Dimension of y values.

    L_dim : int
        Dimension of latent state L.
    """

    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim=None):
        super(MlpNormalODEDecoder, self).__init__(
            x_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim)

        self.x_dim = x_dim  # must be 1
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.L_dim = L_dim

        self.hidden_to_mu = nn.Linear(h_dim + L_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim + L_dim, y_dim)
        self.initial_t = initial_t
        self.nfe = 0

    def decode_latent(self, x, z, latent) -> torch.distributions.Distribution:
        batch_size, num_points, _ = x.size()
        # compute sigma using mlp (t, L(t), z_)
        z = z[:, self.L_dim:]
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        latent_flat = latent.view(batch_size * num_points, -1)
        z_flat = z.view(batch_size * num_points, self.z_dim - self.L_dim)
        # Input is concatenation of z with every row of x
        input_triplets = torch.cat((x_flat, latent_flat, z_flat), dim=1)
        hidden = self.xlz_to_hidden(input_triplets)
        hidden = torch.cat((latent_flat, hidden), dim=1)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)

        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)

        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)

        return Normal(mu, sigma)
    
    


class MlpPoissonODEDecoder(AbstractODEDecoder):
    """
    Maps target times x_target (which we call x for consistency with NPs)
    and samples z (encoding information about the context points)
    to predictions y_target. The decoder is an ODEsolve, using torchdiffeq.
    This version contains no control.
    Models inheriting from MlpPoissonODEDecoder *must* either set self.xlz_to_hidden
    in constructor or override decode_latent(). Optionally, integrate_ode
    and forward can also be overridden.

    Parameters
    ----------
    x_dim : int
        Dimension of x values. Currently only works for dimension of 1.
    z_dim : int
        Dimension of latent variable z. Contains [L0, z_].

    h_dim : int
        Dimension of hidden layer in odefunc.
    y_dim : int
        Dimension of y values.

    L_dim : int
        Dimension of latent state L.
    """

    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim=None):
        super(MlpPoissonODEDecoder, self).__init__(
            x_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim)

        self.x_dim = x_dim  # must be 1
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.L_dim = L_dim

        self.hidden_to_mu = nn.Linear(h_dim + L_dim, y_dim)
        self.hidden_to_alpha = nn.Linear(h_dim + L_dim, y_dim)
        self.initial_t = initial_t
        self.nfe = 0

    def decode_latent(self, x, z, latent) -> torch.distributions.Distribution:
        batch_size, num_points, _ = x.size()
        # compute sigma using mlp (t, L(t), z_)
        z = z[:, self.L_dim:]
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        latent_flat = latent.view(batch_size * num_points, -1)
        z_flat = z.view(batch_size * num_points, self.z_dim - self.L_dim)
        # Input is concatenation of z with every row of x
        input_triplets = torch.cat((x_flat, latent_flat, z_flat), dim=1)
        hidden = self.xlz_to_hidden(input_triplets)
        hidden = torch.cat((latent_flat, hidden), dim=1)

        # [santanu]: for Poisson
        # Reshape output into expected shape
        mu = self.hidden_to_mu(hidden)
        mu = F.softplus(mu)
        mu = mu.view(batch_size, num_points, self.y_dim)

        return Poisson(mu)
        
        # ## [santanu]: for Negative Binomial; to be checked
        # mu = self.hidden_to_mu(hidden)
        # alpha = self.hidden_to_alpha(hidden)
        
        # # Apply activation functions
        # mu = F.softplus(mu)  # Ensure positive mean
        # alpha = F.softplus(alpha) + 1  # Ensure alpha > 1 for numerical stability
        
        # # Reshape outputs
        # mu = mu.view(batch_size, num_points, self.y_dim)
        # alpha = alpha.view(batch_size, num_points, self.y_dim)
        
        # # Calculate p parameter for NegativeBinomial
        # p = alpha / (alpha + mu)

        # # Clip probabilities to avoid numerical issues
        # epsilon = 1e-6
        # p = torch.clamp(p, min=epsilon, max=1-epsilon)
        # return NegativeBinomial(total_count=alpha, probs=p)

        ## santanu: for generalized poisson
        # mu = self.hidden_to_mu(hidden)
        # # [santanu]: for Generalized Poisson
        # # Reshape output into expected shape
        # mu = F.softplus(mu)
        # mu = mu.view(batch_size, num_points, self.y_dim)
        
        # # Define the Generalized Poisson Distribution
        # class GeneralizedPoisson(torch.distributions.Distribution):
        #     def __init__(self, mu, theta):
        #         self.mu = mu
        #         self.theta = theta
        #         super().__init__(batch_shape=mu.size(), validate_args=False)
            
        #     def log_prob(self, value):
        #         # Implement the log probability function for Generalized Poisson
        #         return value * torch.log(self.mu / (1 + self.theta * self.mu)) - (self.mu / (1 + self.theta * self.mu)) - torch.lgamma(value + 1) - (value * torch.log(1 + self.theta * value))
            
        #     def sample(self, sample_shape=torch.Size()):
        #         # Implement the sampling function for Generalized Poisson
        #         raise NotImplementedError("Sampling is not implemented for Generalized Poisson")
        #     def mean(self):
        #         # Mean of the Generalized Poisson Distribution
        #         return self.mu / (1 - self.theta)
        
        # # Assuming theta is a parameter you have defined elsewhere
        # theta = 0.1  # Example value, you should replace this with your actual theta parameter
        # return GeneralizedPoisson(mu, theta)



        

######################################################################################


# Includes batching, now includes a latent state to go through MLP to get mu/sigma
class MlpSonodeDecoder(MlpNormalODEDecoder):
    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_t):
        super(MlpSonodeDecoder, self).__init__(x_dim, z_dim, h_dim, y_dim, L_dim, initial_t,
            L_out_dim=int(L_dim / 2))

    # x needs to be written as t here to use torchdiffeq
    # v, and therefore L and z_ must be vectors as a torch tensor, so they can concatentate
    def integrate_ode(self, t, v):  # v = (L(x), z_)
        self.nfe += 1
        Lv = v[:, int(self.L_dim / 2):self.L_dim]
        z_ = v[:, self.L_dim:]
        batch_size = v.size()[0]
        time = t.view(1, 1).repeat(batch_size, 1)
        vt = torch.cat((v, time), dim=1)
        dLx = Lv
        dLv = self.latent_odefunc(vt)
        dz_ = torch.zeros_like(z_)
        return torch.cat((dLx, dLv, dz_), dim=1)


class VanillaODEDecoder(MlpNormalODEDecoder):
    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_t):
        super(VanillaODEDecoder, self).__init__(
            x_dim, z_dim, h_dim, y_dim, L_dim, initial_t)

        self.latent_to_mu = nn.Linear(L_dim, y_dim)
        self.latent_to_sigma = nn.Linear(L_dim, y_dim)

    def decode_latent(self, x, z, latent):
        batch_size, num_points, _ = x.size()
        latent_flat = latent.view(batch_size * num_points, -1)
        mu = self.latent_to_mu(latent_flat)
        pre_sigma = self.latent_to_sigma(latent_flat)
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return Normal(mu, sigma)


class VanillaSONODEDecoder(MlpNormalODEDecoder):
    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_t):
        super(VanillaSONODEDecoder, self).__init__(
            x_dim, z_dim, h_dim, y_dim, L_dim,
            initial_t, L_out_dim=int(L_dim / 2))

        self.latent_to_mu = nn.Linear(L_dim, y_dim)
        self.latent_to_sigma = nn.Linear(L_dim, y_dim)

    # x needs to be written as t here to use torchdiffeq
    # v, and therefore L and z_ must be vectors as a torch tensor, so they can concatentate
    def integrate_ode(self, t, v):  # v = (L(x), z_)
        self.nfe += 1
        Lv = v[:, int(self.L_dim / 2):self.L_dim]
        z_ = v[:, self.L_dim:]
        batch_size = v.size()[0]
        time = t.view(1, 1).repeat(batch_size, 1)
        vt = torch.cat((v, time), dim=1)
        dLx = Lv
        dLv = self.latent_odefunc(vt)
        dz_ = torch.zeros_like(z_)
        return torch.cat((dLx, dLv, dz_), dim=1)

    def decode_latent(self, x, z, latent):
        batch_size, num_points, _ = x.size()
        latent_flat = latent.view(batch_size * num_points, -1)
        mu = self.latent_to_mu(latent_flat)
        pre_sigma = self.latent_to_sigma(latent_flat)
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return Normal(mu, sigma)


###################################################### Controlled NODE decoders ############################################
    

class AbstractCDEDecoder(nn.Module):
    """
    An Abstract Decoder using a Neural CDE. Child classes must implement decode_latent.

    Parameters
    ----------
    x_dim : int
        Dimension of x values. Currently only works for dimension of 1.
    z_dim : int
        Dimension of latent variable z. Contains [L0, z_].
    h_dim : int
        Dimension of hidden layer in cdefunc.
    y_dim : int
        Dimension of y values.
    L_dim : int
        Dimension of latent state L.
    """

    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim=None, exclude_time=False):
        super(AbstractCDEDecoder, self).__init__()
        # The input is always time.
        assert x_dim == 1

        self.exclude_time = exclude_time
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.L_dim = L_dim
        self.use_mean= True
        if L_out_dim is None:
            L_out_dim = L_dim

        inp_dim = z_dim if exclude_time else z_dim + x_dim

        """
        in the previous ode function, we only integrate over l0 and not over D
        so D goes in but the slope is calculated only wrt L and not D
        """
        self.latent_cdefunc_new = nn.Sequential(
            nn.Linear(inp_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, z_dim*(self.y_dim+self.x_dim)) # including x_dim because we add time in cubic spline data
        )

        self.latent_cdefunc_like_previous = nn.Sequential(
            nn.Linear(inp_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, L_out_dim*(self.y_dim+self.x_dim)) # including x_dim because we add time in cubic spline data
        )



        self.decode_layers = nn.Sequential(
            nn.Linear(x_dim + z_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True)
        )
        self.xlz_to_hidden = self.decode_layers

        self.initial_t = initial_t
        self.nfe = 0
    
    def integrate_cde_new(self, t, v):
        self.nfe += 1 # this is just for the sake of counter
        z_ = v[:, self.L_dim:]
        batch_size = v.size()[0]
        vt = v
        if not self.exclude_time:
            time = t.view(1, 1).repeat(batch_size, 1)
            vt = torch.cat((vt, time), dim=1)

        dL = self.latent_cdefunc_new(vt).view(batch_size, self.z_dim, self.y_dim+self.x_dim)
        return dL

    def integrate_cde_like_previous(self, t, v):
        self.nfe += 1 # this is just for the sake of counter
        z_ = v[:, self.L_dim:]
        batch_size = v.size()[0]
        vt = v
        if not self.exclude_time:
            time = t.view(1, 1).repeat(batch_size, 1)
            vt = torch.cat((vt, time), dim=1)

        # our out should be (batch_size, z_dim, y_dim)
        dL = self.latent_cdefunc_like_previous(vt).view(batch_size, self.L_dim, self.y_dim+self.x_dim) # (batch_size, L_dim*y_dim)
        dz_ = torch.zeros(batch_size, (self.z_dim-self.L_dim), self.y_dim+self.x_dim) # D_dim= z_dim-L_dim
        out= torch.cat((dL, dz_), dim=1)
        # import pdb; pdb.set_trace()
        return out
    
    def decode_latent(self, x, z, latent):
        raise NotImplementedError('The decoding of the latent CDE state is not implemented')

    def forward(self, x, z, y):
        self.nfe = 0
        batch_size, num_points, _ = y.size()
        
        if self.use_mean:
            y_mean = y.mean(dim=0, keepdim=True)  # Shape [1, T, F]
            # Repeat the mean tensor to match the batch size
            y_mean_repeated = y_mean.repeat(batch_size, 1, 1)  # Shape [b, T, F]
            data = torch.cat((x, y_mean_repeated), dim=-1)
        else:
            data = torch.cat((x, y), dim=-1)

        # Create the coefficients for the CDE
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(data)
        X = torchcde.CubicSpline(coeffs)

        # [santanu]: this needs to change
        # times = X.interval
        times= x[0].view(-1) # just use time from anyone because I'm assuming ordered and ascending

        # Initial hidden state
        z0 = z # [santanu]: we need to try two cases here, repeat average and as is

        # import pdb; pdb.set_trace()
        # Integrate using the CDE solver
        # the times here needs to change
        zT = torchcde.cdeint(X=X, func=self.integrate_cde_like_previous, z0=z0, t=times, method='rk4')

        # import pdb; pdb.set_trace()
        return self.decode_latent(x, z, zT[:,:,:self.L_dim]) # only the Latent goes
    

class MlpNormalCDEDecoder(AbstractCDEDecoder):
    """
    Maps target times x_target (which we call x for consistency with NPs)
    and samples z (encoding information about the context points)
    to predictions y_target. The decoder is an CDEsolve, using torchcde.
    This version contains no control.
    Models inheriting from MlpNormalCDEDecoder *must* either set self.xlz_to_hidden
    in constructor or override decode_latent(). Optionally, cde_func
    and forward can also be overridden.

    Parameters
    ----------
    x_dim : int
        Dimension of x values. Currently only works for dimension of 1.
    z_dim : int
        Dimension of latent variable z. Contains [L0, z_].
    h_dim : int
        Dimension of hidden layer in cdefunc.
    y_dim : int
        Dimension of y values.
    L_dim : int
        Dimension of latent state L.
    """

    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim=None):
        super(MlpNormalCDEDecoder, self).__init__(
            x_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim)

        self.x_dim = x_dim  # must be 1
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.L_dim = L_dim

        self.hidden_to_mu = nn.Linear(h_dim + L_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim + L_dim, y_dim)
        self.initial_t = initial_t
        self.nfe = 0

    def decode_latent(self, x, z, latent) -> torch.distributions.Distribution:
        batch_size, num_points, _ = x.size()
        z = z[:, self.L_dim:]
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        x_flat = x.view(batch_size * num_points, self.x_dim)
        # latent_flat = latent.view(batch_size * num_points, -1)
        latent_flat = latent.reshape(batch_size * num_points, -1)
        z_flat = z.view(batch_size * num_points, self.z_dim - self.L_dim)

        input_triplets = torch.cat((x_flat, latent_flat, z_flat), dim=1)
        hidden = self.xlz_to_hidden(input_triplets)
        hidden = torch.cat((latent_flat, hidden), dim=1)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)

        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)

        sigma = 0.1 + 0.9 * torch.nn.functional.softplus(pre_sigma)

        return torch.distributions.Normal(mu, sigma)
    

class MlpPoissonCDEDecoder(AbstractCDEDecoder):
    """
    Maps target times x_target (which we call x for consistency with NPs)
    and samples z (encoding information about the context points)
    to predictions y_target. The decoder is an ODEsolve, using torchdiffeq.
    This version contains no control.
    Models inheriting from MlpPoissonODEDecoder *must* either set self.xlz_to_hidden
    in constructor or override decode_latent(). Optionally, integrate_ode
    and forward can also be overridden.

    Parameters
    ----------
    x_dim : int
        Dimension of x values. Currently only works for dimension of 1.
    z_dim : int
        Dimension of latent variable z. Contains [L0, z_].

    h_dim : int
        Dimension of hidden layer in odefunc.
    y_dim : int
        Dimension of y values.

    L_dim : int
        Dimension of latent state L.
    """

    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim=None):
        super(MlpPoissonCDEDecoder, self).__init__(
            x_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim)

        self.x_dim = x_dim  # must be 1
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.L_dim = L_dim

        self.hidden_to_mu = nn.Linear(h_dim + L_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim + L_dim, y_dim)
        self.initial_t = initial_t
        self.nfe = 0

    def decode_latent(self, x, z, latent) -> torch.distributions.Distribution:
        batch_size, num_points, _ = x.size()
        # compute sigma using mlp (t, L(t), z_)
        z = z[:, self.L_dim:]
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        # latent_flat = latent.view(batch_size * num_points, -1)
        latent_flat = latent.reshape(batch_size * num_points, -1)
        z_flat = z.view(batch_size * num_points, self.z_dim - self.L_dim)
        # Input is concatenation of z with every row of x
        input_triplets = torch.cat((x_flat, latent_flat, z_flat), dim=1)
        hidden = self.xlz_to_hidden(input_triplets)
        hidden = torch.cat((latent_flat, hidden), dim=1)
        mu = self.hidden_to_mu(hidden)
        mu = F.softplus(mu)
        mu = mu.view(batch_size, num_points, self.y_dim)

        return Poisson(mu)
    







##### more sofisticated ode-encoders
###################################################################################################################################################################################################################################
class GRU_unit(nn.Module):
	def __init__(self, latent_dim, input_dim, 
		update_gate = None,
		reset_gate = None,
		new_state_net = None,
		n_units = 100,
		device = torch.device("cpu")):
		super(GRU_unit, self).__init__()

		if update_gate is None:
			self.update_gate = nn.Sequential(
			   nn.Linear(latent_dim + input_dim, n_units),
			   nn.Tanh(),
			   nn.Linear(n_units, latent_dim),
			   nn.Sigmoid())
			init_network_weights(self.update_gate)
		else: 
			self.update_gate  = update_gate

		if reset_gate is None:
			self.reset_gate = nn.Sequential(
			   nn.Linear(latent_dim + input_dim, n_units),
			   nn.Tanh(),
			   nn.Linear(n_units, latent_dim),
			   nn.Sigmoid())
			init_network_weights(self.reset_gate)
		else: 
			self.reset_gate  = reset_gate

		if new_state_net is None:
			self.new_state_net = nn.Sequential(
			   nn.Linear(latent_dim + input_dim, n_units),
			   nn.Tanh(),
			   nn.Linear(n_units, latent_dim))
			init_network_weights(self.new_state_net)
		else: 
			self.new_state_net  = new_state_net


	def forward(self, y_mean, x, masked_update = False):
		y_concat = torch.cat([y_mean, x], -1)

		update_gate = self.update_gate(y_concat)
		reset_gate = self.reset_gate(y_concat)
		concat = torch.cat([y_mean * reset_gate, x], -1)
		
		new_state =self.new_state_net(concat)

		new_y = (1-update_gate) * new_state + update_gate * y_mean

		assert(not torch.isnan(new_y).any())

        # santanu: worry about this later
		if masked_update:
			# IMPORTANT: assumes that x contains both data and mask
			# update only the hidden states for hidden state only if at least one feature is present for the current time point
			n_data_dims = x.size(-1)//2
			mask = x[:, :, n_data_dims:]
			check_mask(x[:, :, :n_data_dims], mask)
			
			mask = (torch.sum(mask, -1, keepdim = True) > 0).float()

			assert(not torch.isnan(mask).any())

			new_y = mask * new_y + (1-mask) * y_mean

			if torch.isnan(new_y).any():
				print("new_y is nan!")
				print(mask)
				print(y_mean)
				exit()

		return new_y
     
class simple_GRU_unit(nn.Module):
    def __init__(self, latent_dim, input_dim, n_units=100):
        super(simple_GRU_unit, self).__init__()
        
        self.input_gate = nn.Linear(latent_dim + input_dim, latent_dim)
        self.forget_gate = nn.Linear(latent_dim + input_dim, latent_dim)
        self.output_gate = nn.Linear(latent_dim + input_dim, latent_dim)
        self.cell_state = nn.Linear(latent_dim + input_dim, latent_dim)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, h_prev, x):
        combined = torch.cat([h_prev, x], dim=-1)
        
        i = self.sigmoid(self.input_gate(combined))
        f = self.sigmoid(self.forget_gate(combined))
        o = self.sigmoid(self.output_gate(combined))
        c_tilde = self.tanh(self.cell_state(combined))
        
        c = f * h_prev + i * c_tilde
        h = o * self.tanh(c)
        
        return h


class DiffeqSolver(nn.Module):
	def __init__(self, input_dim, ode_func, method, latents, 
			odeint_rtol = 1e-4, odeint_atol = 1e-5, device = torch.device("cpu")):
		super(DiffeqSolver, self).__init__()

		self.ode_method = method
		self.latents = latents		
		self.device = device
		self.ode_func = ode_func

		self.odeint_rtol = odeint_rtol
		self.odeint_atol = odeint_atol

	def forward(self, first_point, time_steps_to_predict, backwards = False):
		"""
		# Decode the trajectory through ODE Solver
		"""
		n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
		n_dims = first_point.size()[-1]

		pred_y = odeint(self.ode_func, first_point, time_steps_to_predict, 
			rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
		pred_y = pred_y.permute(1,2,0,3)

		assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
		assert(pred_y.size()[0] == n_traj_samples)
		assert(pred_y.size()[1] == n_traj)

		return pred_y

	def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict, 
		n_traj_samples = 1):
		"""
		# Decode the trajectory through ODE Solver using samples from the prior

		time_steps_to_predict: time steps at which we want to sample the new trajectory
		"""
		func = self.ode_func.sample_next_point_from_prior

		pred_y = odeint(func, starting_point_enc, time_steps_to_predict, 
			rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
		# shape: [n_traj_samples, n_traj, n_tp, n_dim]
		pred_y = pred_y.permute(1,2,0,3)
		return pred_y

class ODEFunc(nn.Module):
	def __init__(self, input_dim, latent_dim, ode_func_net, device = torch.device("cpu")):
		"""
		input_dim: dimensionality of the input
		latent_dim: dimensionality used for ODE. Analog of a continous latent state
		"""
		super(ODEFunc, self).__init__()

		self.input_dim = input_dim
		self.device = device

		init_network_weights(ode_func_net)
		self.gradient_net = ode_func_net

	def forward(self, t_local, y, backwards = False):
		"""
		Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

		t_local: current time point
		y: value at the current time point
		"""
		grad = self.get_ode_gradient_nn(t_local, y)
		if backwards:
			grad = -grad
		return grad

	def get_ode_gradient_nn(self, t_local, y):
		return self.gradient_net(y)

	def sample_next_point_from_prior(self, t_local, y):
		"""
		t_local: current time point
		y: value at the current time point
		"""
		return self.get_ode_gradient_nn(t_local, y)


class Encoder_ODE_RNN(nn.Module):
    # Derive z0 by running ode backwards.
    # For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
    # Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
    # Continue until we get to z0
    def __init__(self, x_dim, y_dim, h_dim, r_dim, batch_size, 
        z0_dim = None, GRU_update = None, 
        n_gru_units = 100, 
        device = torch.device("cpu")):

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim
        self.batch_size = batch_size
        
        super(Encoder_ODE_RNN, self).__init__()

        if z0_dim is None:
            self.z0_dim = self.h_dim
        else:
            self.z0_dim = z0_dim

        if GRU_update is None:
            self.GRU_update = GRU_unit(self.h_dim, self.y_dim, 
                n_units = n_gru_units, 
                device=device).to(device)
        else:
            self.GRU_update = GRU_update
            # self.GRU_update = simple_GRU_unit(self.h_dim, self.y_dim, n_units = n_gru_units)
        
        ode_func_net = create_net(self.h_dim, self.h_dim, 
			n_layers = 1, n_units = 100, nonlinear = nn.Tanh)

        rec_ode_func = ODEFunc(input_dim = self.y_dim, latent_dim = self.h_dim, ode_func_net = ode_func_net, device = device).to(device)
        z0_diffeq_solver = DiffeqSolver(self.y_dim, rec_ode_func, "euler", self.h_dim, odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = self.h_dim
        self.input_dim = self.y_dim
        self.device = device
        self.extra_info = None

        self.transform_z0 = nn.Sequential(
            nn.Linear(self.latent_dim * 2, 100),
            nn.Tanh(),
            nn.Linear(100, self.z0_dim * 2),)
        init_network_weights(self.transform_z0)

        self.fc = nn.Linear(h_dim, r_dim)


    def forward(self, time_steps, data, run_backwards = True, save_info = False):
        # data, time_steps -- observations and their time stamps
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 
        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())

        n_traj, n_tp, n_dims = data.size()
        if len(time_steps) == 1:
            prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
            prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

            xi = data[:,0,:].unsqueeze(0)

            last_yi, _ = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            
            last_yi, _, extra_info = self.run_odernn(
                data, time_steps, run_backwards = run_backwards,
                save_info = save_info)

        mean_z0 = last_yi.reshape(n_traj, self.latent_dim)

        # import pdb; pdb.set_trace()
        # [santanu]: what is this thing doing exactly?
        # I guess we'll need this during masking
        # mean_z0, std_z0 = split_last_dim(self.transform_z0(means_z0))
        # std_z0 = std_z0.abs()
        # if save_info:
        #     self.extra_info = extra_info
        output = self.fc(mean_z0)
        return output


    def run_odernn(self, data, time_steps, 
        run_backwards = True, save_info = False):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 
        n_traj, n_tp, n_dims = data.size()
        time_steps = time_steps[0, :].squeeze(-1) # because it's of shape batch_size, n_time_steps, 1
        extra_info = []
        # import pdb; pdb.set_trace()
        t0 = time_steps[-1]
        if run_backwards:
            t0 = time_steps[0]

        device = get_device(data)

        prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(device)

        prev_t, t_i = time_steps[-1] + 0.01,  time_steps[-1]

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        #print("minimum step: {}".format(minimum_step))

        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())

        latent_ys = []
        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, len(time_steps))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)
        # import pdb; pdb.set_trace()
        for i in time_points_iter:
            if (prev_t - t_i) < minimum_step:
                time_points = torch.stack((prev_t, t_i))
                inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y) * (t_i - prev_t)

                if torch.isnan(inc).any():
                    import pdb; pdb.set_trace()

                assert(not torch.isnan(inc).any())

                ode_sol = prev_y + inc
                ode_sol = torch.stack((prev_y, ode_sol), 2).to(device)

                assert(not torch.isnan(ode_sol).any())
            else:
                n_intermediate_tp = max(2, ((prev_t - t_i) / minimum_step).int())

                time_points = linspace_vector(prev_t, t_i, n_intermediate_tp)
                ode_sol = self.z0_diffeq_solver(prev_y, time_points)

                assert(not torch.isnan(ode_sol).any())

            if torch.mean(ode_sol[:, :, 0, :]  - prev_y) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, :, 0, :]  - prev_y))
                exit()
            #assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)

            yi_ode = ode_sol[:, :, -1, :] # 1,50,2,30; yi_ode: 1,50,30
            xi = data[:,i,:].unsqueeze(0)# 1,50,28
            
            yi = self.GRU_update(yi_ode, xi) # why tf do you need gru_update? # yi: 1,50,30
            prev_y = yi           
            prev_t, t_i = time_steps[i],  time_steps[i-1]

            latent_ys.append(yi)

            if save_info:
                d = {"yi_ode": yi_ode.detach(), #"yi_from_data": yi_from_data,
                     "yi": yi.detach(),
                     "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
                extra_info.append(d)
            

        latent_ys = torch.stack(latent_ys, 1)# 1, 100, 50, 30
        # import pdb; pdb.set_trace()
        assert(not torch.isnan(yi).any())

        return yi, latent_ys, extra_info
    #########################################################################################################################################################################################################################################################################