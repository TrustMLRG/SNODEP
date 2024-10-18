import torch

from torch.nn import Module as TorchModule
from models.models import *
from torch.distributions import LogNormal


class TimeNeuralProcess(nn.Module):
    """
    Implements Neural Process for functions of arbitrary dimensions defined over time
    Code adapted from https://github.com/EmilienDupont/neural-processes

    Parameters
    ----------
    y_dim : int
        Dimension of y values.
    r_dim : int
        Dimension of output representation r.
    z_dim : int
        Dimension of latent variable z
    h_dim : int
        Dimension of hidden layer in encoder and decoder.
    encoder: TorchModule
        Encoder to encode the context.
    decoder: TorchModule
        Decoder to decode the latent variable Z.
    """

    def __init__(self, y_dim, r_dim, z_dim, h_dim, encoder: TorchModule, decoder: TorchModule):
        super(TimeNeuralProcess, self).__init__()
        self.x_dim = 1
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        # Initialize networks
        self.xy_to_r = encoder
        self.r_to_mu_sigma = MuSigmaEncoder(r_dim, z_dim)
        self.xz_to_y = decoder

    def aggregate(self, r_i):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.

        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batch_size, num_points, r_dim)
        """
        return torch.mean(r_i, dim=1)

    def xy_to_mu_sigma(self, x, y, y0):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        y0 : torch.Tensor
        """
        batch_size, num_points, _ = x.size()
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        # Encode each point into a representation r_i
        r_i_flat = self.xy_to_r(x_flat, y_flat)
        # Reshape tensors into batches
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)
        # import pdb; pdb.set_trace()
        # Return parameters of distribution
        return self.r_to_mu_sigma(r)

    def forward(self, x_context, y_context, x_target, y_target=None, y0=None):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.

        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_target.
        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)
        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)
        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training. # [santanu]
        y0 : torch.Tensor or None
            Shape (batch_size, y_dim). The initial state.
        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        # Infer quantities from tensor dimensions
        _, num_target, _ = x_target.size()
        _, _, y_dim = y_context.size()

        if self.training:
            # Encode target and context (context needs to be encoded to
            # calculate kl term)
            mu_target, sigma_target = self.xy_to_mu_sigma(x_target, y_target, y0)
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context, y0)
            # Sample from encoded distribution using reparameterization trick
            q_target = Normal(mu_target, sigma_target)
            q_context = Normal(mu_context, sigma_context)

            # [santanu]: trying lognormal instead poisson because reparametrization trick doesn't work otherwise
            # q_target = LogNormal(mu_target, sigma_target)
            # q_context = LogNormal(mu_context, sigma_context)

            z_sample = q_target.rsample()
            # Get parameters of output distribution
            # p_y_pred = self.xz_to_y(x_target, z_sample)
            p_y_pred = self.xz_to_y(x_target, z_sample, y_target)
            return p_y_pred, q_target, q_context
        else:
            # At testing time, encode only context
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context, y0)
            # Sample from distribution based on context
            q_context = Normal(mu_context, sigma_context)
            # [santanu]: trying lognormal instead poisson because reparametrization trick doesn't work otherwise
            # q_context = LogNormal(mu_context, sigma_context)
            z_sample = q_context.rsample()
            # Predict target points based on context
            # import pdb; pdb.set_trace()
            # p_y_pred = self.xz_to_y(x_target, z_sample)
            p_y_pred = self.xz_to_y(x_target, z_sample, y_target)
            return p_y_pred
        


###################################################################
class TimeNeuralProcessPoisson(nn.Module):
    """
    Implements Neural Process for functions of arbitrary dimensions defined over time
    Code adapted from https://github.com/EmilienDupont/neural-processes

    Parameters
    ----------
    y_dim : int
        Dimension of y values.
    r_dim : int
        Dimension of output representation r.
    z_dim : int
        Dimension of latent variable z.
    h_dim : int
        Dimension of hidden layer in encoder and decoder.
    encoder: TorchModule
        Encoder to encode the context.
    decoder: TorchModule
        Decoder to decode the latent variable Z.
    """

    def __init__(self, y_dim, r_dim, z_dim, h_dim, encoder: TorchModule, decoder: TorchModule):
        super(TimeNeuralProcessPoisson, self).__init__()
        self.x_dim = 1
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        # Initialize networks
        self.xy_to_r = encoder
        self.r_to_mu_sigma = MuSigmaEncoder(r_dim, z_dim)
        self.xz_to_y = decoder

    def aggregate(self, r_i):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.

        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batch_size, num_points, r_dim)
        """
        return torch.mean(r_i, dim=1)

    def xy_to_mu_sigma(self, x, y, y0):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        y0 : torch.Tensor
        """
        batch_size, num_points, _ = x.size()
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        # Encode each point into a representation r_i
        r_i_flat = self.xy_to_r(x_flat, y_flat)
        # Reshape tensors into batches
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)
        # Return parameters of distribution
        return self.r_to_mu_sigma(r)

    def forward(self, x_context, y_context, x_target, y_target=None, y0=None):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.

        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_target.
        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)
        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)
        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.
        y0 : torch.Tensor or None
            Shape (batch_size, y_dim). The initial state.
        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        # Infer quantities from tensor dimensions
        _, num_target, _ = x_target.size()
        _, _, y_dim = y_context.size()

        if self.training:
            # Encode target and context (context needs to be encoded to
            # calculate kl term)
            mu_target, sigma_target = self.xy_to_mu_sigma(x_target, y_target, y0)
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context, y0)
            # Sample from encoded distribution using reparameterization trick
            # q_target = Normal(mu_target, sigma_target)
            # q_context = Normal(mu_context, sigma_context)
            
            # [santanu]: trying lognormal instead poisson because reparametrization trick doesn't work otherwise
            q_target = LogNormal(mu_target, sigma_target)
            q_context = LogNormal(mu_context, sigma_context)

            z_sample = q_target.rsample()
            # Get parameters of output distribution
            p_y_pred = self.xz_to_y(x_target, z_sample, y_target)
            return p_y_pred, q_target, q_context
        else:
            # At testing time, encode only context
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context, y0)
            #[santanu] IMPORTANT: sampling from Normal here gives a drop in mse loss
            # Sample from distribution based on context
            # q_context = Normal(mu_context, sigma_context)
            # [santanu]: trying lognormal instead poisson because reparametrization trick doesn't work otherwise
            q_context = LogNormal(mu_context, sigma_context)
            z_sample = q_context.rsample()
            # Predict target points based on context
            p_y_pred = self.xz_to_y(x_target, z_sample, y_target)
            return p_y_pred
###################################################################


class NeuralODEProcess(TimeNeuralProcess):
    """Uses a random ODE initial state L(0) and control signal D to encode the context.
       Assumes that the decoder is ODE-based.
    """

    def __init__(self, y_dim, r_dim, h_dim, L_dim, D_dim, encoder: TorchModule,
                 decoder: TorchModule):
        z_dim = L_dim + D_dim
        super(NeuralODEProcess, self).__init__(y_dim, r_dim, z_dim, h_dim, encoder, decoder)

        # Remove parent network that is unused in the NDP
        self.r_to_mu_sigma = None
        # Initialise separate mu-sigma encoders for L and D
        self.L_r_to_mu_sigma = MuSigmaEncoder(r_dim, L_dim)
        self.D_r_to_mu_sigma = MuSigmaEncoder(r_dim, D_dim)

    def xy_to_mu_sigma(self, x, y, y0):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, 1)
        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        y0 : torch.Tensor
            Shape (batch_size, y_dim) The batch of initial positions.
        """
        batch_size, num_points, _ = x.size()
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        # Encode each point into a representation r_i for the initial position and dynamics
        # import pdb; pdb.set_trace()
        L_r_i_flat, D_r_i_flat = self.xy_to_r(x_flat, y_flat, y0)
        # import pdb; pdb.set_trace()
        # Reshape tensors into batches
        if L_r_i_flat.size(0) == batch_size:
            # In this case, we have used only y0 to compute L_r_i
            L_r_i = L_r_i_flat.view(batch_size, 1, self.r_dim)
        else:
            # In this case, we have used the whole context to compute L_r_i
            L_r_i = L_r_i_flat.view(batch_size, num_points, self.r_dim)
        D_r_i = D_r_i_flat.view(batch_size, num_points, self.r_dim)
        # Aggregate representations r_i into a single representation r
        L_r = self.aggregate(L_r_i)
        D_r = self.aggregate(D_r_i)
        # Return parameters of distribution
        L_mu, L_sigma = self.L_r_to_mu_sigma(L_r)
        D_mu, D_sigma = self.D_r_to_mu_sigma(D_r)
        # Combine L and D in a single random vector
        mu = torch.cat([L_mu, D_mu], dim=1)
        sigma = torch.cat([L_sigma, D_sigma], dim=1)
        # import pdb; pdb.set_trace()
        return mu, sigma
    

###################################################
# basically just a copy of the above
class NeuralODEProcessPoisson(TimeNeuralProcessPoisson):
    """Uses a random ODE initial state L(0) and control signal D to encode the context.
       Assumes that the decoder is ODE-based.
    """

    def __init__(self, y_dim, r_dim, h_dim, L_dim, D_dim, encoder: TorchModule,
                 decoder: TorchModule):
        z_dim = L_dim + D_dim
        super(NeuralODEProcessPoisson, self).__init__(y_dim, r_dim, z_dim, h_dim, encoder, decoder)

        # Remove parent network that is unused in the NDP
        self.r_to_mu_sigma = None
        # Initialise separate mu-sigma encoders for L and D
        self.L_r_to_mu_sigma = MuSigmaEncoder(r_dim, L_dim)
        self.D_r_to_mu_sigma = MuSigmaEncoder(r_dim, D_dim)

    def xy_to_mu_sigma(self, x, y, y0):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, 1)
        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        y0 : torch.Tensor
            Shape (batch_size, y_dim) The batch of initial positions.
        """
        batch_size, num_points, _ = x.size()
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        # Encode each point into a representation r_i for the initial position and dynamics
        L_r_i_flat, D_r_i_flat = self.xy_to_r(x_flat, y_flat, y0)
        # Reshape tensors into batches
        if L_r_i_flat.size(0) == batch_size:
            # In this case, we have used only y0 to compute L_r_i
            L_r_i = L_r_i_flat.view(batch_size, 1, self.r_dim)
        else:
            # In this case, we have used the whole context to compute L_r_i
            L_r_i = L_r_i_flat.view(batch_size, num_points, self.r_dim)
        D_r_i = D_r_i_flat.view(batch_size, num_points, self.r_dim)
        # Aggregate representations r_i into a single representation r
        L_r = self.aggregate(L_r_i)
        D_r = self.aggregate(D_r_i)
        # Return parameters of distribution
        L_mu, L_sigma = self.L_r_to_mu_sigma(L_r)
        D_mu, D_sigma = self.D_r_to_mu_sigma(D_r)
        # Combine L and D in a single random vector
        mu = torch.cat([L_mu, D_mu], dim=1)
        sigma = torch.cat([L_sigma, D_sigma], dim=1)
        return mu, sigma
####################################################################

class NeuralODEProcessLSTM(TimeNeuralProcess):
    """Uses a random ODE initial state L(0) and control signal D to encode the context.
       Assumes that the decoder is ODE-based.
    """

    def __init__(self, y_dim, r_dim, h_dim, L_dim, D_dim, encoder: TorchModule,
                 decoder: TorchModule):
        z_dim = L_dim + D_dim
        super(NeuralODEProcessLSTM, self).__init__(y_dim, r_dim, z_dim, h_dim, encoder, decoder)

        # Remove parent network that is unused in the NDP
        self.r_to_mu_sigma = None
        # Initialise separate mu-sigma encoders for L and D
        self.L_r_to_mu_sigma = MuSigmaEncoder(r_dim, L_dim)
        self.D_r_to_mu_sigma = MuSigmaEncoder(r_dim, D_dim)

    def xy_to_mu_sigma(self, x, y, y0):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, 1)
        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        y0 : torch.Tensor
            Shape (batch_size, y_dim) The batch of initial positions.
        """
        batch_size, num_points, _ = x.size()
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat= x
        y_flat= y
        # x_flat = x.view(batch_size * num_points, self.x_dim)
        # y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        # Encode each point into a representation r_i for the initial position and dynamics
        # L_r_i_flat, D_r_i_flat = self.xy_to_r(x_flat, y_flat, y0)
        # # Reshape tensors into batches
        # if L_r_i_flat.size(0) == batch_size:
        #     # In this case, we have used only y0 to compute L_r_i
        #     L_r_i = L_r_i_flat.view(batch_size, 1, self.r_dim)
        # else:
        #     # In this case, we have used the whole context to compute L_r_i
        #     L_r_i = L_r_i_flat.view(batch_size, num_points, self.r_dim)
        # D_r_i = D_r_i_flat.view(batch_size, num_points, self.r_dim)
        # # Aggregate representations r_i into a single representation r
        # L_r = self.aggregate(L_r_i)
        # D_r = self.aggregate(D_r_i)
        # Return parameters of distribution

        L_r, D_r = self.xy_to_r(x_flat, y_flat, y0)
        L_mu, L_sigma = self.L_r_to_mu_sigma(L_r)
        D_mu, D_sigma = self.D_r_to_mu_sigma(D_r)
        # Combine L and D in a single random vector
        mu = torch.cat([L_mu, D_mu], dim=1)
        sigma = torch.cat([L_sigma, D_sigma], dim=1)
        return mu, sigma
    

###################################################
# basically just a copy of the above
class NeuralODEProcessPoissonLSTM(TimeNeuralProcessPoisson):
    """Uses a random ODE initial state L(0) and control signal D to encode the context.
       Assumes that the decoder is ODE-based.
    """

    def __init__(self, y_dim, r_dim, h_dim, L_dim, D_dim, encoder: TorchModule,
                 decoder: TorchModule):
        z_dim = L_dim + D_dim
        super(NeuralODEProcessPoissonLSTM, self).__init__(y_dim, r_dim, z_dim, h_dim, encoder, decoder)

        # Remove parent network that is unused in the NDP
        self.r_to_mu_sigma = None
        # Initialise separate mu-sigma encoders for L and D
        self.L_r_to_mu_sigma = MuSigmaEncoder(r_dim, L_dim)
        self.D_r_to_mu_sigma = MuSigmaEncoder(r_dim, D_dim)

    def xy_to_mu_sigma(self, x, y, y0):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, 1)
        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        y0 : torch.Tensor
            Shape (batch_size, y_dim) The batch of initial positions.
        """
        batch_size, num_points, _ = x.size()
        x_flat= x
        y_flat= y
        # # Flatten tensors, as encoder expects one dimensional inputs
        # x_flat = x.view(batch_size * num_points, self.x_dim)
        # y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        # # Encode each point into a representation r_i for the initial position and dynamics
        # L_r_i_flat, D_r_i_flat = self.xy_to_r(x_flat, y_flat, y0)
        # # Reshape tensors into batches
        # if L_r_i_flat.size(0) == batch_size:
        #     # In this case, we have used only y0 to compute L_r_i
        #     L_r_i = L_r_i_flat.view(batch_size, 1, self.r_dim)
        # else:
        #     # In this case, we have used the whole context to compute L_r_i
        #     L_r_i = L_r_i_flat.view(batch_size, num_points, self.r_dim)
        # D_r_i = D_r_i_flat.view(batch_size, num_points, self.r_dim)
        # # Aggregate representations r_i into a single representation r
        # L_r = self.aggregate(L_r_i)
        # D_r = self.aggregate(D_r_i)
        # Return parameters of distribution
        import pdb; pdb.set_trace()
        L_r, D_r = self.xy_to_r(x_flat, y_flat, y0)
        L_mu, L_sigma = self.L_r_to_mu_sigma(L_r)
        D_mu, D_sigma = self.D_r_to_mu_sigma(D_r)
        # Combine L and D in a single random vector
        mu = torch.cat([L_mu, D_mu], dim=1)
        sigma = torch.cat([L_sigma, D_sigma], dim=1)
        return mu, sigma
#################################################################


class MlpNeuralODEProcess(NeuralODEProcess):
    def __init__(self, y_dim, r_dim, z_dim, h_dim, L_dim, initial_t, irregular_freq=1):
        D_dim = z_dim - L_dim
        ctx_encoder = Encoder(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim)
        encoder = SingleContextNDPEncoder(context_encoder=ctx_encoder)
        if irregular_freq == 1:
            decoder = MlpNormalODEDecoder(x_dim=1, y_dim=y_dim, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        else:
            decoder = MlpNormalODEDecoder(x_dim=1, y_dim=y_dim//2, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        super(MlpNeuralODEProcess, self).__init__(y_dim, r_dim, h_dim, L_dim, D_dim,
            encoder, decoder)
        
class MlpNeuralODEProcessPoisson(NeuralODEProcessPoisson):
    def __init__(self, y_dim, r_dim, z_dim, h_dim, L_dim, initial_t, irregular_freq=1):
        D_dim = z_dim - L_dim
        ctx_encoder = Encoder(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim)
        encoder = SingleContextNDPEncoder(context_encoder=ctx_encoder)
        if irregular_freq == 1:
            decoder = MlpPoissonODEDecoder(x_dim=1, y_dim=y_dim, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        else:
            decoder = MlpPoissonODEDecoder(x_dim=1, y_dim=y_dim//2, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        super(MlpNeuralODEProcessPoisson, self).__init__(y_dim, r_dim, h_dim, L_dim, D_dim,
            encoder, decoder)
        
class MlpNeuralCDEProcess(NeuralODEProcess):
    def __init__(self, y_dim, r_dim, z_dim, h_dim, L_dim, initial_t, irregular_freq=1):
        D_dim = z_dim - L_dim
        ctx_encoder = Encoder(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim)
        encoder = SingleContextNDPEncoder(context_encoder=ctx_encoder)
        if irregular_freq == 1:
            decoder = MlpNormalCDEDecoder(x_dim=1, y_dim=y_dim, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        else:
            decoder = MlpNormalCDEDecoder(x_dim=1, y_dim=y_dim//2, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        super(MlpNeuralCDEProcess, self).__init__(y_dim, r_dim, h_dim, L_dim, D_dim,
            encoder, decoder)
        
class MlpNeuralCDEProcessPoisson(NeuralODEProcessPoisson):
    def __init__(self, y_dim, r_dim, z_dim, h_dim, L_dim, initial_t, irregular_freq=1):
        D_dim = z_dim - L_dim
        ctx_encoder = Encoder(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim)
        encoder = SingleContextNDPEncoder(context_encoder=ctx_encoder)
        if irregular_freq == 1:
            decoder = MlpPoissonCDEDecoder(x_dim=1, y_dim=y_dim, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        else:
            decoder = MlpPoissonCDEDecoder(x_dim=1, y_dim=y_dim//2, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        super(MlpNeuralCDEProcessPoisson, self).__init__(y_dim, r_dim, h_dim, L_dim, D_dim,
            encoder, decoder)

############################################
# [santanu]: Skipping total SDE for now
class MlpNeuralODEProcessLSTM_last_op(NeuralODEProcessLSTM):
    def __init__(self, y_dim, r_dim, z_dim, h_dim, L_dim, initial_t, batch_size, irregular_freq=1):
        D_dim = z_dim - L_dim
        ctx_encoder = LSTM_Encoder_last_op(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim, batch_size=batch_size)
        encoder = SingleContextNDPEncoder(context_encoder=ctx_encoder)
        if irregular_freq == 1:
            decoder = MlpNormalODEDecoder(x_dim=1, y_dim=y_dim, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        else:
            decoder = MlpNormalODEDecoder(x_dim=1, y_dim=y_dim//2, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        super(MlpNeuralODEProcessLSTM_last_op, self).__init__(y_dim, r_dim, h_dim, L_dim, D_dim,
            encoder, decoder)
        
class MlpNeuralODEProcessPoissonLSTM_last_op(NeuralODEProcessPoissonLSTM):
    def __init__(self, y_dim, r_dim, z_dim, h_dim, L_dim, initial_t, batch_size, irregular_freq=1) :
        D_dim = z_dim - L_dim
        ctx_encoder = LSTM_Encoder_last_op(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim, batch_size=batch_size)        
        encoder = SingleContextNDPEncoder(context_encoder=ctx_encoder)
        if irregular_freq == 1:
            decoder = MlpPoissonODEDecoder(x_dim=1, y_dim=y_dim, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        else:
            decoder = MlpPoissonODEDecoder(x_dim=1, y_dim=y_dim//2, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        super(MlpNeuralODEProcessPoissonLSTM_last_op, self).__init__(y_dim, r_dim, h_dim, L_dim, D_dim,
            encoder, decoder)
        
class MlpNeuralODEProcessBILSTM(NeuralODEProcessLSTM):
    def __init__(self, y_dim, r_dim, z_dim, h_dim, L_dim, initial_t, batch_size, irregular_freq=1):
        D_dim = z_dim - L_dim
        # ctx_encoder = LSTM_Encoder_all_op(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim, batch_size=batch_size) # the output dimension is similar to Encoder
        ctx_encoder = BILSTM_Encoder_last_op(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim, batch_size=batch_size)
        encoder = SingleContextNDPEncoder(context_encoder=ctx_encoder)
        if irregular_freq == 1:
            decoder = MlpNormalODEDecoder(x_dim=1, y_dim=y_dim, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        else:
            decoder = MlpNormalODEDecoder(x_dim=1, y_dim=y_dim//2, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        super(MlpNeuralODEProcessBILSTM, self).__init__(y_dim, r_dim, h_dim, L_dim, D_dim,
            encoder, decoder)
        
class MlpNeuralODEProcessPoissonBILSTM(NeuralODEProcessPoissonLSTM):
    def __init__(self, y_dim, r_dim, z_dim, h_dim, L_dim, initial_t, batch_size, irregular_freq=1):
        D_dim = z_dim - L_dim
        # ctx_encoder = LSTM_Encoder_all_op(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim, batch_size=batch_size)
        ctx_encoder = BILSTM_Encoder_last_op(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim, batch_size=batch_size)
        encoder = SingleContextNDPEncoder(context_encoder=ctx_encoder)
        if irregular_freq == 1:
            decoder = MlpPoissonODEDecoder(x_dim=1, y_dim=y_dim, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        else:
            decoder = MlpPoissonODEDecoder(x_dim=1, y_dim=y_dim//2, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        super(MlpNeuralODEProcessPoissonBILSTM, self).__init__(y_dim, r_dim, h_dim, L_dim, D_dim,
            encoder, decoder)


class MlpNeuralODEProcessODE_Encoder(NeuralODEProcessLSTM):
    def __init__(self, y_dim, r_dim, z_dim, h_dim, L_dim, initial_t, batch_size, irregular_freq=1):
        D_dim = z_dim - L_dim
        # ctx_encoder = NeuralODEEncoder(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim, batch_size=batch_size)
        ctx_encoder = Encoder_ODE_RNN(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim, batch_size=batch_size)
        encoder = SingleContextNDPEncoder(context_encoder=ctx_encoder)
        # encoder = SingleContextNDPEncoder_for_ode_encoder(context_encoder=ctx_encoder)
        if irregular_freq == 1:
            decoder = MlpNormalODEDecoder(x_dim=1, y_dim=y_dim, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        else:
            decoder = MlpNormalODEDecoder(x_dim=1, y_dim=y_dim//2, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        super(MlpNeuralODEProcessODE_Encoder, self).__init__(y_dim, r_dim, h_dim, L_dim, D_dim,
            encoder, decoder)
        
class MlpNeuralODEProcessPoissonODE_Encoder(NeuralODEProcessPoissonLSTM):
    def __init__(self, y_dim, r_dim, z_dim, h_dim, L_dim, initial_t, batch_size, irregular_freq=1):
        D_dim = z_dim - L_dim
        # ctx_encoder = NeuralODEEncoder(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim, batch_size=batch_size)        
        ctx_encoder = Encoder_ODE_RNN(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim, batch_size=batch_size)
        encoder = SingleContextNDPEncoder(context_encoder=ctx_encoder)
        # encoder = SingleContextNDPEncoder_for_ode_encoder(context_encoder=ctx_encoder)
        if irregular_freq == 1:
            decoder = MlpPoissonODEDecoder(x_dim=1, y_dim=y_dim, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        else:
            decoder = MlpPoissonODEDecoder(x_dim=1, y_dim=y_dim//2, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        super(MlpNeuralODEProcessPoissonODE_Encoder, self).__init__(y_dim, r_dim, h_dim, L_dim, D_dim,
            encoder, decoder)



class MlpNeuralODEProcessLSTM_all_op(NeuralODEProcessLSTM):
    def __init__(self, y_dim, r_dim, z_dim, h_dim, L_dim, initial_t, batch_size, irregular_freq=1):
        D_dim = z_dim - L_dim
        ctx_encoder = LSTM_Encoder_all_op(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim, batch_size=batch_size) # the output dimension is similar to Encoder
        encoder = SingleContextNDPEncoder(context_encoder=ctx_encoder)
        if irregular_freq == 1:
            decoder = MlpNormalODEDecoder(x_dim=1, y_dim=y_dim, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        else:
            decoder = MlpNormalODEDecoder(x_dim=1, y_dim=y_dim//2, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        super(MlpNeuralODEProcessLSTM_all_op, self).__init__(y_dim, r_dim, h_dim, L_dim, D_dim,
            encoder, decoder)
        
class MlpNeuralODEProcessPoissonLSTM_all_op(NeuralODEProcessPoissonLSTM):
    def __init__(self, y_dim, r_dim, z_dim, h_dim, L_dim, initial_t, batch_size, irregular_freq=1):
        D_dim = z_dim - L_dim
        ctx_encoder = LSTM_Encoder_all_op(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim, batch_size=batch_size)        
        encoder = SingleContextNDPEncoder(context_encoder=ctx_encoder)
        if irregular_freq == 1:
            decoder = MlpPoissonODEDecoder(x_dim=1, y_dim=y_dim, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        else:
            decoder = MlpPoissonODEDecoder(x_dim=1, y_dim=y_dim//2, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                initial_t=initial_t)
        super(MlpNeuralODEProcessPoissonLSTM_all_op, self).__init__(y_dim, r_dim, h_dim, L_dim, D_dim,
            encoder, decoder)

class MlpNeuralODE2Process(NeuralODEProcess):
    def __init__(self, y_dim, r_dim, z_dim, h_dim, L_dim, initial_t):
        D_dim = z_dim - L_dim
        ctx_encoder = Encoder(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim)
        encoder = SingleContextNDPEncoder(context_encoder=ctx_encoder)
        decoder = MlpSonodeDecoder(x_dim=1, y_dim=y_dim, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
            initial_t=initial_t)
        super(MlpNeuralODE2Process, self).__init__(y_dim, r_dim, h_dim, L_dim, D_dim,
            encoder, decoder)


class VanillaNeuralODEProcess(NeuralODEProcess):
    def __init__(self, y_dim, r_dim, z_dim, h_dim, L_dim, initial_t):
        D_dim = z_dim - L_dim
        ctx_encoder = Encoder(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim)
        encoder = SingleContextNDPEncoder(context_encoder=ctx_encoder)
        decoder = VanillaODEDecoder(x_dim=1, y_dim=y_dim, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
            initial_t=initial_t)
        super(VanillaNeuralODEProcess, self).__init__(y_dim, r_dim, h_dim, L_dim, D_dim,
            encoder, decoder)


class VanillaNeuralODE2Process(NeuralODEProcess):
    def __init__(self, y_dim, r_dim, z_dim, h_dim, L_dim, initial_t):
        D_dim = z_dim - L_dim
        ctx_encoder = Encoder(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim)
        encoder = SingleContextNDPEncoder(context_encoder=ctx_encoder)

        decoder = VanillaSONODEDecoder(x_dim=1, y_dim=y_dim, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
            initial_t=initial_t)
        super(VanillaNeuralODE2Process, self).__init__(y_dim, r_dim, h_dim, L_dim, D_dim,
            encoder, decoder)

class MlpNeuralProcess(TimeNeuralProcess):
    def __init__(self, y_dim, r_dim, z_dim, h_dim, irregular_freq=1):
        encoder = Encoder(x_dim=1, y_dim=y_dim, r_dim=r_dim, h_dim=h_dim)
        if irregular_freq == 1:
            decoder = NPMlpDecoder(x_dim=1, z_dim=z_dim, h_dim=h_dim, y_dim=y_dim)
        else:
            decoder = NPMlpDecoder(x_dim=1, z_dim=z_dim, h_dim=h_dim, y_dim=y_dim//2)
        super(MlpNeuralProcess, self).__init__(y_dim, r_dim, z_dim, h_dim, encoder, decoder)