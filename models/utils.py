import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import torch.nn as nn

def context_target_split(x, y, num_context, num_extra_target, locations = None, use_y0=True, ordered_time=False):
	"""Given inputs x and their value y, return random subsets of points for
	context and target. Note that following conventions from "Empirical
	Evaluation of Neural Process Objectives" the context points are chosen as a
	subset of the target points.

	Parameters
	----------
	x : torch.Tensor
		Shape (batch_size, num_points, x_dim)

	y : torch.Tensor
		Shape (batch_size, num_points, y_dim)

	num_context : int
		Number of context points.

	num_extra_target : int
		Number of additional target points.
	"""
	num_points = x.shape[1]
	# Sample locations of context and target points
	if locations is None:
		points = np.arange(num_points)
		size = num_context + num_extra_target
		initial_loc = np.array([])
		if use_y0:
			# Make sure that y0 is part of the context and that it is not sampled twice.
			points = points[1:]
			size -= 1
			initial_loc = np.array([0])
		# import pdb; pdb.set_trace()
		
		# [santanu]: this is the culprit for the unordered-ness of things
		if not ordered_time:
			locations = np.random.choice(points, size=size, replace=False)
		elif ordered_time:
			locations = points[:size]
			# locations= np.random.choice(points[:size], size=size, replace=False)
			
		locations = np.concatenate([initial_loc, locations])
	
	if not isinstance(locations, torch.Tensor):
		locations = torch.tensor(locations, device=x.device)

	# Use tensor indexing for efficient slicing
	context_indices = locations[:num_context]
	target_indices = locations
	# import pdb; pdb.set_trace()
	x_context = x.index_select(1, context_indices)
	y_context = y.index_select(1, context_indices)
	x_target = x.index_select(1, target_indices)
	y_target = y.index_select(1, target_indices)
	y0 = y[:, 0, :]

	# import pdb; pdb.set_trace()
	return x_context, y_context, x_target, y_target, y0

	# import pdb; pdb.set_trace()
	# x_context = x[:, locations[:num_context], :]
	# y_context = y[:, locations[:num_context], :]
	# x_target = x[:, locations, :]
	# y_target = y[:, locations, :]
	# # import pdb; pdb.set_trace()
	# return x_context, y_context, x_target, y_target, y[:, 0, :]

	# debug trial
	# print("Locations shape:", locations.shape)
	# print("x shape:", x.shape)
	# print("y shape:", y.shape)
	# print("num_context:", num_context)

	# # Ensure locations is a tensor on the same device as x and y
	# if not isinstance(locations, torch.Tensor):
	# 	locations = torch.tensor(locations, device=x.device)

	# # Check if locations are within bounds
	# if locations.max() >= x.shape[1] or locations.min() < 0:
	# 	raise ValueError(f"Locations out of bounds. Max location: {locations.max()}, x shape: {x.shape}")

	# # Safeguard against num_context being larger than available locations
	# num_context = min(num_context, len(locations))

	# try:
	# 	x_context = x[:, locations[:num_context], :]
	# 	print("x_context shape:", x_context.shape)

	# 	import pdb; pdb.set_trace()
	
	# 	y_context = y[:, locations[:num_context], :]
	# 	print("y_context shape:", y_context.shape)

	# 	x_target = x[:, locations, :]
	# 	print("x_target shape:", x_target.shape)

	# 	y_target = y[:, locations, :]
	# 	print("y_target shape:", y_target.shape)

	# except Exception as e:
	# 	print(f"Error occurred: {str(e)}")
	# 	print(f"locations[:num_context]: {locations[:num_context]}")
	# 	raise

	# return x_context, y_context, x_target, y_target, y[:, 0, :]


def img_mask_to_np_input(img, mask, normalize=True):
	"""
	Given an image and a mask, return x and y tensors expected by Neural
	Process. Specifically, x will contain indices of unmasked points, e.g.
	[[1, 0], [23, 14], [24, 19]] and y will contain the corresponding pixel
	intensities, e.g. [[0.2], [0.73], [0.12]] for grayscale or
	[[0.82, 0.71, 0.5], [0.42, 0.33, 0.81], [0.21, 0.23, 0.32]] for RGB.

	Parameters
	----------
	img : torch.Tensor
		Shape (N, C, H, W). Pixel intensities should be in [0, 1]

	mask : torch.ByteTensor
		Binary matrix where 0 corresponds to masked pixel and 1 to a visible
		pixel. Shape (N, H, W). Note the number of unmasked pixels must be the
		SAME for every mask in batch.

	normalize : bool
		If true normalizes pixel locations x to [-1, 1] and pixel intensities to
		[-0.5, 0.5]
	"""
	batch_size, num_channels, height, width = img.size()
	# Create a mask which matches exactly with image size which will be used to
	# extract pixel intensities
	mask_img_size = mask.unsqueeze(1).repeat(1, num_channels, 1, 1)
	# Number of points corresponds to number of visible pixels in mask, i.e. sum
	# of non zero indices in a mask (here we assume every mask has same number
	# of visible pixels)
	num_points = mask[0].nonzero().size(0)
	# Compute non zero indices
	# Shape (num_nonzeros, 3), where each row contains index of batch, height and width of nonzero
	nonzero_idx = mask.nonzero()
	# The x tensor for Neural Processes contains (height, width) indices, i.e.
	# 1st and 2nd indices of nonzero_idx (in zero based indexing)
	x = nonzero_idx[:, 1:].view(batch_size, num_points, 2).float()
	# The y tensor for Neural Processes contains the values of non zero pixels
	y = img[mask_img_size].view(batch_size, num_channels, num_points)
	# Ensure correct shape, i.e. (batch_size, num_points, num_channels)
	y = y.permute(0, 2, 1)

	if normalize:
		# TODO: make this separate for height and width for non square image
		# Normalize x to [-1, 1]
		x = (x - float(height) / 2) / (float(height) / 2)
		# Normalize y's to [-0.5, 0.5]
		y -= 0.5

	return x, y


def random_context_target_mask(img_size, num_context, num_extra_target):
	"""Returns random context and target masks where 0 corresponds to a hidden
	value and 1 to a visible value. The visible pixels in the context mask are
	a subset of the ones in the target mask.

	Parameters
	----------
	img_size : tuple of ints
		E.g. (1, 32, 32) for grayscale or (3, 64, 64) for RGB.

	num_context : int
		Number of context points.

	num_extra_target : int
		Number of additional target points.
	"""
	_, height, width = img_size
	# Sample integers without replacement between 0 and the total number of
	# pixels. The measurements array will then contain pixel indices
	# corresponding to locations where pixels will be visible.
	measurements = np.random.choice(range(height * width),
									size=num_context + num_extra_target,
									replace=False)
	# Create empty masks
	context_mask = torch.zeros(width, height).byte()
	target_mask = torch.zeros(width, height).byte()
	# Update mask with measurements
	for i, m in enumerate(measurements):
		row = int(m / width)
		col = m % width
		target_mask[row, col] = 1
		if i < num_context:
			context_mask[row, col] = 1
	return context_mask, target_mask


def batch_context_target_mask(img_size, num_context, num_extra_target,
							  batch_size, repeat=False):
	"""Returns batch of context and target masks, where the visible pixels in
	the context mask are a subset of those in the target mask.

	Parameters
	----------
	img_size : see random_context_target_mask

	num_context : see random_context_target_mask

	num_extra_target : see random_context_target_mask

	batch_size : int
		Number of masks to create.

	repeat : bool
		If True, repeats one mask across batch.
	"""
	context_mask_batch = torch.zeros(batch_size, *img_size[1:]).byte()
	target_mask_batch = torch.zeros(batch_size, *img_size[1:]).byte()
	if repeat:
		context_mask, target_mask = random_context_target_mask(img_size,
															   num_context,
															   num_extra_target)
		for i in range(batch_size):
			context_mask_batch[i] = context_mask
			target_mask_batch[i] = target_mask
	else:
		for i in range(batch_size):
			context_mask, target_mask = random_context_target_mask(img_size,
																   num_context,
																   num_extra_target)
			context_mask_batch[i] = context_mask
			target_mask_batch[i] = target_mask
	return context_mask_batch, target_mask_batch


def xy_to_img(x, y, img_size):
	"""Given an x and y returned by a Neural Process, reconstruct image.
	Missing pixels will have a value of 0.

	Parameters
	----------
	x : torch.Tensor
		Shape (batch_size, num_points, 2) containing normalized indices.

	y : torch.Tensor
		Shape (batch_size, num_points, num_channels) where num_channels = 1 for
		grayscale and 3 for RGB, containing normalized pixel intensities.

	img_size : tuple of ints
		E.g. (1, 32, 32) for grayscale or (3, 64, 64) for RGB.
	"""
	_, height, width = img_size
	batch_size, _, _ = x.size()
	# Unnormalize x and y
	x = x * float(height / 2) + float(height / 2)
	x = x.long()
	y += 0.5
	# Permute y so it matches order expected by image
	# (batch_size, num_points, num_channels) -> (batch_size, num_channels, num_points)
	y = y.permute(0, 2, 1)
	# Initialize empty image
	img = torch.zeros((batch_size,) + img_size)
	for i in range(batch_size):
		img[i, :, x[i, :, 0], x[i, :, 1]] = y[i, :, :]
	return img


def inpaint(model, img, context_mask, device):
	"""
	Given an image and a set of context points, the model samples pixel
	intensities for the remaining pixels in the image.

	Parameters
	----------
	model : models.NeuralProcessImg instance

	img : torch.Tensor
		Shape (channels, height, width)

	context_mask : torch.Tensor
		Binary tensor where 1 corresponds to a visible pixel and 0 to an
		occluded pixel. Shape (height, width). Must have dtype=torch.uint8
		or similar. 

	device : torch.device
	"""
	is_training = model.neural_process.training
	# For inpainting, use Neural Process in prediction mode
	model.neural_process.training = False
	target_mask = 1 - context_mask	# All pixels which are not in context
	# Add a batch dimension to tensors and move to GPU
	img_batch = img.unsqueeze(0).to(device)
	context_batch = context_mask.unsqueeze(0).to(device)
	target_batch = target_mask.unsqueeze(0).to(device)
	p_y_pred = model(img_batch, context_batch, target_batch)
	# Transform Neural Process output back to image
	x_target, _ = img_mask_to_np_input(img_batch, target_batch)
	# Use the mean (i.e. loc) parameter of normal distribution as predictions
	# for y_target
	img_rec = xy_to_img(x_target.cpu(), p_y_pred.loc.detach().cpu(), img.size())
	img_rec = img_rec[0]  # Remove batch dimension
	# Add context points back to image
	context_mask_img = context_mask.unsqueeze(0).repeat(3, 1, 1)
	img_rec[context_mask_img] = img[context_mask_img]
	# Reset model to mode it was in before inpainting
	model.neural_process.training = is_training
	return img_rec


def create_plots_directory(base_dir='plots', exp_name= 'base'):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plots_dir = os.path.join(base_dir, exp_name+':'+timestamp)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir

# def visualize_gene_expression_histograms(y, p_y_pred, considered_genes, epoch=0, base_save_dir='plots', bins=np.linspace(0, 10, 100), time_points=None):
#     # Ensure y and p_y_pred are on the CPU and detached from the computational graph
#     y = y.cpu().detach().numpy()
#     p_y_sample = p_y_pred.sample().cpu().detach().numpy()

#     if time_points is None:
#         time_points = list(range(y.shape[1]))  # Default to all time points if none specified

#     num_time_points = len(time_points)
#     num_genes = y.shape[2]

#     epoch_save_dir = os.path.join(base_save_dir, f'epoch_{epoch}')
#     if not os.path.exists(epoch_save_dir):
#         os.makedirs(epoch_save_dir)

#     for g in considered_genes:
#         if g >= num_genes:
#             print(f"Gene index {g} is out of bounds for the number of genes {num_genes}")
#             continue
        
#         # Collect all gene expression values for the gene across specified time points
#         all_actual_exps = [y[:, t, g] for t in time_points]
#         all_pred_exps = [p_y_sample[:, t, g] for t in time_points]
        
#         # Calculate the bin width and offsets for the bars
#         bin_width = 0.8
#         bar_width = bin_width / (2 * num_time_points)  # Adjust for two sets: actual and predicted
#         offsets = np.linspace(-bar_width * (num_time_points - 1), bar_width * (num_time_points - 1), num_time_points)

#         plt.figure(figsize=(10, 6))

#         for j, t in enumerate(time_points):
#             actual_counts, _ = np.histogram(all_actual_exps[j], bins=bins, density=True)
#             pred_counts, _ = np.histogram(all_pred_exps[j], bins=bins, density=True)
#             bin_centers = 0.5 * (bins[:-1] + bins[1:])
            
#             plt.bar(bin_centers + offsets[j], actual_counts, width=bar_width, label=f'Actual t{t}', alpha=0.5)
#             plt.bar(bin_centers + offsets[j] + bar_width, pred_counts, width=bar_width, label=f'Predicted t{t}', alpha=0.5)

#         plt.title(f'Gene Expression Distribution for Gene {g} at Epoch {epoch}')
#         plt.legend(loc='upper right')
#         plt.xlabel('Value')
#         plt.ylabel('Density')
#         plt.savefig(os.path.join(epoch_save_dir, f'gene_{g}_epoch_{epoch}.png'))
#         plt.close()

def visualize_gene_expression_histograms(y, p_y_pred, considered_genes, data_type, epoch=0, base_save_dir='plots', bins=np.linspace(0, 10, 100), time_points=None):
    """
	script was written for gene-expression originally, slowly generalizing it
	"""
	# Ensure y and p_y_pred are on the CPU and detached from the computational graph
    y = y.cpu().detach().numpy()
    p_y_sample = p_y_pred.sample().cpu().detach().numpy()

    # import pdb; pdb.set_trace()

    if time_points is None:
        time_points = list(range(y.shape[1]))  # Default to all time points if none specified

    num_time_points = len(time_points)
    num_genes = y.shape[2]

    epoch_save_dir = os.path.join(base_save_dir, f'epoch_{epoch}')
    if not os.path.exists(epoch_save_dir):
        os.makedirs(epoch_save_dir)

    for g in considered_genes:
        if g >= num_genes:
            print(f"Index {g} is out of bounds for the number of genes {num_genes}")
            continue
        
        # Create subplots for each time point
        fig, axes = plt.subplots(num_time_points, 2, figsize=(15, 5 * num_time_points))
        
        for j, t in enumerate(time_points):
            ax_actual, ax_pred = axes[j] if num_time_points > 1 else (axes, axes)

            actual_exps = y[:, t, g]
            pred_exps = p_y_sample[:, t, g]

            # Histogram and line plot for actual gene expression values
            actual_counts, bins_actual, _ = ax_actual.hist(actual_exps, bins=bins, density=True, alpha=0.5, label='Actual')
            bin_centers_actual = 0.5 * (bins_actual[:-1] + bins_actual[1:])
            ax_actual.plot(bin_centers_actual, actual_counts, 'r-', label='Actual Line')
            ax_actual.set_title(f'Actual {data_type} at t{t}')
            ax_actual.set_xlabel('Value')
            ax_actual.set_ylabel('Density')
            ax_actual.legend()

            # Histogram and line plot for predicted gene expression values
            pred_counts, bins_pred, _ = ax_pred.hist(pred_exps, bins=bins, density=True, alpha=0.5, label='Predicted')
            bin_centers_pred = 0.5 * (bins_pred[:-1] + bins_pred[1:])
            ax_pred.plot(bin_centers_pred, pred_counts, 'b-', label='Predicted Line')
            ax_pred.set_title(f'Predicted {data_type} at t{t}')
            ax_pred.set_xlabel('Value')
            ax_pred.set_ylabel('Density')
            ax_pred.legend()

        fig.suptitle(f'{data_type} Distribution for x {g} at Epoch {epoch}')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(epoch_save_dir, f'x_{g}_epoch_{epoch}.png'))
        plt.close()


def init_network_weights(net, std = 0.1):
	for m in net.modules():
		if isinstance(m, nn.Linear):
			nn.init.normal_(m.weight, mean=0, std=std)
			nn.init.constant_(m.bias, val=0)


def create_net(n_inputs, n_outputs, n_layers = 1, 
	n_units = 100, nonlinear = nn.Tanh):
	layers = [nn.Linear(n_inputs, n_units)]
	for i in range(n_layers):
		layers.append(nonlinear())
		layers.append(nn.Linear(n_units, n_units))

	layers.append(nonlinear())
	layers.append(nn.Linear(n_units, n_outputs))
	return nn.Sequential(*layers)

def get_device(tensor):
	device = torch.device("cpu")
	if tensor.is_cuda:
		device = tensor.get_device()
	return device


def split_last_dim(data):
	last_dim = data.size()[-1]
	last_dim = last_dim//2

	if len(data.size()) == 3:
		res = data[:,:,:last_dim], data[:,:,last_dim:]

	if len(data.size()) == 2:
		res = data[:,:last_dim], data[:,last_dim:]
	return res

def check_mask(data, mask):
	#check that "mask" argument indeed contains a mask for data
	n_zeros = torch.sum(mask == 0.).cpu().numpy()
	n_ones = torch.sum(mask == 1.).cpu().numpy()

	# mask should contain only zeros and ones
	assert((n_zeros + n_ones) == np.prod(list(mask.size())))

	# all masked out elements should be zeros
	assert(torch.sum(data[mask == 0.] != 0.) == 0)

def linspace_vector(start, end, n_points):
	# start is either one value or a vector
	size = np.prod(start.size())

	assert(start.size() == end.size())
	if size == 1:
		# start and end are 1d-tensors
		res = torch.linspace(start, end, n_points)
	else:
		# start and end are vectors
		res = torch.Tensor()
		for i in range(0, start.size(0)):
			res = torch.cat((res, 
				torch.linspace(start[i], end[i], n_points)),0)
		res = torch.t(res.reshape(start.size(0), n_points))
	return res