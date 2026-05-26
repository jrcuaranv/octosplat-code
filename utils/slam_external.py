"""
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file found here:
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
#
# For inquiries contact  george.drettakis@inria.fr

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE #####
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################
"""

import numpy as np
import torch
import torch.nn.functional as func
from torch.autograd import Variable
from math import exp
import math
import open3d as o3d
from sklearn.neighbors import KDTree
print_colors = {
    'red': "\033[31m",
    'green': "\033[32m",
    'yellow': "\033[33m",
    'blue': "\033[34m",
        'reset': "\033[0m"
}
RED = print_colors['red']
GREEN = print_colors['green']
YELLOW = print_colors['yellow']
BLUE = print_colors['blue']
RESET = print_colors['reset']

def _compute_density_statistics(points, tree, density_k=8, radius_scale=1.8):
    # Fast path: try using scipy + sklearn sparse graph ops to avoid Python loops
    density_k = max(3, int(density_k))
    N = points.shape[0]
    query_k = min(N, density_k + 1)

    # Attempt to import scipy; if missing, fall back to original loop-based implementation
    try:
        from sklearn.neighbors import NearestNeighbors
        import scipy.sparse as sp
        print("Using optimized density statistics computation with sparse graph operations.")
    except Exception:
        print("Scipy or sklearn not available, falling back to original density statistics computation.")
        # fallback: original implementation using tree.query_radius
        density_k = max(3, int(density_k))
        query_k = min(points.shape[0], density_k + 1)
        distances, indices = tree.query(points, k=query_k, return_distance=True)
        if query_k <= 1:
            return distances, indices, np.zeros(points.shape[0], dtype=bool), np.ones(points.shape[0]) * np.finfo(np.float64).eps

        neighbor_distances = distances[:, 1:query_k]
        kth_distances = neighbor_distances[:, -1]
        radius = float(np.median(kth_distances) * radius_scale)
        radius = max(radius, np.finfo(np.float64).eps)

        radius_neighbors = tree.query_radius(points, r=radius)
        avg_neighbor_distance = neighbor_distances.mean(axis=1)
        global_mean = float(avg_neighbor_distance.mean())
        global_std = float(avg_neighbor_distance.std(ddof=0)) + np.finfo(np.float64).eps

        local_means = np.empty(points.shape[0], dtype=np.float64)
        local_stds = np.empty(points.shape[0], dtype=np.float64)
        local_counts = np.empty(points.shape[0], dtype=np.int32)

        for idx, neighbors in enumerate(radius_neighbors):
            neighbors = neighbors[neighbors != idx]
            local_counts[idx] = neighbors.size
            if neighbors.size < 3:
                local_means[idx] = global_mean
                local_stds[idx] = global_std
                continue

            local_values = avg_neighbor_distance[neighbors]
            local_means[idx] = float(local_values.mean())
            local_stds[idx] = float(local_values.std(ddof=0)) + np.finfo(np.float64).eps

        low_density = local_counts < max(4, density_k // 2)
        distance_outlier = avg_neighbor_distance > (local_means + 1.75 * local_stds)
        distance_ratio_outlier = avg_neighbor_distance > (local_means * 1.35)
        noisy_mask = low_density | distance_outlier | distance_ratio_outlier
        return distances, indices, noisy_mask, radius_neighbors

    # --- optimized path using sparse graph operations ---
    nbrs = NearestNeighbors(n_neighbors=query_k, algorithm='kd_tree', n_jobs=-1).fit(points)
    distances, indices = nbrs.kneighbors(points)
    if query_k <= 1:
        return distances, indices, np.zeros(N, dtype=bool), [np.array([], dtype=np.int32) for _ in range(N)]

    neighbor_distances = distances[:, 1:query_k]  # shape (N, k-1)
    kth = neighbor_distances[:, -1]
    radius = float(np.median(kth) * radius_scale)
    radius = max(radius, np.finfo(np.float64).eps)

    # Build sparse connectivity matrix (CSR). mode='connectivity' returns a 0/1 matrix.
    A = nbrs.radius_neighbors_graph(points, radius, mode='connectivity').tocsr()
    # remove self-connections and empty entries
    try:
        A.setdiag(0)
    except Exception:
        pass
    A.eliminate_zeros()

    avg_neighbor_distance = neighbor_distances.mean(axis=1)
    global_mean = float(avg_neighbor_distance.mean())
    global_std = float(avg_neighbor_distance.std(ddof=0)) + np.finfo(np.float64).eps

    # local counts
    local_counts = np.asarray(A.sum(axis=1)).ravel().astype(np.int32)

    # Sparse matmul to compute sum and sumsq of neighbor avg distances
    s1 = A.dot(avg_neighbor_distance)
    s2 = A.dot(avg_neighbor_distance**2)

    local_means = np.empty(N, dtype=np.float64)
    local_stds = np.empty(N, dtype=np.float64)

    nonzero = local_counts > 0
    local_means[nonzero] = s1[nonzero] / local_counts[nonzero]
    mean_sq = np.zeros_like(s1)
    mean_sq[nonzero] = s2[nonzero] / local_counts[nonzero]
    var = np.maximum(mean_sq - local_means**2, 0.0)
    local_stds[nonzero] = np.sqrt(var[nonzero]) + np.finfo(np.float64).eps

    small_mask = local_counts < 3
    local_means[small_mask] = global_mean
    local_stds[small_mask] = global_std

    low_density = local_counts < max(4, density_k // 2)
    distance_outlier = avg_neighbor_distance > (local_means + 1.75 * local_stds)
    distance_ratio_outlier = avg_neighbor_distance > (local_means * 1.35)
    noisy_mask = low_density | distance_outlier | distance_ratio_outlier

    # Convert sparse adjacency to list-of-arrays for compatibility with callers
    indptr = A.indptr
    indices_array = A.indices
    radius_neighbors = [indices_array[indptr[i]:indptr[i+1]].copy() for i in range(N)]

    return distances, indices, noisy_mask, radius_neighbors


def build_rotation(q, device="cuda"):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device=device)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def calc_mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def calc_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calc_iou(img1, img2):
    pass

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def calc_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = func.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = func.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = func.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = func.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = func.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def accumulate_mean2d_gradient(variables):
    # print("Variables['means2D_gradient_accum'].shape:", variables['means2D_gradient_accum'].shape) #jrcv added
    # print("Variables['means2D'].shape:", variables['means2D'].shape) #jrcv added
    # print("Variables['means2D'].grad.shape:", variables['means2D'].grad.shape) #jrcv added
    # print("Variables['seen'].shape:", variables['seen'].shape) #jrcv added
    # if means2D is missing or means2D.grad is None, return variables
    if 'means2D' not in variables.keys() or variables['means2D'].grad is None:
        return variables

    n = min(
        variables['means2D_gradient_accum'].shape[0],
        variables['seen'].shape[0],
        variables['means2D'].grad.shape[0]
    )
    seen_n = variables['seen'][:n]
    grad_n = variables['means2D'].grad[:n, :2]

    variables['means2D_gradient_accum'][:n][seen_n] += torch.norm(grad_n[seen_n], dim=-1)
    variables['denom'][:n][seen_n] += 1
    return variables


def update_params_and_optimizer(new_params, params, params_opt_exclude, optimizer):
    for k, v in new_params.items():
        if k in params_opt_exclude:
            params[k] = new_params[k]
            continue
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)

        stored_state["exp_avg"] = torch.zeros_like(v)
        stored_state["exp_avg_sq"] = torch.zeros_like(v)
        del optimizer.state[group['params'][0]]

        group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
        optimizer.state[group['params'][0]] = stored_state
        params[k] = group["params"][0]
    return params


def cat_params_to_optimizer(new_params, params, params_opt_exclude, optimizer):
    for k, v in new_params.items():
        if k in params_opt_exclude:
            params[k] = torch.cat((params[k], v), dim=0)
            continue
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
            del optimizer.state[group['params'][0]]
            # if group['params'][0].dim() == 1:
            #     group["params"][0] = group["params"][0].view(-1, 1)
            # if v.dim() == 1:
            #     v = v.view(-1, 1)
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            # if group['params'][0].dim() == 1:
            #     group["params"][0] = group["params"][0].view(-1, 1)
            # if v.dim() == 1:
            #     v = v.view(-1, 1)
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            params[k] = group["params"][0]
    return params


def remove_points(to_remove, params, params_opt_exclude, variables, optimizer):
    to_keep = ~to_remove
    keys = [k for k in params.keys() if k not in ['cam_unnorm_rots', 'cam_trans', 'means3D_2', 'unnorm_rotations_2', 'logit_opacities_2', 'log_scales_2']]
    for k in keys:
        # Keys not in optimizer
        if k in params_opt_exclude:
            params[k] = params[k][to_keep]
            continue
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            params[k] = group["params"][0]
    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
    variables['denom'] = variables['denom'][to_keep]
    variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
    # variables['seen'] = variables['seen'][to_keep] # jrcv added. Not recommended.
    # variables['means2D'] = variables['means2D'][to_keep] # jrcv added. Its wrong. Commented
    # variables['timestep'] = variables['timestep'][to_keep] # jrcv added
    

    if 'timestep' in variables.keys():
        variables['timestep'] = variables['timestep'][to_keep]
    return params, variables


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def prune_gaussians(params, params_opt_exclude, variables, optimizer, iter, prune_dict):
    if iter <= prune_dict['stop_after']:
        if (iter >= prune_dict['start_after']) and (iter % prune_dict['prune_every'] == 0):
            print(f"{RED}\nPruning Gaussians at iteration {iter}{RESET}")
            if iter == prune_dict['stop_after']:
                remove_threshold = prune_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = prune_dict['removal_opacity_threshold']
            # Remove Gaussians with low opacity
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            print(f"Iteration {iter}: (Pruning) Number of points to remove based on opacity: {to_remove.sum().item()}")
            # Remove Gaussians that are too big
            if iter >= prune_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 5e-3 #0.1 * variables['scene_radius']
                print(f"Iteration {iter}: (Pruning) Number of big points to remove based on size: {big_points_ws.sum().item()}")
                to_remove = torch.logical_or(to_remove, big_points_ws)
            print(f"Iteration {iter}: (Prunning) Total Number of points to remove: {to_remove.sum().item()}")
            params, variables = remove_points(to_remove, params, params_opt_exclude, variables, optimizer)
            torch.cuda.empty_cache()
        
        # Reset Opacities for all Gaussians
        if iter > 0 and iter % prune_dict['reset_opacities_every'] == 0 and prune_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)
        
    return params, variables

def prune_aux_gaussians(params, params_opt_exclude, variables, optimizer):
    to_remove = params['semantic_ids']==0.555
    params, variables = remove_points(to_remove, params, params_opt_exclude, variables, optimizer)
    torch.cuda.empty_cache()
    return params, variables

def prune_outlier_semantics(params, params_opt_exclude, variables, optimizer, device = "cuda"):
    semantic_targets = [[1,0,0],[0,0,0],[0,1,0]] #rgb semantics
    print("Prunning outlier semantics")
    # np.savetxt('/home/jose/params.txt',params['semantic_colors'].detach().cpu().numpy())
    # np.savetxt('/home/jose/means3D.txt',params['means3D'].detach().cpu().numpy())
    # np.savetxt('/home/jose/opt_count.txt',params['opt_count'].detach().cpu().numpy())
    masks = []
    for sem_t in semantic_targets:

        sem_target = torch.tensor(sem_t).to(device)
        rmse = torch.linalg.norm(sem_target - params['semantic_colors'].clip(0,1), axis=1)/math.sqrt(3)
        to_keep_mask = rmse < 0.01
        masks.append(to_keep_mask)
    
    to_keep = masks[0] | masks[1] | masks[2]
    to_remove = ~to_keep
    print("Number of points to remove based on semantic outlier pruning: {}".format(to_remove.sum().item()))
    # to_remove = 0*to_remove

    
    # to_remove = params['semantic_ids'][invalid_sem_mask]
    params, variables = remove_points(to_remove, params, params_opt_exclude, variables, optimizer)
    
    params['semantic_colors'] = params['semantic_colors'].clip(0,1) # might not be necessary
    torch.cuda.empty_cache()
    return params, variables

def prune_background_semantics(params, params_opt_exclude, variables, optimizer, iter, prune_dict, device = "cuda"):
    if iter <= prune_dict['stop_after']:
        if (iter >= prune_dict['start_after']) and (iter % prune_dict['prune_every'] == 0):
            semantic_target = [0,0,0]
            print(f"{BLUE}\nPrunning background semantics at iteration {iter}{RESET}")
            sem_target = torch.tensor(semantic_target).to(device)
            rmse = torch.linalg.norm(sem_target - params['semantic_colors'].clip(0,1), axis=1)/math.sqrt(3)
            to_remove = rmse < 0.2
            
            print("Number of points to remove based on background pruning: {}".format(to_remove.sum().item()))
            params, variables = remove_points(to_remove, params, params_opt_exclude, variables, optimizer)
            
            # params['semantic_colors'] = params['semantic_colors'].clip(0,1) # might not be necessary
            torch.cuda.empty_cache()
    return params, variables


def prune_outliers(params, params_opt_exclude, variables, optimizer, device = "cuda"):
    # Prune outliers based on radius outlier removal in Open3D
    means3D = params['means3D'].detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means3D)
    pcd, inliers_idx = pcd.remove_radius_outlier(nb_points=10, radius=0.04)
    
    to_keep = np.zeros(len(means3D), dtype=bool)
    to_keep[inliers_idx] = True
    to_remove = ~to_keep
    params, variables = remove_points(to_remove, params, params_opt_exclude, variables, optimizer)
    
    torch.cuda.empty_cache()
    return params, variables

def prune_outliers_based_on_density_statistics(params, params_opt_exclude, variables, optimizer, iter, prune_dict, device = "cuda"):
    if iter <= prune_dict['stop_after']:
        if (iter >= prune_dict['start_after']) and (iter % prune_dict['prune_every'] == 0):
            means3D = params['means3D'].detach().cpu().numpy()
            tree = KDTree(means3D, leaf_size=40, metric="euclidean")
            knn_k = min(means3D.shape[0], 24) 
            _, _, noisy_mask, _ = _compute_density_statistics(means3D, tree, density_k=min(8, knn_k - 1), radius_scale=1.8) 
            # tree = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(means3D))
            # _, _, noisy_mask, _ = _compute_density_statistics(means3D, tree, density_k=8, radius_scale=1.8)
            to_remove = noisy_mask
            print(f"{BLUE}Iteration {iter}: Number of points to remove based on density statistics: {to_remove.sum().item()}{RESET}")
            params, variables = remove_points(to_remove, params, params_opt_exclude, variables, optimizer)
            
            torch.cuda.empty_cache()
    return params, variables

def densify(params, variables, optimizer, iter, densify_dict, params_opt_exclude, device="cuda"):
    if iter <= densify_dict['stop_after']:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = densify_dict['grad_thresh']
        if (iter >= densify_dict['start_after']) and (iter % densify_dict['densify_every'] == 0):
            print("\nDensifying at iteration {}".format(iter))
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0
            to_clone = torch.logical_and(grads >= grad_thresh, (
                        torch.max(torch.exp(params['log_scales']), dim=1).values <= 0.01 * variables['scene_radius']))
            
            print("Iteration {}: Number of points to clone: {}".format(iter, to_clone.sum().item()))
            new_params = {k: v[to_clone] for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}
            params = cat_params_to_optimizer(new_params, params, params_opt_exclude, optimizer)
            num_pts = params['means3D'].shape[0]

            padded_grad = torch.zeros(num_pts, device=device)
            padded_grad[:grads.shape[0]] = grads
            to_split = torch.logical_and(padded_grad >= grad_thresh,
                                         torch.max(torch.exp(params['log_scales']), dim=1).values > 0.01 * variables[
                                             'scene_radius'])
            n = densify_dict['num_to_split_into']  # number to split into
            new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}
            stds = torch.exp(params['log_scales'])[to_split].repeat(n, 3)
            means = torch.zeros((stds.size(0), 3), device=device)
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(params['unnorm_rotations'][to_split], device=device).repeat(n, 1, 1)
            new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n))
            params = cat_params_to_optimizer(new_params, params, params_opt_exclude, optimizer)
            num_pts = params['means3D'].shape[0]

            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device=device)
            variables['denom'] = torch.zeros(num_pts, device=device)
            variables['max_2D_radius'] = torch.zeros(num_pts, device=device)
            to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device=device)))
            params, variables = remove_points(to_remove, params, params_opt_exclude, variables, optimizer)

            if iter == densify_dict['stop_after']:
                remove_threshold = densify_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = densify_dict['removal_opacity_threshold']
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            if iter >= densify_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                print(f"Iteration {iter}: Number of big points to remove: {big_points_ws.sum().item()}")
                to_remove = torch.logical_or(to_remove, big_points_ws)
                print(f"Iteration {iter}: Total number of points to remove: {to_remove.sum().item()}")
            params, variables = remove_points(to_remove, params, params_opt_exclude, variables, optimizer)

            torch.cuda.empty_cache()

        # Reset Opacities for all Gaussians (This is not desired for mapping on only current frame)
        if iter > 0 and iter % densify_dict['reset_opacities_every'] == 0 and densify_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, params_opt_exclude, optimizer)

    return params, variables

def densify_v2(params, variables, optimizer, iter, densify_dict, params_opt_exclude, device="cuda"):
    if iter <= densify_dict['stop_after']:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = densify_dict['grad_thresh']
        if (iter >= densify_dict['start_after']) and (iter % densify_dict['densify_every'] == 0):
            print(f"{GREEN}\nDensifying at iteration {iter}{RESET}")
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0
            ######
            # semantic_targets = [[1,0,0],[0,1,0]] #rgb semantics
            # masks = []
            # for sem_t in semantic_targets:
            #     sem_target = torch.tensor(sem_t).to(device)
            #     rmse = torch.linalg.norm(sem_target - params['semantic_colors'].clip(0,1), axis=1)/math.sqrt(3)
            #     to_keep_mask = rmse < 0.01
            #     masks.append(to_keep_mask)
            # to_keep_sem = masks[0] | masks[1]
            
            ######
            # print("INitial params: \n")
            # for k, v in params.items():
            #     print(f"params[{k}].shape: {v.shape}")

            to_clone = torch.logical_and(grads >= grad_thresh, (
                        torch.max(torch.exp(params['log_scales']), dim=1).values <= 1e-3))#0.01 * variables['scene_radius']))
            # to_clone = torch.logical_and(to_clone, to_keep_sem) # only clone points with valid semantics # jrcv added. TESTING
            # to_clone = 0*to_clone # fixing a bug. TODO: fix logic and remove
            print(f"Iteration {iter}: Number of points to clone: {to_clone.sum().item()}")
            if to_clone.sum() > 0:
                
                new_params = {k: v[to_clone] for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}
                params = cat_params_to_optimizer(new_params, params, params_opt_exclude, optimizer)
                
            num_pts = params['means3D'].shape[0]

            padded_grad = torch.zeros(num_pts, device=device)
            padded_grad[:grads.shape[0]] = grads
            to_split = torch.logical_and(padded_grad >= grad_thresh,
                                            torch.max(torch.exp(params['log_scales']), dim=1).values > 2.5e-3) #0.01 * variables['scene_radius'])
            print(f"Iteration {iter}: Number of points to split: {to_split.sum().item()}")
            # to_split = 0*to_split # fixing a bug. TODO: fix logic and remove
            if to_split.sum() > 2:
                n = densify_dict['num_to_split_into']  # number to split into
                new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}
                stds = torch.exp(params['log_scales'])[to_split].repeat(n, 3)
                means = torch.zeros((stds.size(0), 3), device=device)
                samples = torch.normal(mean=means, std=stds)
                rots = build_rotation(params['unnorm_rotations'][to_split], device=device).repeat(n, 1, 1)
                new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1).clip(-0.003,0.003) # adding a small clip to prevent outliers due to large splits. TODO: fix logic and remove
                new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n)) # scale down the new points
                params = cat_params_to_optimizer(new_params, params, params_opt_exclude, optimizer)
            
            num_pts = params['means3D'].shape[0]

            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device=device)
            variables['denom'] = torch.zeros(num_pts, device=device)
            variables['max_2D_radius'] = torch.zeros(num_pts, device=device)
            variables['timestep'] = torch.zeros(num_pts, device=device) #jrcv added
            variables['seen'] = torch.zeros(num_pts, dtype=torch.bool, device=device) # jrcv added
            # variables['means2D'] = torch.zeros(num_pts, device=device) #jrcv added. Its wrong.
            print("\nVariables scene radius:", variables['scene_radius'])
            if to_split.sum() > 2:
                to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device=device)))
                params, variables = remove_points(to_remove, params, params_opt_exclude, variables, optimizer)

            if iter == densify_dict['stop_after']:
                remove_threshold = densify_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = densify_dict['removal_opacity_threshold']
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            print(f"Iteration {iter}: Number of points to remove based on opacity: {to_remove.sum().item()}")
            if iter >= densify_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 5e-3 #0.02 * variables['scene_radius']
                print(f"Iteration {iter}: Number of big points to remove: {big_points_ws.sum().item()}")
                to_remove = torch.logical_or(to_remove, big_points_ws)
                print(f"Iteration {iter}: Total number of points to remove: {to_remove.sum().item()}")
            params, variables = remove_points(to_remove, params, params_opt_exclude, variables, optimizer)

            torch.cuda.empty_cache()
        # Reset Opacities for all Gaussians (This is not desired for mapping on only current frame)
        if iter > 0 and iter % densify_dict['reset_opacities_every'] == 0 and densify_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, params_opt_exclude, optimizer)

        # print("Variables['max_2D_radius'].shape after removal:", variables['max_2D_radius'].shape)
        # print("Params['means3D'].shape after removal:", params['means3D'].shape)
        
    return params, variables



def update_learning_rate(optimizer, means3D_scheduler, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in optimizer.param_groups:
            if param_group["name"] == "means3D":
                lr = means3D_scheduler(iteration)
                param_group['lr'] = lr
                return lr


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper