import os

import numpy as np
import random
import torch
from utils.convert_ply import convert

import subprocess


def get_gpu_memory():
    """Return a list of dicts with memory stats (in MiB) for each GPU.

    Each dict has keys: index, memory_used, memory_total, memory_free.
    Returns None if nvidia-smi is unavailable or fails.
    """
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    gpus = []
    for line in output.strip().splitlines():
        index, used, total, free = (int(x) for x in line.split(","))
        gpus.append(
            {
                "index": index,
                "memory_used": used,
                "memory_total": total,
                "memory_free": free,
            }
        )
    return gpus[0]["memory_used"]


def seed_everything(seed=42):
    """
        Set the `seed` value for torch and numpy seeds. Also turns on
        deterministic execution for cudnn.
        
        Parameters:
        - seed:     A hashable seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed} (type: {type(seed)})")


def params2cpu(params):
    res = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            res[k] = v.detach().cpu().contiguous().numpy()
        else:
            res[k] = v
    return res


def save_params(output_params, output_dir, save_ply=True, file_prefix=""):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, file_prefix+"params.npz")
    np.savez(save_path, **to_save)
    if save_ply:
        convert(to_save, os.path.join(output_dir, file_prefix+"params.ply"), is_semantic=False)
        convert(to_save, os.path.join(output_dir, file_prefix+"seg_params.ply"), is_semantic=True)        


def save_params_ckpt(output_params, output_dir, time_idx):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **to_save)


def save_seq_params(all_params, output_dir):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **params_to_save)


def save_seq_params_ckpt(all_params, output_dir,time_idx):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **params_to_save)