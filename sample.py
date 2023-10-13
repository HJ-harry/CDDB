import os
import copy
import argparse
import random
from pathlib import Path
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

from logger import Logger
import distributed_util as dist_util
from i2sb import Runner, download_ckpt
from corruption import build_corruption
from dataset import imagenet
from i2sb import ckpt_util

import colored_traceback.always
from ipdb import set_trace as debug

RESULT_DIR = Path("results")


def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def build_subset_per_gpu(opt, dataset, log):
    n_data = len(dataset)
    n_gpu  = opt.global_size
    n_dump = (n_data % n_gpu > 0) * (n_gpu - n_data % n_gpu)

    # create index for each gpu
    total_idx = np.concatenate([np.arange(n_data), np.zeros(n_dump)]).astype(int)
    idx_per_gpu = total_idx.reshape(-1, n_gpu)[:, opt.global_rank]
    log.info(f"[Dataset] Add {n_dump} data to the end to be devided by {n_gpu=}. Total length={len(total_idx)}!")

    # build subset
    indices = idx_per_gpu.tolist()
    subset = Subset(dataset, indices)
    log.info(f"[Dataset] Built subset for gpu={opt.global_rank}! Now size={len(subset)}!")
    return subset

def collect_all_subset(sample, log):
    batch, *xdim = sample.shape
    gathered_samples = dist_util.all_gather(sample, log)
    gathered_samples = [sample.cpu() for sample in gathered_samples]
    # [batch, n_gpu, *xdim] --> [batch*n_gpu, *xdim]
    return torch.stack(gathered_samples, dim=1).reshape(-1, *xdim)

def build_partition(opt, full_dataset, log):
    n_samples = len(full_dataset)

    part_idx, n_part = [int(s) for s in opt.partition.split("_")]
    assert part_idx < n_part and part_idx >= 0
    assert n_samples % n_part == 0

    n_samples_per_part = n_samples // n_part
    start_idx = part_idx * n_samples_per_part
    end_idx = (part_idx+1) * n_samples_per_part

    indices = [i for i in range(start_idx, end_idx)]
    subset = Subset(full_dataset, indices)
    log.info(f"[Dataset] Built partition={opt.partition}, {start_idx=}, {end_idx=}! Now size={len(subset)}!")
    return subset

def build_val_dataset(opt, log, corrupt_type):
    if "sr4x" in corrupt_type:
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log)
    elif "inpaint" in corrupt_type:
        mask = corrupt_type.split("-")[1]
        val_dataset = imagenet.InpaintingVal10kSubset(opt, log, mask) # subset 10k val + mask
    elif corrupt_type == "mixture":
        from corruption.mixture import MixtureCorruptDatasetVal
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log)
        val_dataset = MixtureCorruptDatasetVal(opt, val_dataset) # subset 10k val + mixture
    else:
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log) # subset 10k val

    # build partition
    if opt.partition is not None:
        val_dataset = build_partition(opt, val_dataset, log)
    return val_dataset

def get_recon_imgs_fn(opt, nfe):
    if opt.use_cddb_deep:
        suffix = f"samples_cddb_deep_nfe{nfe}_step{opt.step_size}"
        sample_dir = RESULT_DIR / opt.ckpt / suffix
    elif opt.use_cddb:
        suffix = f"samples_cddb_nfe{nfe}_step{opt.step_size}"
        sample_dir = RESULT_DIR / opt.ckpt / suffix
    else:
        suffix = f"samples_nfe{nfe}_clip_1k"
        sample_dir = RESULT_DIR / opt.ckpt / suffix
    os.makedirs(sample_dir, exist_ok=True)

    recon_imgs_fn = sample_dir / "recon{}.pt".format(
        "" if opt.partition is None else f"_{opt.partition}"
    )
    return recon_imgs_fn, sample_dir

def compute_batch(ckpt_opt, corrupt_type, corrupt_method, out):
    if "inpaint" in corrupt_type:
        clean_img, y, mask = out
        corrupt_img = clean_img * (1. - mask) + mask
        x1          = clean_img * (1. - mask) + mask * torch.randn_like(clean_img)
        x1_pinv = corrupt_img.to(opt.device)
        x1_forw = corrupt_img.to(opt.device)
    elif corrupt_type == "mixture":
        clean_img, corrupt_img, y = out
        mask = None
        x1_pinv = None
        x1_forw = None
    elif "blur" in corrupt_type:
        clean_img, y = out
        mask = None
        corrupt_img_y, corrupt_img_pinv = corrupt_method(clean_img.to(opt.device))
        corrupt_img = corrupt_img_y
        x1 = corrupt_img_y.to(opt.device)
        x1_pinv = corrupt_img_pinv.to(opt.device)
        x1_forw = x1
    else: # sr, jpeg case
        clean_img, y = out
        mask = None
        corrupt_img_pinv, corrupt_img_y = corrupt_method(clean_img.to(opt.device))
        corrupt_img = corrupt_img_pinv
        x1 = corrupt_img.to(opt.device)
        x1_pinv = x1
        x1_forw = corrupt_img_y.to(opt.device)

    cond = x1.detach() if ckpt_opt.cond_x1 else None

    return corrupt_img, x1, mask, cond, y, clean_img, x1_pinv, x1_forw

# @torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")

    # get (default) ckpt option
    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / opt.ckpt)
    corrupt_type = ckpt_opt.corrupt
    nfe = opt.nfe or ckpt_opt.interval-1

    # build corruption method
    corrupt_method = build_corruption(opt, log, corrupt_type=corrupt_type)

    # build imagenet val dataset
    val_dataset = build_val_dataset(opt, log, corrupt_type)
    n_samples = len(val_dataset)

    # build dataset per gpu and loader
    subset_dataset = build_subset_per_gpu(opt, val_dataset, log)
    val_loader = DataLoader(subset_dataset,
        batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
    )

    # build runner
    runner = Runner(ckpt_opt, log, save_opt=False)

    # handle use_fp16 for ema
    if opt.use_fp16:
        runner.ema.copy_to() # copy weight from ema to net
        runner.net.diffusion_model.convert_to_fp16()
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99) # re-init ema with fp16 weight

    # create save folder
    recon_imgs_fn, sample_dir = get_recon_imgs_fn(opt, nfe)
    
    for t in ["input", "recon", "label", "extra"]:
        (sample_dir / t).mkdir(exist_ok=True, parents=True)
    log.info(f"Recon images will be saved to {sample_dir}!")

    recon_imgs = []
    ys = []
    num = 0
    
    log_count = 10
    
    for loader_itr, out in enumerate(val_loader):
        corrupt_img, x1, mask, cond, y, clean_img, x1_pinv, x1_forw = compute_batch(ckpt_opt, corrupt_type, corrupt_method, out)

        if opt.use_cddb_deep:
            sv_idx = str(loader_itr).zfill(3)
            results_dir = sample_dir / f"{sv_idx}"
            # results_dir = None
            xs, pred_x0s = runner.cddb_deep_sampling(
                ckpt_opt, x1, x1_pinv, x1_forw, mask=mask, corrupt_type=corrupt_type, 
                corrupt_method=corrupt_method, cond=cond, clip_denoise=opt.clip_denoise, 
                nfe=nfe, verbose=opt.n_gpu_per_node==1, log_count=log_count, step_size=opt.step_size,
                results_dir=results_dir
            )
        elif opt.use_cddb:
            sv_idx = str(loader_itr).zfill(3)
            results_dir = sample_dir / f"{sv_idx}"
            xs, pred_x0s = runner.cddb_sampling(
                ckpt_opt, x1, x1_pinv, x1_forw, mask=mask, corrupt_type=corrupt_type, 
                corrupt_method=corrupt_method, cond=cond, clip_denoise=opt.clip_denoise, 
                nfe=nfe, verbose=opt.n_gpu_per_node==1, log_count=log_count, step_size=opt.step_size,
                results_dir=results_dir,
            )
        else:
            xs, pred_x0s = runner.ddpm_sampling(
                ckpt_opt, x1, x1_pinv, x1_forw, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, 
                nfe=nfe, verbose=opt.n_gpu_per_node==1, log_count=log_count
            )
        recon_img = xs[:, 0, ...].to(opt.device)

        assert recon_img.shape == corrupt_img.shape

        if loader_itr == 0 and opt.global_rank == 0: # debug
            os.makedirs(".debug", exist_ok=True)
            tu.save_image((corrupt_img+1)/2, ".debug/corrupt.png")
            tu.save_image((recon_img+1)/2, ".debug/recon.png")
            log.info("Saved debug images!")

        # [-1,1]
        gathered_recon_img = collect_all_subset(recon_img, log)
        recon_imgs.append(gathered_recon_img)

        y = y.to(opt.device)
        gathered_y = collect_all_subset(y, log)
        ys.append(gathered_y)

        num += len(gathered_recon_img)
        log.info(f"Collected {num} recon images!")
        
        # save input, recon, label also as image files
        for idx in range(len(corrupt_img)):
            sv_idx = str(opt.batch_size * loader_itr + idx).zfill(3)
    
            input_idx = (corrupt_img[idx:idx+1, ...] + 1) / 2
            recon_idx = (recon_img[idx:idx+1, ...] + 1) / 2
            label_idx = (clean_img[idx:idx+1, ...] + 1) / 2
            tu.save_image(input_idx, str(sample_dir / f"input" / f"{sv_idx}.png"))
            tu.save_image(recon_idx, str(sample_dir / f"recon" / f"{sv_idx}.png"))
            tu.save_image(label_idx, str(sample_dir / f"label" / f"{sv_idx}.png"))  
        
        dist.barrier()

    del runner

    arr = torch.cat(recon_imgs, axis=0)[:n_samples]
    label_arr = torch.cat(ys, axis=0)[:n_samples]

    if opt.global_rank == 0:
        torch.save({"arr": arr, "label_arr": label_arr}, recon_imgs_fn)
        log.info(f"Save at {recon_imgs_fn}")
    dist.barrier()

    log.info(f"Sampling complete! Collect recon_imgs={arr.shape}, ys={label_arr.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--n-gpu-per-node", type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,  default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,  default=1,           help="The number of nodes in multi node env")

    # data
    parser.add_argument("--image-size",     type=int,  default=256)
    parser.add_argument("--dataset-dir",    type=Path, default="/dataset",  help="path to LMDB dataset")
    parser.add_argument("--partition",      type=str,  default=None,        help="e.g., '0_4' means the first 25% of the dataset")
    parser.add_argument("--add-noise",      action="store_true",            help="If true, add small gaussian noise to y")

    # sample
    parser.add_argument("--batch-size",     type=int,  default=32)
    parser.add_argument("--ckpt",           type=str,  default=None,        help="the checkpoint name from which we wish to sample")
    parser.add_argument("--nfe",            type=int,  default=None,        help="sampling steps")
    parser.add_argument("--clip-denoise",   action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16",       action="store_true",            help="use fp16 network weight for faster sampling")
    parser.add_argument("--eta",            type=float, default=1.0,        help="ddim stochasticity. 1.0 recovers ddpm")
    parser.add_argument("--use-cddb-deep",  action="store_true",            help="use cddb-deep")
    parser.add_argument("--use-cddb",       action="store_true",            help="use cddb")
    parser.add_argument("--step-size",      type=float, default=1.0,        help="step size for gradient descent")
    parser.add_argument("--prob_mask",      type=float, default=0.35,       help="probability of masking")
    

    arg = parser.parse_args()

    opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device="cuda",
    )
    opt.update(vars(arg))

    # one-time download: ADM checkpoint
    download_ckpt("data/")

    set_seed(opt.seed)

    if opt.distributed:
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.n_gpu_per_node
            global_size = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
            p = Process(target=dist_util.init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        dist_util.init_processes(0, opt.n_gpu_per_node, main, opt)
