# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from time import time
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn import L1Loss

from models.SimMIM_swin_FD import SimMIMSwinFD
from models.SimMIM_swin import SimMIMSwin
from models.SimMIM import SimMIMViT

from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import *
from utils.ops import *
from utils.utils import AverageMeter, distributed_all_gather
import torch.multiprocessing
import openpyxl
from openpyxl import Workbook
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
print('Setting resource limit:', str(resource.getrlimit(resource.RLIMIT_NOFILE)))


def scale_tensor(tensor, scale):
    if tensor.is_complex():
        return torch.complex(tensor.real * scale, tensor.imag * scale)
    else:
        return tensor * scale


def unscale_tensor(tensor, scale):
    if tensor.is_complex():
        return torch.complex(tensor.real / scale, tensor.imag / scale)
    else:
        return tensor / scale


def main():
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def calculate_ssim_and_psnr(reconstructed, original):
        ssim_value = ssim(reconstructed, original, data_range=original.max() - original.min())
        psnr_value = psnr(reconstructed, original, data_range=original.max() - original.min())
        return ssim_value, psnr_value

    def validate(val_loader, model, global_step):
        model.eval()
        total_ssim = 0
        total_psnr = 0
        total_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if batch is None or batch['image'] is None:
                    continue

                img = batch['image'].cuda()

                loss_val, pred_pixel_val, mask_val = model(img)
                output = model.unpatchify(pred_pixel_val)
                output = output.permute(0, 1, 3, 4, 2)
                img = img.squeeze(1).cpu().numpy()
                output = output.squeeze(1).cpu().numpy()

                for i in range(img.shape[0]):
                    ssim_val, psnr_val = calculate_ssim_and_psnr(output[i], img[i])
                    total_ssim += ssim_val
                    total_psnr += psnr_val
                total_batches += img.shape[0]

        # mean SSIM, PSNR
        avg_ssim = total_ssim / total_batches
        avg_psnr = total_psnr / total_batches

        # Excel
        wb = openpyxl.load_workbook(val_excel_path)
        ws = wb.active
        ws.append([global_step, avg_ssim, avg_psnr])
        wb.save(val_excel_path)

    def train(args, global_step, train_loader, val_best, scaler):
        model.train()
        loss_train = []
        run_loss = AverageMeter()

        for step, batch in enumerate(train_loader):
            if batch is None or batch['image'] is None:
                continue
            t1 = time()
            img = batch['image']
            img = img.cuda()

            # with torch.no_grad():
            with autocast(enabled=args.amp):
                loss, pred_pixel, mask = model(img)

            loss_train.append(loss.item())
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.lrdecay:
                scheduler.step()

            optimizer.zero_grad()

            run_loss.update(loss.item(), n=args.batch_size)

            lr = optimizer.param_groups[0]["lr"]

            if args.distributed:
                if dist.get_rank() == 0:
                    print("Step:{}/{}, Loss:{:.4f}, Time:{:.4f}".format
                          (global_step, args.num_steps, loss, time() - t1))
            else:
                print("Step:{}/{}, Loss:{:.4f}, "
                      "lr:{:.8f}, Time:{:.4f}".format(global_step, args.num_steps,
                                                      run_loss.avg,
                                                      lr, time() - t1))

            global_step += 1
            if args.distributed:
                val_cond = (dist.get_rank() == 0) and (global_step % args.eval_num == 0)
            else:
                val_cond = global_step % args.eval_num == 0

            freq = 20000
            val_freq = global_step % freq == 0
            if val_cond:
                checkpoint = {
                    "global_step": global_step,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if args.lrdecay else None,
                }
                print("save new model_current_epoch.pt")
                save_ckp(checkpoint, logdir + "/model_current_epoch.pt")

            if val_freq:
                validate(val_loader, model, global_step)

                checkpoint = {
                    "global_step": global_step,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if args.lrdecay else None,
                }
                save_ckp(checkpoint, logdir + "/model_step" + str(global_step) + ".pt")

        return global_step, loss, val_best


    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--logdir", default="logs_SimMIMSwin", type=str, help="directory to save logs")
    parser.add_argument("--epochs", default=1000, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=944000, type=int, help="number of training iterations")
    parser.add_argument("--eval_num", default=50, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=10000, type=int, help="warmup steps")  #3000
    # model_dict
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--pretrained", default=None, help="pretrain")
    parser.add_argument("--img_size", default=[96, 96, 96], help="Inputs to ViT/swin")
    parser.add_argument("--patch_size", default=[4, 4, 4], help="patch size of ViT/swin")
    parser.add_argument("--embed_dim", default=96, type=int, help="embed dim of swin")
    parser.add_argument("--hidden_size", default=768, type=int, help="hidden size of ViT")
    parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp of ViT")
    parser.add_argument("--num_layers", default=12, type=int, help="num layers of ViT")  #12
    parser.add_argument("--num_heads", default=12, type=int, help="num heads of ViT")
    parser.add_argument("--pos_embed", default='conv', help="pos_embed of ViT")
    parser.add_argument("--dropout_rate", default=0, type=int, help="drop rate of ViT")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--masking_ratio", default=0.75, type=int, help="mask ratio of ViT/swin")
    parser.add_argument("--revise_keys", default=[], help="mask ratio of ViT")

    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
    parser.add_argument("--batch_size", default=16, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--lr", default=0.0002, type=float, help="learning rate")
    parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lrdecay", default=True, help="enable learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
    parser.add_argument("--loss_type", default="SSL", type=str)
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--resume", default=None, type=str, help="resume training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
    parser.add_argument("--noamp", default=True, help="do NOT use amp for training")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--smartcache_dataset", default=False, help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", default=True, help="use monai cache Dataset")
    # losses
    parser.add_argument('--loss_pixel_w', default=1.0, type=float, help='weight for pixel level loss (MSE loss)')
    parser.add_argument('--loss_frequency_w', default=1.0, type=float, help='weight for focal frequency loss')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='the scaling factor alpha of the spectrum weight matrix for flexibility')
    parser.add_argument('--ave_spectrum', action='store_true', help='whether to use minibatch average spectrum')
    parser.add_argument('--log_matrix', action='store_true',
                        help='whether to adjust the spectrum weight matrix by logarithm')
    parser.add_argument('--batch_matrix', action='store_true',
                        help='whether to calculate the spectrum weight matrix using batch-based statistics')


    args = parser.parse_args()
    logdir = args.logdir

    torch.cuda.set_device(0)

    args.amp = True
    torch.backends.cudnn.benchmark = True
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    model = SimMIMSwin(**vars(args))
    model.cuda()

    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, amsgrad=True)

    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

        elif args.lr_schedule == "poly":

            def lambdas(epoch):
                return (1 - float(epoch) / float(args.epochs)) ** 0.9

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    global_step = 0
    if args.resume:
        print('resume from previous checkpoints')
        model_pth = args.resume
        model_dict = torch.load(model_pth)
        model.load_state_dict(model_dict["state_dict"], strict=False)
        global_step = model_dict["global_step"]
        optimizer.load_state_dict(model_dict["optimizer"])
        if args.lrdecay and "scheduler" in model_dict and model_dict["scheduler"] is not None:
            scheduler.load_state_dict(model_dict["scheduler"])

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.local_rank])

    train_loader, val_loader = get_loader(args)

    best_val = 1e8
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None


    val_excel_path = "val.xlsx"
    if not os.path.exists(val_excel_path):
        wb = Workbook()
        ws = wb.active
        ws.append(["Global Step", "Average SSIM", "Average PSNR"])
        wb.save(val_excel_path)

    while global_step < args.num_steps:
        global_step, loss, best_val = train(args, global_step, train_loader, best_val, scaler)
    checkpoint = {"epoch": args.epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}

    if args.distributed:
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), logdir + "final_model.pth")
        dist.destroy_process_group()
    else:
        torch.save(model.state_dict(), logdir + "final_model.pth")  # Only the parameter
    save_ckp(checkpoint, logdir + "/model_final_epoch.pt")  # all, finally


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


if __name__ == "__main__":
    main()