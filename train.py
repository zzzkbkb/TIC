import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=./pretrained/tic_light/3/runs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import image_models

# λ ∈ {0.016, 0.018, 0.020, 0.022, 0.024}
# clip_max_norm ∈ {0, 2.0}

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        num_pixels = N * H * W
        out = {}

        # -------------------------------------------------
        # BPP (total + per latent)
        # -------------------------------------------------
        bpp_total = 0.0
        for k, likelihoods in output["likelihoods"].items():
            bpp = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            out[f"bpp_{k}"] = bpp
            bpp_total = bpp_total + bpp

        out["bpp_loss"] = bpp_total

        # -------------------------------------------------
        # Reconstruction
        # -------------------------------------------------
        x_hat = output["x_hat"]

        # 对齐空间尺寸（必须）
        if x_hat.shape[-2:] != target.shape[-2:]:
            x_hat = F.interpolate(
                x_hat,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # MSE loss
        out["mse_loss"] = self.mse(x_hat, target)

        # -------------------------------------------------
        # High-frequency losses
        # -------------------------------------------------
        gx_hat, gy_hat = self.gradient(x_hat)
        gx, gy = self.gradient(target)
        # grad_loss = F.l1_loss(gx_hat, gx) + F.l1_loss(gy_hat, gy)
        #
        # lap_loss = F.l1_loss(
        #     self.laplacian(x_hat),
        #     self.laplacian(target)
        # )

        # out["grad_loss"] = grad_loss  # Gradient Loss梯度损失，图像的一阶导数
        # out["lap_loss"] = lap_loss  # Laplacian Loss 拉普拉斯损失，图像的二阶导数

        # -------------------------------------------------
        # Total loss
        # -------------------------------------------------
        out["loss"] = (
                out["bpp_loss"]
                + self.lmbda * 255 ** 2 * out["mse_loss"]
                # + 0.1 * grad_loss
                # + 0.05 * lap_loss
        )

        return out

    def gradient(self, x):
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return gx, gy

    def laplacian(self, x):
        lap = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]],
            device=x.device,
            dtype=x.dtype,
        ).view(1, 1, 3, 3)
        lap = lap.repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, lap, padding=1, groups=x.size(1))


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def init(args):
    base_dir = f'./pretrained/{args.model}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)

    return base_dir


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    # Clear existing handlers to avoid duplication
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    # Use append mode ('a') to add new logs to existing file
    log_file_handler = logging.FileHandler(log_dir, mode='a', encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
        model, criterion, train_dataloader, optimizer, aux_optimizer,
        epoch, clip_max_norm, writer
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad(set_to_none=True)
        aux_optimizer.zero_grad(set_to_none=True)

        # ---- student forward ----
        out_net = model(d)

        # ---- RD loss ----
        out_criterion = criterion(out_net, d)
        total_loss = out_criterion["loss"]

        # ---- backward main ----
        total_loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        # ---- backward aux (entropy bottleneck quantiles) ----
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        # # 每 100 个 batch 打印一次 (16 * 100 = 1600 张图片)
        # if i % 100 == 0:
        #     # eff_lambda = criterion.lmbda * (current_gain ** 4) if current_gain else criterion.lmbda
        #     # 计算当前处理的图片总数
        #     logging.info(
        #         f'[{i * len(d)}/{len(train_dataloader.dataset)}] | '
        #         f'Loss: {total_loss.item():.3f} | '
        #         f'MSE: {out_criterion["mse_loss"].item():.5f} | '
        #         f'Bpp: {out_criterion["bpp_loss"].item():.4f} | '
        #         f'Aux: {aux_loss.item():.2f}'
        #     )
        # 每 100 个 batch 打印一次 (16 * 100 = 1600 张图片)
        if i % 100 == 0:
            # global_step = epoch * len(train_dataloader) + i # 原始的 step 计数
            writer.add_scalar('Loss/total_loss', total_loss.item(), epoch) # 使用 epoch 作为 global_step
            writer.add_scalar('Loss/mse_loss', out_criterion["mse_loss"].item(), epoch) # 使用 epoch 作为 global_step
            writer.add_scalar('Loss/bpp_loss', out_criterion["bpp_loss"].item(), epoch) # 使用 epoch 作为 global_step
            writer.add_scalar('Loss/aux_loss', aux_loss.item(), epoch) # 使用 epoch 作为 global_step
            if "bpp_y" in out_criterion:
                writer.add_scalar('Loss/bpp_y', out_criterion["bpp_y"].item(), epoch) # 使用 epoch 作为 global_step
            if "bpp_z" in out_criterion:
                writer.add_scalar('Loss/bpp_z', out_criterion["bpp_z"].item(), epoch) # 使用 epoch 作为 global_step
            if "bpp_hf" in out_criterion:
                writer.add_scalar('Loss/bpp_hf', out_criterion["bpp_hf"].item(), epoch) # 使用 epoch 作为 global_step
            
            # 记录 final_spatial_attn_scale
            if hasattr(model, 'final_spatial_attn_scale'):
                writer.add_scalar('Attention/final_spatial_attn_scale', model.final_spatial_attn_scale.item(), epoch)

            log_str = (
                f'[{i * len(d)}/{len(train_dataloader.dataset)}] | '
                f'Loss: {total_loss.item():.3f} | '
                f'MSE: {out_criterion["mse_loss"].item():.5f} | '
                f'Bpp: {out_criterion["bpp_loss"].item():.4f}'
            )

            # ---- 拆分 bpp ----
            if "bpp_y" in out_criterion:
                log_str += f' | Bpp_y: {out_criterion["bpp_y"].item():.4f}'
            if "bpp_z" in out_criterion:
                log_str += f' | Bpp_z: {out_criterion["bpp_z"].item():.4f}'
            if "bpp_hf" in out_criterion:
                log_str += f' | Bpp_hf: {out_criterion["bpp_hf"].item():.4f}'

            log_str += f' | Aux: {aux_loss.item():.2f}'

            logging.info(log_str)


def validate_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr = -10 * math.log10(mse_loss.avg) if mse_loss.avg > 0 else 0

    logging.info(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.3f} | "
        f"MSE loss: {mse_loss.avg:.5f} | "
        f"PSNR: {psnr:.5f} | "
        f"BPP: {bpp_loss.avg:.4f} | "
        f"Aux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg


def save_checkpoint(state, is_best, base_dir, log_dir, filename="checkpoint.pth.tar"):
    # Save current checkpoint (overwrite each epoch)
    state["log_dir"] = log_dir # 保存 TensorBoard 的 log_dir
    torch.save(state, base_dir + filename)
    # Save best checkpoint if current is best
    if is_best:
        shutil.copyfile(base_dir + filename, base_dir + "checkpoint_best_loss.pth.tar")


def save_separate_models(net, base_dir, epoch, is_best):
    """Save separate compression (edge) and decompression (server) state_dicts.
    Works with the modified TIC (no context, sequential g_a/h_a, lightweight h_s blocks).
    """
    model = net.module if hasattr(net, "module") else net
    sd = model.state_dict()

    # --------- what edge(compress) needs ----------
    # g_a, h_a, h_s (hyper-synthesis for gaussian params), entropy_bottleneck, gaussian_conditional
    compress_prefixes = (
        "g_a.",
        "h_a.",
        "h_s_backbone.",
        "h_s_y_head.",
        "entropy_bottleneck.",
        "gaussian_conditional.",
        "hf_enc.",
        "hf_context.",
        "hf_gaussian.",
    )

    # --------- what server(decompress) needs ----------
    # g_s, h_s blocks, entropy_bottleneck, gaussian_conditional
    decompress_prefixes = (
        "g_s0.", "g_s1.", "g_s2.", "g_s3.", "g_s4.", "g_s5.",
        "g_s6_feat.",
        "h_s_backbone.",
        "h_s_y_head.",
        "entropy_bottleneck.",
        "gaussian_conditional.",
        "hf_dec.",
        "hf_gaussian.",
        "main_refine.",
        "feature_fusion_attn.",
        "feature_fusion.",
        "to_rgb.",
        "final_spatial_attn.",
    )

    extra_compress_keys = (
        "hf_fusion_scale",
    )

    extra_decompress_keys = (
        "hf_fusion_scale",
        "final_spatial_attn_scale",
    )

    def filter_state_dict(prefixes, extra_keys):
        out = {}
        for k, v in sd.items():
            if k in extra_keys or any(k.startswith(p) for p in prefixes):
                out[k] = v
        return out

    compression_state_dict = {
        "epoch": epoch,
        "N": getattr(model, "N", None),
        "M": getattr(model, "M", None),
        "state_dict": filter_state_dict(compress_prefixes, extra_compress_keys),
    }

    decompression_state_dict = {
        "epoch": epoch,
        "N": getattr(model, "N", None),
        "M": getattr(model, "M", None),
        "state_dict": filter_state_dict(decompress_prefixes, extra_decompress_keys),
    }

    # Save current models (overwrite each epoch)
    comp_path = os.path.join(base_dir, "compression_model.pth.tar")
    decomp_path = os.path.join(base_dir, "decompression_model.pth.tar")

    torch.save(compression_state_dict, comp_path)
    torch.save(decompression_state_dict, decomp_path)

    # Save best models if current is best
    if is_best:
        best_comp_path = os.path.join(base_dir, "compression_model_best.pth.tar")
        best_decomp_path = os.path.join(base_dir, "decompression_model_best.pth.tar")
        shutil.copyfile(comp_path, best_comp_path)
        shutil.copyfile(decomp_path, best_decomp_path)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="tic_light",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="/mnt/d/Downloads/flicker_2W_images", help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=1000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quality-level",
        type=int,
        default=3,
        help="Quality level (default: %(default)s)",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=64,
        help="Channel N for tic_light",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=128,
        help="Channel M for tic_light",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=2e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type= float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
    parser.add_argument(
        "--gpu-id",
        type=str,
        default=0,
        help="GPU ids (default: %(default)s)",
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip-max-norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        '--name',
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'),
        type=str,
        help='Result dir name',
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    base_dir = init(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    setup_logger(base_dir + '/training.log')
    # ============================================================
    # TensorBoard writer setup
    # ============================================================
    writer = None
    if args.checkpoint:
        # If resuming from checkpoint, try to load the previous log_dir
        try:
            checkpoint_data = torch.load(args.checkpoint, map_location="cpu")
            if "log_dir" in checkpoint_data:
                writer = SummaryWriter(log_dir=checkpoint_data["log_dir"])
                logging.info(f"Resuming TensorBoard logs to: {checkpoint_data['log_dir']}")
            else:
                logging.warning("Checkpoint does not contain 'log_dir'. Creating new TensorBoard log directory.")
        except Exception as e:
            logging.error(f"Error loading log_dir from checkpoint: {e}. Creating new TensorBoard log directory.")
            
    if writer is None: # If not resumed or failed to resume, create a new one
        log_dir = os.path.join(base_dir, 'runs', args.name)
        writer = SummaryWriter(log_dir=log_dir)
        logging.info(f"Creating new TensorBoard log directory: {log_dir}")

    msg = f'======================= {args.name} ======================='
    logging.info(msg)
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))
    logging.info('=' * len(msg))

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    extra_kwargs = {}
    if args.model == "tic_light":
        extra_kwargs["N"] = args.N
        extra_kwargs["M"] = args.M
    net = image_models[args.model](quality=int(args.quality_level), **extra_kwargs)
    net = net.to(device)
    if hasattr(net, "set_pooling"):
        net.set_pooling(pool_kernel=1, pool_stride=1)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15, threshold=1e-4, min_lr=5e-6)   # 把MultiStepLR改成了ReduceLROnPlateau
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        # logging.info("Loading", args.checkpoint)
        logging.info("Loading checkpoint: %s", args.checkpoint)

        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])   #微调时候修改的，正常需要加上

        # ★★★ 关键：强制覆盖 learning rate（finetune） ★★★
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        logging.info('======Current epoch %s ======' % epoch)
        logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            writer
        )

        cur_loss = validate_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(cur_loss)
        is_best = cur_loss < best_loss
        best_loss = min(cur_loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),  # 模型的状态字典，包含模型的所有参数（权重和偏置）
                    "loss": cur_loss,
                    "optimizer": optimizer.state_dict(),  # 主优化器的状态字典（Adam优化器的动量、方差等）
                    "aux_optimizer": aux_optimizer.state_dict(),  # 辅助优化器的状态字典（用于优化量化参数等）
                    "lr_scheduler": lr_scheduler.state_dict(),  # 学习率调度器的状态字典（当前的学习率等）
                },
                is_best,
                base_dir,
                writer.log_dir # 传入 log_dir
            )
            # Save separate compression and decompression models
            save_separate_models(net, base_dir, epoch, is_best)

            # also save raw state_dict copies
            state_dict = net.state_dict()
            full_latest_path = os.path.join(base_dir, "tic_light_full_latest.pth")
            torch.save(state_dict, full_latest_path)
            if is_best:
                full_best_path = os.path.join(base_dir, "tic_light_full_best.pth")
                torch.save(state_dict, full_best_path)
    writer.close()


if __name__ == "__main__":
    main(sys.argv[1:])
