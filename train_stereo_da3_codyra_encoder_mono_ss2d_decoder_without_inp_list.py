from __future__ import print_function, division

import argparse
import logging
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from core.raft_stereo_da3_codyra_encoder_mono_ss2d_decoder_without_inp_list import RAFTStereo

from evaluate_stereo_da3_codyra_encoder_mono_ss2d_decoder_without_inp_list import *
import core.stereo_datasets as datasets

try:
    from torch.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self, device="cuda", enabled=True):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict):
            pass


def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def save_checkpoint(path, model, optimizer, scheduler, scaler, total_steps, global_batch_num):
    """Save complete training state for resuming"""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'total_steps': total_steps,
        'global_batch_num': global_batch_num,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    """Load checkpoint with backward compatibility

    Returns:
        tuple: (total_steps, global_batch_num) or (0, 0) if loading old format
    """
    logging.info(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path)

    # Check if this is a new format checkpoint (dict with 'model' key)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        # New format with complete training state
        model.load_state_dict(checkpoint['model'], strict=True)

        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info("Restored optimizer state")

        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            logging.info("Restored scheduler state")

        if scaler is not None and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
            logging.info("Restored scaler state")

        total_steps = checkpoint.get('total_steps', 0)
        global_batch_num = checkpoint.get('global_batch_num', 0)
        logging.info(f"Resuming from step {total_steps}, batch {global_batch_num}")

        return total_steps, global_batch_num
    else:
        # Old format - just model weights
        model.load_state_dict(checkpoint, strict=True)
        logging.info("Loaded old format checkpoint (model weights only)")
        return 0, 0


class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler, log_dir):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.log_dir = log_dir
        tensorboard_dir = Path(log_dir) / 'tensorboard'
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tensorboard_dir))

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            tensorboard_dir = Path(self.log_dir) / 'tensorboard'
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tensorboard_dir))

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            tensorboard_dir = Path(self.log_dir) / 'tensorboard'
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tensorboard_dir))

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):
    # Create timestamped log directory
    timestamp = time.strftime('%Y-%m-%d_%Hh%Mm%Ss', time.localtime())
    log_dir = Path('train_log') / f'{timestamp}_{args.name}'
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = log_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Log directory: {log_dir}")
    logging.info(f"Checkpoint directory: {checkpoint_dir}")

    model = nn.DataParallel(RAFTStereo(args))

    # Print parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total params: {total_params/1e6:.2f}M")
    logging.info(f"Trainable params: {trainable_params/1e6:.2f}M")

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    scaler = GradScaler("cuda", enabled=args.mixed_precision)

    total_steps = 0
    global_batch_num = 0

    # Load checkpoint if provided
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        total_steps, global_batch_num = load_checkpoint(
            args.restore_ckpt, model, optimizer, scheduler, scaler
        )

    logger = Logger(model, scheduler, str(log_dir))
    logger.total_steps = total_steps  # Sync logger with resumed step count

    model.cuda()
    model.train()
    model.module.freeze_bn()  # We keep BatchNorm frozen

    should_keep_training = True

    while should_keep_training:

        pbar = tqdm(train_loader, desc="Train", leave=False)
        for i_batch, (_, *data_blob) in enumerate(pbar):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            assert model.training
            flow_predictions = model(image1, image2, iters=args.train_iters)
            assert model.training

            # Task loss
            task_loss, metrics = sequence_loss(flow_predictions, flow, valid)

            # CoDyRA: Add sparsity loss in sparse phase
            if total_steps >= args.num_steps // 2:
                if total_steps == args.num_steps // 2:
                    model.module.set_sparse_mode(True)
                    logging.info(f"=== Sparse phase started at step {total_steps} ===")
                sparsity_loss = model.module.compute_sparsity_loss()
                loss = task_loss + sparsity_loss
                logger.writer.add_scalar("sparsity_loss", sparsity_loss.item(), global_batch_num)
            else:
                loss = task_loss

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            # CoDyRA: Update importance weights in sparse phase
            if total_steps >= args.num_steps // 2:
                kappa = args.max_kappa * min((total_steps - args.num_steps // 2) / (args.num_steps // 2), 1)
                model.module.update_iws(kappa)

            logger.push(metrics)

            if total_steps % args.validation_frequency == args.validation_frequency - 1:
                save_path = checkpoint_dir / f'{total_steps + 1}_{args.name}.pth'
                logging.info(f"Saving file {save_path.absolute()}")
                save_checkpoint(save_path, model, optimizer, scheduler, scaler, total_steps, global_batch_num)

                results = validate_tartanair_occ(model.module, iters=args.valid_iters, save_dir=str(log_dir))

                logger.write_dict(results)

                model.train()
                model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 10000:
            save_path = checkpoint_dir / f'{total_steps + 1}_epoch_{args.name}.pth'
            logging.info(f"Saving file {save_path}")
            save_checkpoint(save_path, model, optimizer, scheduler, scaler, total_steps, global_batch_num)

    print("FINISHED TRAINING")
    logger.close()
    PATH = checkpoint_dir / f'{args.name}.pth'
    save_checkpoint(PATH, model, optimizer, scheduler, scaler, total_steps, global_batch_num)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--da3_pretrained', default='depthanything/checkpoints/DA3-BASE/model.safetensors',
                        help="DA3 pretrained weights")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720],
                        help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=16,
                        help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32,
                        help='number of flow-field updates during validation forward pass')
    parser.add_argument('--validation_frequency', type=int, default=10000,
                        help='validation and checkpoint saving frequency (in training steps)')

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg",
                        help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true',
                        help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'],
                        help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'],
                        help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')

    # CoDyRA parameters
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--max_kappa', type=float, default=0.005, help='ISTA threshold for sparsification')
    parser.add_argument('--lambda_reg', type=float, default=1e-4, help='Sparsity penalty weight')

    # MDE initialization parameters
    parser.add_argument('--use_mde_init', action='store_true', help='Use MDE (mono depth) to initialize flow')
    parser.add_argument('--baseline', type=float, default=0.25, help='Stereo baseline (TartanAir: 0.25)')
    parser.add_argument('--focal', type=float, default=320.0, help='Focal length (TartanAir: 320.0)')

    # SS2D scan direction (for ablation study)
    parser.add_argument('--scan_mode', type=str, default='cross2d', choices=['cross2d', 'bidi', 'unidi'],
                        help='SS2D scan direction mode: cross2d (4-dir), bidi (2-dir horizontal), unidi (1-dir)')

    # SS2D hyperparameters (for ablation study)
    parser.add_argument('--d_state', type=int, default=16, help='SSM state dimension (default: 16)')
    parser.add_argument('--ssm_ratio', type=float, default=1.0, help='SSM expansion ratio (default: 1.0)')

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    train(args)