from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from raft_stereo_da3_codyra_encoder_mono_ss2d_decoder_without_inp_list import RAFTStereo, autocast
import stereo_datasets as datasets
from utils.utils import InputPadder

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(device_type='cuda', enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {'eth3d-epe': epe, 'eth3d-d1': d1}


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(device_type='cuda', enabled=mixed_prec):
            start = time.time()
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1)%10 == 0:
            logging.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-epe': epe, 'kitti-d1': d1}


@torch.no_grad()
def validate_things(model, iters=32, mixed_prec=False):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(device_type='cuda', enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        out = (epe > 1.0)
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation FlyingThings: %f, %f" % (epe, d1))
    return {'things-epe': epe, 'things-d1': d1}


@torch.no_grad()
def validate_tartanair_occ(model, iters=32, mixed_prec=False, use_depth_metrics=True, save_results=True, result_name='', save_dir=None):
    """Validate on TartanAir-Occ test set"""
    from core.depth_metrics import compute_depth_metrics, disparity_to_depth
    from datetime import datetime
    import os
    from pathlib import Path

    model.eval()
    val_dataset = datasets.TartanAir(split='test')

    # Collect all error values for detailed statistics
    all_errors = []
    depth_metrics_all = {'abs_rel': [], 'sq_rel': [], 'rmse': [], 'rmse_log': [], 'a1': [], 'a2': [], 'a3': []}

    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=14)
        image1, image2 = padder.pad(image1, image2)

        with autocast(device_type='cuda', enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)

        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        # Calculate per-pixel errors
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
        val = valid_gt.flatten() >= 0.5
        errors = epe.flatten()[val].cpu().numpy()
        all_errors.extend(errors)

        # Depth metrics (TartanAir: baseline=0.25m, focal=320px)
        if use_depth_metrics:
            pred_depth = disparity_to_depth(flow_pr.abs().unsqueeze(0), baseline=0.25, focal=320.0)
            gt_depth = disparity_to_depth(flow_gt.abs().unsqueeze(0), baseline=0.25, focal=320.0)
            metrics = compute_depth_metrics(pred_depth, gt_depth)
            for k in depth_metrics_all:
                depth_metrics_all[k].append(metrics[k])

    all_errors = np.array(all_errors)

    # Disparity metrics
    result = {
        'tartanair_occ-AvgErr': np.mean(all_errors),
        'tartanair_occ-MedErr': np.median(all_errors),
        'tartanair_occ-bad0.5': 100 * np.mean(all_errors > 0.5),
        'tartanair_occ-bad1.0': 100 * np.mean(all_errors > 1.0),
        'tartanair_occ-bad2.0': 100 * np.mean(all_errors > 2.0),
        'tartanair_occ-bad4.0': 100 * np.mean(all_errors > 4.0),
    }

    # Depth metrics
    if use_depth_metrics:
        for k in depth_metrics_all:
            result[f'tartanair_occ-{k}'] = np.mean(depth_metrics_all[k])

    # Print results
    print("\n" + "="*60)
    print("TartanAir-Occ Evaluation Results")
    print("="*60)
    print("\nDisparity Metrics:")
    print(f"  AvgErr:     {result['tartanair_occ-AvgErr']:.4f} px")
    print(f"  MedErr:     {result['tartanair_occ-MedErr']:.4f} px")
    print(f"  bad 0.5:    {result['tartanair_occ-bad0.5']:.2f}%")
    print(f"  bad 1.0:    {result['tartanair_occ-bad1.0']:.2f}%")
    print(f"  bad 2.0:    {result['tartanair_occ-bad2.0']:.2f}%")
    print(f"  bad 4.0:    {result['tartanair_occ-bad4.0']:.2f}%")

    if use_depth_metrics:
        print("\nDepth Metrics:")
        print(f"  abs_rel:    {result['tartanair_occ-abs_rel']:.4f}")
        print(f"  sq_rel:     {result['tartanair_occ-sq_rel']:.4f}")
        print(f"  rmse:       {result['tartanair_occ-rmse']:.4f} m")
        print(f"  rmse_log:   {result['tartanair_occ-rmse_log']:.4f}")
        print(f"  a1 (δ<1.25): {result['tartanair_occ-a1']:.4f} ({result['tartanair_occ-a1']*100:.2f}%)")
        print(f"  a2 (δ<1.56): {result['tartanair_occ-a2']:.4f} ({result['tartanair_occ-a2']*100:.2f}%)")
        print(f"  a3 (δ<1.95): {result['tartanair_occ-a3']:.4f} ({result['tartanair_occ-a3']*100:.2f}%)")
    print("="*60)

    # Save results to file
    if save_results:
        if save_dir is not None:
            eval_dir = Path(save_dir) / 'eval_results'
        else:
            eval_dir = Path('eval_results')
        eval_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = eval_dir / f"{timestamp}_{result_name}.txt" if result_name else eval_dir / f"{timestamp}_tartanair_occ.txt"
        filename = str(filename)

        with open(filename, 'w') as f:
            f.write("="*60 + "\n")
            f.write("TartanAir-Occ Evaluation Results\n")
            f.write("="*60 + "\n")
            f.write(f"\nTimestamp: {timestamp}\n")
            f.write(f"Test samples: {len(val_dataset)}\n")
            f.write(f"Iterations: {iters}\n")

            f.write("\n" + "-"*60 + "\n")
            f.write("Disparity Metrics\n")
            f.write("-"*60 + "\n")
            f.write(f"AvgErr:     {result['tartanair_occ-AvgErr']:.4f} px\n")
            f.write(f"MedErr:     {result['tartanair_occ-MedErr']:.4f} px\n")
            f.write(f"bad 0.5:    {result['tartanair_occ-bad0.5']:.2f}%\n")
            f.write(f"bad 1.0:    {result['tartanair_occ-bad1.0']:.2f}%\n")
            f.write(f"bad 2.0:    {result['tartanair_occ-bad2.0']:.2f}%\n")
            f.write(f"bad 4.0:    {result['tartanair_occ-bad4.0']:.2f}%\n")

            if use_depth_metrics:
                f.write("\n" + "-"*60 + "\n")
                f.write("Depth Metrics\n")
                f.write("-"*60 + "\n")
                f.write(f"abs_rel:    {result['tartanair_occ-abs_rel']:.4f}\n")
                f.write(f"sq_rel:     {result['tartanair_occ-sq_rel']:.4f}\n")
                f.write(f"rmse:       {result['tartanair_occ-rmse']:.4f} m\n")
                f.write(f"rmse_log:   {result['tartanair_occ-rmse_log']:.4f}\n")
                f.write(f"a1 (δ<1.25): {result['tartanair_occ-a1']:.4f} ({result['tartanair_occ-a1']*100:.2f}%)\n")
                f.write(f"a2 (δ<1.56): {result['tartanair_occ-a2']:.4f} ({result['tartanair_occ-a2']*100:.2f}%)\n")
                f.write(f"a3 (δ<1.95): {result['tartanair_occ-a3']:.4f} ({result['tartanair_occ-a3']*100:.2f}%)\n")

            f.write("\n" + "="*60 + "\n")

        print(f"\n✅ Results saved to: {filename}\n")

    return result


@torch.no_grad()
def validate_middlebury(model, iters=32, split='F', mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(device_type='cuda', enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.reshape(-1) >= -0.5) & (flow_gt[0].reshape(-1) > -1000)

        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Validation Middlebury{split}: EPE {epe}, D1 {d1}")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-d1': d1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", required=True, choices=["eth3d", "kitti", "things", "tartanair_occ"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors. 
    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, split=args.dataset[-1], mixed_prec=use_mixed_precision)

    elif args.dataset == 'things':
        validate_things(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'tartanair_occ':
        validate_tartanair_occ(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
