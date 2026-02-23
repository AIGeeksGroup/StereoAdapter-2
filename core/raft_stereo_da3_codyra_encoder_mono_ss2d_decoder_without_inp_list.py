import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.ss2d_update_without_inp_list import BasicMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from core.utils.utils import coords_grid, upflow8
from depthanything.depth_anything_3.da3_stereo import DepthAnything3
# CoDyRA implementation
class CoDyRA_QKV(nn.Module):
    """CoDyRA for QKV with dynamic rank"""
    def __init__(self, qkv, r, max_kappa=0.005, lambda_reg=1e-4):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.r = r
        self.max_kappa = max_kappa
        self.lambda_reg = lambda_reg

        # LoRA parameters (Q, K, V)
        self.w_a_q = nn.Parameter(torch.zeros(r, self.dim))
        self.w_b_q = nn.Parameter(torch.zeros(self.dim, r))
        self.w_a_k = nn.Parameter(torch.zeros(r, self.dim))
        self.w_b_k = nn.Parameter(torch.zeros(self.dim, r))
        self.w_a_v = nn.Parameter(torch.zeros(r, self.dim))
        self.w_b_v = nn.Parameter(torch.zeros(self.dim, r))

        # Importance weights
        self.i_w_q = nn.Parameter(torch.ones(r))
        self.i_w_k = nn.Parameter(torch.ones(r))
        self.i_w_v = nn.Parameter(torch.ones(r))

        self.is_sparse = False

    def _codyra_result(self, linear_a, linear_b, i_w, x):
        x = x @ linear_a.T
        x *= i_w
        x = x @ linear_b.T
        return x

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self._codyra_result(self.w_a_q, self.w_b_q, self.i_w_q, x)
        new_k = self._codyra_result(self.w_a_k, self.w_b_k, self.i_w_k, x)
        new_v = self._codyra_result(self.w_a_v, self.w_b_v, self.i_w_v, x)
        qkv[:, :, :self.dim] += new_q
        qkv[:, :, self.dim:-self.dim] += new_k
        qkv[:, :, -self.dim:] += new_v
        return qkv

    def compute_sparsity_loss(self):
        if not self.is_sparse:
            return 0
        return self.lambda_reg * (self.i_w_q.abs().sum() + self.i_w_k.abs().sum() + self.i_w_v.abs().sum())

    def update_iws(self, kappa):
        if not self.is_sparse or kappa == 0:
            return
        with torch.no_grad():
            for i_w in [self.i_w_q, self.i_w_k, self.i_w_v]:
                if i_w.grad is None:
                    continue
                signs = torch.sign(i_w)
                i_w.copy_(torch.where(i_w.abs() > kappa, i_w + signs * kappa, torch.zeros_like(i_w)))


try:
    from torch.amp import autocast
except ImportError:
    # Fallback for older PyTorch
    try:
        autocast = torch.cuda.amp.autocast
    except ImportError:
        # dummy autocast for PyTorch < 1.6
        class autocast:
            def __init__(self, device_type='cuda', enabled=True):
                pass
            def __enter__(self):
                pass
            def __exit__(self, *args):
                pass


class RAFTStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        context_dims = args.hidden_dims

        self.baseline = getattr(args, 'baseline', 0.25)
        self.focal = getattr(args, 'focal', 320.0)
        self.use_mde_init = getattr(args, 'use_mde_init', True)
        logging.info(f"MDE init: use_mde_init={self.use_mde_init}, baseline={self.baseline}, focal={self.focal}")

        pretrained_path = getattr(args, 'da3_pretrained', None)
        self.fnet = DepthAnything3(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768],
            pretrained_path=pretrained_path,
        )

        r = getattr(args, 'lora_r', 16)
        max_kappa = getattr(args, 'max_kappa', 0.005)
        lambda_reg = getattr(args, 'lambda_reg', 1e-4)

        for param in self.fnet.pretrained.pretrained.parameters():
            param.requires_grad = False

        self.codyra_layers = nn.ModuleList()
        for blk in self.fnet.pretrained.pretrained.blocks:
            codyra = CoDyRA_QKV(blk.attn.qkv, r, max_kappa, lambda_reg)
            nn.init.kaiming_uniform_(codyra.w_a_q, a=math.sqrt(5))
            nn.init.kaiming_uniform_(codyra.w_a_k, a=math.sqrt(5))
            nn.init.kaiming_uniform_(codyra.w_a_v, a=math.sqrt(5))
            blk.attn.qkv = codyra
            self.codyra_layers.append(codyra)

        logging.info(f"CoDyRA applied (r={r}, kappa={max_kappa}, lambda={lambda_reg})")

        # Feature projection head
        self.fmap_proj = nn.Sequential(
            ResidualBlock(128, 128, 'instance', stride=1),
            nn.Conv2d(128, 256, 3, padding=1)
        )

        # Context encoder outputs (net)
        self.outputs_net = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(128, 128, args.context_norm, stride=1),
                nn.Conv2d(128, context_dims[i], 3, padding=1)
            ) for i in range(3)
        ])

        # Context encoder outputs (inp)
        self.outputs_inp = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(128, 128, args.context_norm, stride=1),
                nn.Conv2d(128, context_dims[i], 3, padding=1)
            ) if i < 2
            else nn.Conv2d(128, context_dims[i], 3, padding=1)
            for i in range(3)
        ])

        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        # Check SS2D backend
        self._check_ss2d_backend()

    def _check_ss2d_backend(self):
        try:
            import selective_scan_cuda_oflex
            oflex_available = True
        except ImportError:
            oflex_available = False

        try:
            import selective_scan_cuda
            mamba_available = True
        except ImportError:
            mamba_available = False

        backend = "oflex" if oflex_available else ("mamba" if mamba_available else "torch")
        logging.info(f"SS2D backend: {backend}")

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def set_sparse_mode(self, is_sparse):
        for layer in self.codyra_layers:
            layer.is_sparse = is_sparse

    def compute_sparsity_loss(self):
        return sum(layer.compute_sparsity_loss() for layer in self.codyra_layers)

    def update_iws(self, kappa):
        for layer in self.codyra_layers:
            layer.update_iws(kappa)

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor * H, factor * W)

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """

        # Save original input size for final upsampling
        H_orig, W_orig = image1.shape[2], image1.shape[3]

        # ========== Normalization: ImageNet stats for DinoV2 ==========
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image1.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image1.device)
        image1 = (image1 / 255.0 - mean) / std
        image2 = (image2 / 255.0 - mean) / std

        # run the context network
        with autocast(device_type='cuda', enabled=self.args.mixed_precision):
            # ========== Extract features from DA3 ==========
            feats1 = self.fnet(image1, return_features=True)
            feats2 = self.fnet(image2, return_features=True)

            # Downsample DA3 features to match RAFT resolution [H/4, H/8, H/16]
            # Use ceil to maintain pyramid structure, crop output later
            factor = 2 ** self.args.n_downsample
            H, W = image1.shape[2], image1.shape[3]
            target_sizes = [
                (math.ceil(H / factor), math.ceil(W / factor)),
                (math.ceil(H / (factor * 2)), math.ceil(W / (factor * 2))),
                (math.ceil(H / (factor * 4)), math.ceil(W / (factor * 4)))
            ]

            feats1 = [F.interpolate(feats1[i], size=target_sizes[i], mode='bilinear', align_corners=True) for i in range(3)]
            feats2 = [F.interpolate(feats2[i], size=target_sizes[i], mode='bilinear', align_corners=True) for i in range(3)]

            fmap1 = self.fmap_proj(feats1[0])
            fmap2 = self.fmap_proj(feats2[0])

            cnet_list = [(self.outputs_net[i](feats1[i]), self.outputs_inp[i](feats1[i])) for i in range(3)]
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

        if self.args.corr_implementation == "reg":  # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "alt":  # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "reg_cuda":  # Faster version of reg
            corr_block = CorrBlockFast1D
        elif self.args.corr_implementation == "alt_cuda":  # Faster version of alt
            corr_block = AlternateCorrBlock
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        coords0, coords1 = self.initialize_flow(net_list[0])

        # MDE initialization: use DA3 depth to initialize disparity
        if self.use_mde_init:
            depth_pred = self.fnet(image1, return_features=False)
            depth_pred = depth_pred.clamp(min=1e-6)
            mde_disp = (self.baseline * self.focal) / depth_pred

            Hc, Wc = net_list[0].shape[2], net_list[0].shape[3]
            mde_disp_low = F.interpolate(mde_disp, size=(Hc, Wc), mode='bilinear', align_corners=True) / float(factor)

            coords1[:, 0:1] = coords0[:, 0:1] - mde_disp_low

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            with autocast(device_type='cuda', enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru:  # Update low-res SS2D
                    net_list = self.update_block(net_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:  # Update low-res SS2D and mid-res SS2D
                    net_list = self.update_block(net_list, iter32=self.args.n_gru_layers == 3, iter16=True,
                                                 iter08=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, corr, flow,
                                                                  iter32=self.args.n_gru_layers == 3,
                                                                  iter16=self.args.n_gru_layers >= 2)

            # in stereo mode, project flow onto epipolar
            delta_flow[:, 1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters - 1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:, :1]

            # Always resize to original input size to ensure shape match
            flow_up = F.interpolate(flow_up, size=(H_orig, W_orig), mode='bilinear', align_corners=True)
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
