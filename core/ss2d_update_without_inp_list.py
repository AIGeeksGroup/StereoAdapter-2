import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add VMamba to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'VMamba'))
from vmamba import SS2D


class ConvSS2D(nn.Module):
    """
    SS2D-based update block that replaces ConvGRU gate mechanism.

    Input:
        - h: hidden state (B, hidden_dim, H, W)
        - x_list: additional input features to concatenate

    Output:
        - h: updated hidden state (B, hidden_dim, H, W)

    Logic:
        Uses SS2D (State Space Model) to replace the traditional GRU gating mechanism.
        Instead of explicit z, r, q gates, SS2D learns the update pattern through
        its selective scan mechanism.

    Scan modes (for ablation study):
        - "cross2d" (default): 4-directional (L→R, T→B, R→L, B→T)
        - "bidi": 2-directional horizontal (L→R, R→L) - aligned with epipolar lines
        - "unidi": 1-directional (L→R only)

    SSM hyperparameters (for ablation study):
        - d_state: SSM state dimension (default: 16)
        - ssm_ratio: SSM expansion ratio (default: 1.0)
    """
    def __init__(self, hidden_dim, input_dim, d_state=16, ssm_ratio=1.0, d_conv=3, scan_mode="cross2d"):
        super(ConvSS2D, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.scan_mode = scan_mode
        self.d_state = d_state
        self.ssm_ratio = ssm_ratio

        # Project concatenated input to hidden_dim for SS2D processing
        self.input_proj = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 1)

        # Map scan_mode to forward_type
        # v3: cross2d (4-directional), v052d: bidi (2-directional), v051d: unidi (1-directional)
        forward_type_map = {
            "cross2d": "v3",      # 4-directional: L→R, T→B, R→L, B→T
            "bidi": "v052d",      # 2-directional: L→R, R→L (horizontal only)
            "unidi": "v051d",     # 1-directional: L→R only
        }
        forward_type = forward_type_map.get(scan_mode, "v3")

        # SS2D module
        self.ss2d = SS2D(
            d_model=hidden_dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.0,
            initialize="v0",
            forward_type=forward_type,
            channel_first=True,
        )

    def forward(self, h, *x_list):
        # Concatenate all inputs
        x = torch.cat(x_list, dim=1)

        # Concatenate hidden state and input
        hx = torch.cat([h, x], dim=1)

        # Project to hidden_dim
        hx = self.input_proj(hx)

        # Apply SS2D to learn the update
        delta_h = self.ss2d(hx)
        delta_h = torch.tanh(delta_h)
        # Residual connection: h_new = h + delta_h
        h = h + delta_h

        return h


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args

        cor_planes = args.corr_levels * (2*args.corr_radius + 1)

        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)


def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)


def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class BasicMultiUpdateBlock(nn.Module):
    """
    Multi-scale update block using SS2D gates instead of ConvGRU.

    Input:
        - net: list of hidden states at different scales
        - inp: list of input features at different scales
        - corr: correlation features
        - flow: current flow estimate
        - iter08/16/32: flags to control which scales to update
        - update: whether to compute flow update

    Output:
        - net: updated hidden states
        - mask: upsampling mask
        - delta_flow: flow update

    Logic:
        Hierarchical update using SS2D at three scales (1/32, 1/16, 1/8).
        SS2D replaces GRU gating for learning temporal dependencies.

    Args (for ablation study):
        args.scan_mode: Scan direction mode
            - "cross2d" (default): 4-directional
            - "bidi": 2-directional horizontal (for epipolar geometry)
            - "unidi": 1-directional
        args.d_state: SSM state dimension (default: 16)
        args.ssm_ratio: SSM expansion ratio (default: 1.0)
    """
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128

        # Get SS2D hyperparameters from args (for ablation study)
        scan_mode = getattr(args, 'scan_mode', 'cross2d')
        d_state = getattr(args, 'd_state', 16)
        ssm_ratio = getattr(args, 'ssm_ratio', 1.0)

        # Replace ConvGRU with ConvSS2D
        self.gru08 = ConvSS2D(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1),
                              d_state=d_state, ssm_ratio=ssm_ratio, scan_mode=scan_mode)
        self.gru16 = ConvSS2D(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2],
                              d_state=d_state, ssm_ratio=ssm_ratio, scan_mode=scan_mode)
        self.gru32 = ConvSS2D(hidden_dims[0], hidden_dims[1],
                              d_state=d_state, ssm_ratio=ssm_ratio, scan_mode=scan_mode)

        self.flow_head = FlowHead(hidden_dims[2], hidden_dim=256, output_dim=2)
        factor = 2**self.args.n_downsample

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor**2)*9, 1, padding=0))

    def forward(self, net, corr=None, flow=None, iter08=True, iter16=True, iter32=True, update=True):

        if iter32:
            net[2] = self.gru32(net[2], pool2x(net[1]))
        if iter16:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru16(net[1], pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru16(net[1], pool2x(net[0]))
        if iter08:
            motion_features = self.encoder(flow, corr)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru08(net[0], motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru08(net[0], motion_features)

        if not update:
            return net

        delta_flow = self.flow_head(net[0])

        # scale mask to balance gradients
        mask = .25 * self.mask(net[0])
        return net, mask, delta_flow
