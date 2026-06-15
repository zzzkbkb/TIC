import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import random
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from timm.models.layers import trunc_normal_
from .utils import update_registered_buffers, SeparableConv2d, conv, deconv

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def haar_wavelet_split(x):
    """
    Input: (B, 3, H, W)
    Output:
       LL: (B, 3, H/2, W/2) - 低频
       LH: (B, 3, H/2, W/2) - 垂直方向高频
       HL: (B, 3, H/2, W/2) - 水平方向高频
       HH: (B, 3, H/2, W/2) - 对角方向高频
    """
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    LL = (x00 + x01 + x10 + x11) / 4.0
    LH = (x00 + x01 - x10 - x11) / 4.0
    HL = (x00 - x01 + x10 - x11) / 4.0
    HH = (x00 - x01 - x10 + x11) / 4.0
    return LL, LH, HL, HH


class LGCEM(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(LGCEM, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.inter_channels = in_channels // ratio

        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=False),
            nn.LayerNorm([self.inter_channels, 1, 1]),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.inter_channels, in_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        n, c, h, w = x.size()

        # Context Modeling
        input_x = x.view(n, c, h * w)  # (N, C, H*W)
        context_mask = self.conv_mask(x).view(n, 1, h * w)  # (N, 1, H*W)
        context_mask = self.softmax(context_mask)  # (N, 1, H*W)

        context = torch.bmm(input_x, context_mask.transpose(1, 2))  # (N, C, 1)
        context = context.view(n, c, 1, 1)  # (N, C, 1, 1)

        # Transform
        channel_add_term = self.channel_add_conv(context)
        out = x + channel_add_term
        return out


# 创新：使用 tanh 和交替的 MLP 网络以及残差门控
# SiLU vs ReLU
class BCFM(nn.Module):
    def __init__(self, c1, c2, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        hidden1 = max(1, c2 // reduction)
        self.mlp1 = nn.Sequential(
            nn.Conv2d(c2, hidden1, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden1, c1, 1, bias=False),
            nn.Tanh(),
        )

        hidden2 = max(1, c1 // reduction)
        self.mlp2 = nn.Sequential(
            nn.Conv2d(c1, hidden2, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden2, c2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x1, x2):
        # Residual gating: multiplier in [0, 2]
        g1 = self.mlp1(self.avg_pool(x2))
        x1_out = x1 * (1.0 + g1)

        g2 = self.mlp2(self.avg_pool(x1))
        x2_out = x2 * (1.0 + g2)

        return x1_out, x2_out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_att = torch.cat([avg_out, max_out], dim=1)
        x_att = self.conv1(x_att)
        return self.sigmoid(x_att)


class TIC_light(nn.Module):
    """
    Final version:
      - forward() and compress()/decompress() share EXACT same semantics
      - hyperprior runs on y
      - gaussian params predict distribution of y directly
      - decompress() does NOT require any additional input
    """

    def __init__(self, N, M, act="silu"):
        super().__init__()
        self.N = N
        self.M = M

        Act = nn.SiLU if act.lower() != "relu" else nn.ReLU

        # ---------------- g_a ----------------
        # Input is now LL (H/2), so we adjust strides to reach M (H/32 of original = H/16 of LL)
        self.g_a = nn.Sequential(
            SeparableConv2d(3, N, kernel_size=5, stride=2, padding=2),
            Act(inplace=True),
            nn.Conv2d(N, N, kernel_size=1, stride=1, padding=0),

            SeparableConv2d(N, N, kernel_size=3, stride=2, padding=1),
            Act(inplace=True),
            nn.Conv2d(N, N, kernel_size=1, stride=1, padding=0),

            SeparableConv2d(N, N, kernel_size=3, stride=2, padding=1),
            Act(inplace=True),
            nn.Conv2d(N, N, kernel_size=1, stride=1, padding=0),

            SeparableConv2d(N, M, kernel_size=3, stride=2, padding=1),
            LGCEM(M),
        )

        # ---------------- directional HF encoder ----------------
        # Preserve LH / HL / HH identities and allocate 8 latent channels to each subband.
        self.hf_band_dim = 8
        self.hf_lat_dim = self.hf_band_dim * 3

        self.hf_enc_shared = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),  # -> H/4
            Act(inplace=True),
            nn.Conv2d(24, 16, kernel_size=3, stride=2, padding=1),  # -> H/8
            Act(inplace=True),
            LGCEM(16),
        )
        self.hf_enc_head_lh = nn.Conv2d(16, self.hf_band_dim, kernel_size=1, stride=1, padding=0)
        self.hf_enc_head_hl = nn.Conv2d(16, self.hf_band_dim, kernel_size=1, stride=1, padding=0)
        self.hf_enc_head_hh = nn.Conv2d(16, self.hf_band_dim, kernel_size=1, stride=1, padding=0)

        # ---------------- h_a ----------------
        self.h_a = nn.Sequential(
            SeparableConv2d(M, N, kernel_size=3, stride=1, padding=1),
            Act(inplace=True),
            nn.Conv2d(N, N, kernel_size=1, stride=1, padding=0),

            SeparableConv2d(N, N, kernel_size=3, stride=2, padding=1),
            Act(inplace=True),
            nn.Conv2d(N, N, kernel_size=1, stride=1, padding=0),

            SeparableConv2d(N, N, kernel_size=3, stride=2, padding=1),
        )

        # ---------------- h_s (Cloud-side Parameter Prediction) ----------------
        # Shared backbone
        self.h_s_backbone = nn.Sequential(
            SeparableConv2d(N, N, kernel_size=3, stride=1, padding=1),
            Act(inplace=True),
            nn.Conv2d(N, N, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode="nearest"),

            SeparableConv2d(N, N, kernel_size=3, stride=1, padding=1),
            Act(inplace=True),
            nn.Conv2d(N, N, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode="nearest"),

            SeparableConv2d(N, N, kernel_size=3, stride=1, padding=1),
            Act(inplace=True),
            nn.Conv2d(N, N, kernel_size=1, stride=1, padding=0),
        )

        # Head 1: Y Parameters (Scale & Mean) - Output size matches y (H/32)
        self.h_s_y_head = SeparableConv2d(N, M * 2, kernel_size=3, stride=1, padding=1)

        # ---------------- g_s (RSTB decoder) ----------------
        self.dec_dim = min(64, N)
        self.hf_fusion_scale = nn.Parameter(torch.tensor(0.5))

        depths = [2, 4, 6, 2, 2]
        num_heads = [4, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 4.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.2
        norm_layer = nn.LayerNorm
        use_checkpoint = False

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        depths_r = depths[::-1]
        num_heads_r = num_heads[::-1]

        from compressai.layers import RSTB
        self.g_s0 = deconv(M, N, kernel_size=3, stride=2)
        self.g_s1 = RSTB(
            dim=N, input_resolution=(32, 32), depth=depths_r[2], num_heads=num_heads_r[2],
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths_r[:2]):sum(depths_r[:3])],
            norm_layer=norm_layer, use_checkpoint=use_checkpoint
        )
        self.g_s2 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s3 = RSTB(
            dim=N, input_resolution=(64, 64), depth=depths_r[3], num_heads=num_heads_r[3],
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths_r[:3]):sum(depths_r[:4])],
            norm_layer=norm_layer, use_checkpoint=use_checkpoint
        )
        self.g_s4 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s5 = RSTB(
            dim=N, input_resolution=(128, 128), depth=depths_r[4], num_heads=num_heads_r[4],
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths_r[:4]):sum(depths_r[:5])],
            norm_layer=norm_layer, use_checkpoint=use_checkpoint
        )

        # 主干只解到 H/2 feature，不直接出 RGB
        self.g_s6_feat = deconv(N, self.dec_dim, kernel_size=5, stride=2)
        self.main_refine = nn.Sequential(
            SeparableConv2d(self.dec_dim, self.dec_dim, kernel_size=3, padding=1),
            Act(inplace=True),
            LGCEM(self.dec_dim),
            SeparableConv2d(self.dec_dim, self.dec_dim, kernel_size=3, padding=1),
            Act(inplace=True),
            LGCEM(self.dec_dim),
            SeparableConv2d(self.dec_dim, self.dec_dim, kernel_size=3, padding=1),
        )

        # HF decoder: H/8 -> H/4 -> H/2
        self.hf_dec = nn.Sequential(
            nn.Conv2d(self.hf_lat_dim, 32, kernel_size=3, stride=1, padding=1),
            Act(inplace=True),
            LGCEM(32),

            deconv(32, self.dec_dim, kernel_size=3, stride=2),  # H/8 -> H/4
            Act(inplace=True),
            SeparableConv2d(self.dec_dim, self.dec_dim, kernel_size=3, stride=1, padding=1),
            Act(inplace=True),
            LGCEM(self.dec_dim),

            deconv(self.dec_dim, self.dec_dim, kernel_size=5, stride=2),  # H/4 -> H/2
            Act(inplace=True),
            SeparableConv2d(self.dec_dim, self.dec_dim, kernel_size=3, stride=1, padding=1),
            Act(inplace=True),
            LGCEM(self.dec_dim),
        )

        self.hf_refine = nn.Sequential(
            SeparableConv2d(self.dec_dim, self.dec_dim, kernel_size=3, padding=1),
            Act(inplace=True),
            LGCEM(self.dec_dim),
            SeparableConv2d(self.dec_dim, self.dec_dim, kernel_size=3, padding=1),
            Act(inplace=True),
            LGCEM(self.dec_dim),
            SeparableConv2d(self.dec_dim, self.dec_dim, kernel_size=3, padding=1),
        )

        self.feature_fusion_attn = BCFM(self.dec_dim, self.dec_dim)

        self.feature_fusion = nn.Sequential(
            nn.Conv2d(self.dec_dim * 2, self.dec_dim, kernel_size=3, stride=1, padding=1),
            Act(inplace=True),
            SeparableConv2d(self.dec_dim, self.dec_dim, kernel_size=3, stride=1, padding=1),
            Act(inplace=True),
            LGCEM(self.dec_dim),
            SeparableConv2d(self.dec_dim, self.dec_dim, kernel_size=3, stride=1, padding=1),
        )

        # 融合后再 to_rgb: H/2 -> H
        self.to_rgb = nn.Sequential(
            SeparableConv2d(self.dec_dim, self.dec_dim, kernel_size=3, stride=1, padding=1),
            Act(inplace=True),
            deconv(self.dec_dim, 3, kernel_size=5, stride=2),
        )

        # ---------------- entropy models ----------------
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

        self.hf_gaussian = GaussianConditional(None)

        self.final_spatial_attn = SpatialAttention(kernel_size=7)  # 实例化最终的空间注意力模块
        self.final_spatial_attn_scale = nn.Parameter(torch.zeros(1))

        # Post-processing network
        self.post_process_net = nn.Sequential(
            SeparableConv2d(3, 32, kernel_size=3, stride=1, padding=1),
            Act(inplace=True),
            SeparableConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            Act(inplace=True),
            SeparableConv2d(32, 3, kernel_size=3, stride=1, padding=1),
        )

        # ---------------- directional HF prior / residual entropy model ----------------
        self.hf_prior_stem = nn.Sequential(
            nn.Conv2d(M, N, kernel_size=1, stride=1, padding=0),
            Act(inplace=True),
            nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1),
            Act(inplace=True),
            nn.Conv2d(N, 32, kernel_size=3, stride=1, padding=1),
            Act(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            Act(inplace=True),
        )
        self.hf_pred_lh = nn.Conv2d(32, self.hf_band_dim, kernel_size=3, stride=1, padding=1)
        self.hf_pred_hl = nn.Conv2d(32, self.hf_band_dim, kernel_size=3, stride=1, padding=1)
        self.hf_pred_hh = nn.Conv2d(32, self.hf_band_dim, kernel_size=3, stride=1, padding=1)

        self.hf_param_lh = nn.Conv2d(32, self.hf_band_dim * 2, kernel_size=3, stride=1, padding=1)
        self.hf_param_hl = nn.Conv2d(32, self.hf_band_dim * 2, kernel_size=3, stride=1, padding=1)
        self.hf_param_hh = nn.Conv2d(32, self.hf_band_dim * 2, kernel_size=3, stride=1, padding=1)

    def g_s(self, x):
        # keep your robust padding to window=8
        def _pad_to_window(t, win=8):
            B, C, H, W = t.shape
            pad_h = (win - H % win) % win
            pad_w = (win - W % win) % win
            if pad_h or pad_w:
                t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
            return t, (H, W), (pad_h, pad_w)

        def _unpad(t, orig_hw, pads):
            H, W = orig_hw
            pad_h, pad_w = pads
            if pad_h or pad_w:
                t = t[:, :, :H, :W]
            return t

        x = self.g_s0(x)
        x, orig_hw, pads = _pad_to_window(x, win=8)
        x = self.g_s1(x, (x.shape[-2], x.shape[-1]))
        x = _unpad(x, orig_hw, pads)

        x = self.g_s2(x)
        x, orig_hw, pads = _pad_to_window(x, win=8)
        x = self.g_s3(x, (x.shape[-2], x.shape[-1]))
        x = _unpad(x, orig_hw, pads)

        x = self.g_s4(x)
        x, orig_hw, pads = _pad_to_window(x, win=8)
        x = self.g_s5(x, (x.shape[-2], x.shape[-1]))
        x = _unpad(x, orig_hw, pads)

        x = self.g_s6_feat(x)
        x = x + self.main_refine(x)
        return x

    def aux_loss(self):
        return sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _cat_hf_bands(self, bands):
        return torch.cat(list(bands), dim=1)

    def _split_hf_bands(self, hf_tensor):
        return torch.split(hf_tensor, self.hf_band_dim, dim=1)

    def _encode_hf_bands(self, LH, HL, HH):
        feat_lh = self.hf_enc_shared(LH)
        feat_hl = self.hf_enc_shared(HL)
        feat_hh = self.hf_enc_shared(HH)

        hf_lat_lh = self.hf_enc_head_lh(feat_lh)
        hf_lat_hl = self.hf_enc_head_hl(feat_hl)
        hf_lat_hh = self.hf_enc_head_hh(feat_hh)
        return hf_lat_lh, hf_lat_hl, hf_lat_hh
    
    # 用低频分量预测高频分量的先验分布
    def _hf_prior_from_y(self, y_ref, target_size):
        y_up = F.interpolate(y_ref, size=target_size, mode="nearest")
        prior_feat = self.hf_prior_stem(y_up)

        pred_lh = self.hf_pred_lh(prior_feat)
        pred_hl = self.hf_pred_hl(prior_feat)
        pred_hh = self.hf_pred_hh(prior_feat)

        lh_params = self.hf_param_lh(prior_feat)
        hl_params = self.hf_param_hl(prior_feat)
        hh_params = self.hf_param_hh(prior_feat)

        lh_scales, lh_means = lh_params.chunk(2, 1)
        hl_scales, hl_means = hl_params.chunk(2, 1)
        hh_scales, hh_means = hh_params.chunk(2, 1)

        pred_bands = (pred_lh, pred_hl, pred_hh)
        scale_bands = (lh_scales, hl_scales, hh_scales)
        mean_bands = (lh_means, hl_means, hh_means)
        return pred_bands, scale_bands, mean_bands

    # 对这些方向的高频残差进行解码，得到高频分量的重建
    def _reconstruct_hf_from_residual(self, hf_residual_hat, pred_bands):
        """
        Reconstruct high-frequency bands from their residuals and predictions.
        """
        residual_bands = self._split_hf_bands(hf_residual_hat)
        hf_hat_bands = [r + p for r, p in zip(residual_bands, pred_bands)]
        return self._cat_hf_bands(hf_hat_bands)

    def _encode(self, x):
        """
        Shared encoder path for forward() and compress().
        """
        # 1. Haar Split
        LL, LH, HL, HH = haar_wavelet_split(x)

        # 2. Encode Branches
        y = self.g_a(LL)
        hf_lat_bands = self._encode_hf_bands(LH, HL, HH)
        z = self.h_a(y)

        # 3. Hyperprior
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        # 4. Predict Parameters from Z
        params_feat = self.h_s_backbone(z_hat)

        # Predict Y params
        y_params = self.h_s_y_head(params_feat)
        if y_params.shape[-2:] != y.shape[-2:]:
            y_params = F.interpolate(y_params, size=y.shape[-2:], mode="nearest")
        scales_hat, means_hat = y_params.chunk(2, 1)
        return y, z, z_hat, z_likelihoods, scales_hat, means_hat, hf_lat_bands

    def _decode_image(self, y_hat, hf_hat):
        main_feat = self.g_s(y_hat)  # H/2, dec_dim
        hf_feat = self.hf_dec(hf_hat)  # H/2, dec_dim
        hf_feat = hf_feat + self.hf_refine(hf_feat)

        if hf_feat.shape[-2:] != main_feat.shape[-2:]:
            hf_feat = F.interpolate(
                hf_feat,
                size=main_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        main_feat_mod, hf_feat_mod = self.feature_fusion_attn(main_feat, hf_feat)
        fused_feat = self.feature_fusion(torch.cat([main_feat_mod, hf_feat_mod], dim=1))
        fused_feat = fused_feat + main_feat + self.hf_fusion_scale * hf_feat

        x_hat = self.to_rgb(fused_feat)

        spatial_att_map = self.final_spatial_attn(x_hat)
        x_hat = x_hat + self.final_spatial_attn_scale * (x_hat * spatial_att_map)
        
        # Apply post-processing network
        x_hat = self.post_process_net(x_hat)
        #x_hat = x_hat + self.post_process_net(x_hat)
        return x_hat

    # ---------------- forward (training / entropy estimation) ----------------
    def forward(self, x):
        y, z, z_hat, z_likelihoods, scales_hat, means_hat, hf_lat_bands = self._encode(x)

        mode = "noise" if self.training else "dequantize"

        y_hat = self.gaussian_conditional.quantize(y, mode, means_hat)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        target_size = hf_lat_bands[0].shape[-2:]
        pred_bands, scale_bands, mean_bands = self._hf_prior_from_y(y_hat, target_size)
        hf_residual_bands = [h - p for h, p in zip(hf_lat_bands, pred_bands)]
        hf_residual = self._cat_hf_bands(hf_residual_bands)
        hf_scales_hat = self._cat_hf_bands(scale_bands)
        hf_means_hat = self._cat_hf_bands(mean_bands)

        hf_residual_hat = self.hf_gaussian.quantize(hf_residual, mode, hf_means_hat)
        _, hf_likelihoods = self.hf_gaussian(hf_residual, hf_scales_hat, means=hf_means_hat)
        hf_hat = self._reconstruct_hf_from_residual(hf_residual_hat, pred_bands)

        x_hat = self._decode_image(y_hat, hf_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods, "hf": hf_likelihoods},
            "y": y,
            "z": z,
        }

    # ---------------- update / load ----------------
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)
        self.hf_gaussian.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if isinstance(m, EntropyBottleneck):
                updated |= m.update(force=force)
        return updated

    def load_state_dict(self, state_dict, strict=True):
        if all(
                f"entropy_bottleneck.{k}" in state_dict
                for k in ["_quantized_cdf", "_offset", "_cdf_length"]
        ):
            update_registered_buffers(
                self.entropy_bottleneck,
                "entropy_bottleneck",
                ["_quantized_cdf", "_offset", "_cdf_length"],
                state_dict,
            )

        if all(
                f"gaussian_conditional.{k}" in state_dict
                for k in ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"]
        ):
            update_registered_buffers(
                self.gaussian_conditional,
                "gaussian_conditional",
                ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                state_dict,
            )

        if all(
                f"hf_gaussian.{k}" in state_dict
                for k in ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"]
        ):
            update_registered_buffers(
                self.hf_gaussian,
                "hf_gaussian",
                ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                state_dict,
            )

        try:
            super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            if strict and ("Missing key(s)" in str(e) or "Unexpected key(s)" in str(e)):
                super().load_state_dict(state_dict, strict=False)
            else:
                raise

    @classmethod
    def from_state_dict(cls, state_dict):
        pw_keys = []
        for k in state_dict.keys():
            m = re.match(r"g_a\.(\d+)\.pointwise\.weight$", k)
            if m:
                pw_keys.append((int(m.group(1)), k))

        if pw_keys:
            pw_keys.sort(key=lambda x: x[0])
            N = state_dict[pw_keys[0][1]].size(0)
            M = state_dict[pw_keys[-1][1]].size(0)
        else:
            N = state_dict["g_a.0.weight"].size(0)
            last = sorted(
                [k for k in state_dict.keys() if re.match(r"g_a\.\d+\.weight$", k)],
                key=lambda s: int(s.split(".")[1]),
            )[-1]
            M = state_dict[last].size(0)

        net = cls(N, M)
        # 兼容旧权重：如果缺少 level_embed 等新键，允许非严格加载
        # Missing keys will be initialized randomly (no effect if not used or handled)
        try:
            net.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            # 只有当确实是因为缺少键导致错误时才降级为 strict=False
            if "Missing key(s)" in str(e):
                print(f"Warning: Loading old checkpoint without new keys (level_embed, etc.). Strict mode disabled.")
                net.load_state_dict(state_dict, strict=False)
            else:
                raise e
        return net

    # ---------------- real bitstream codec ----------------
    @torch.no_grad()
    def compress(self, x):
        y, z, z_hat, z_likelihoods, scales_hat, means_hat, hf_lat_bands = self._encode(x)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params_feat = self.h_s_backbone(z_hat)

        y_params = self.h_s_y_head(params_feat)
        if y_params.shape[-2:] != y.shape[-2:]:
            y_params = F.interpolate(y_params, size=y.shape[-2:], mode="nearest")
        scales_hat, means_hat = y_params.chunk(2, 1)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=means_hat)

        target_size = hf_lat_bands[0].shape[-2:]
        pred_bands, scale_bands, mean_bands = self._hf_prior_from_y(y_hat, target_size)
        hf_residual_bands = [h - p for h, p in zip(hf_lat_bands, pred_bands)]
        hf_residual = self._cat_hf_bands(hf_residual_bands)
        hf_scales_hat = self._cat_hf_bands(scale_bands)
        hf_means_hat = self._cat_hf_bands(mean_bands)

        hf_indexes = self.hf_gaussian.build_indexes(hf_scales_hat)
        hf_strings = self.hf_gaussian.compress(hf_residual, hf_indexes, means=hf_means_hat)

        return {
            "strings": [y_strings, z_strings, hf_strings],
            "shape": tuple(z.size()[-2:]),
            "y_shape": tuple(y.size()[-2:]),
            "hf_shape": tuple(target_size),
        }

    @torch.no_grad()
    def decompress(self, strings, shape, y_shape=None, hf_shape=None):
        assert isinstance(strings, list) and len(strings) == 3
        shape = tuple(shape)
        if y_shape is None:
            y_shape = (shape[0] * 4, shape[1] * 4)
        if hf_shape is None:
            hf_shape = (shape[0] * 16, shape[1] * 16)

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        params_feat = self.h_s_backbone(z_hat)

        y_params = self.h_s_y_head(params_feat)
        if y_params.shape[-2:] != y_shape:
            y_params = F.interpolate(y_params, size=y_shape, mode="nearest")
        scales_hat, means_hat = y_params.chunk(2, 1)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)

        pred_bands, scale_bands, mean_bands = self._hf_prior_from_y(y_hat, hf_shape)
        hf_scales_hat = self._cat_hf_bands(scale_bands)
        hf_means_hat = self._cat_hf_bands(mean_bands)

        hf_indexes = self.hf_gaussian.build_indexes(hf_scales_hat)
        hf_residual_hat = self.hf_gaussian.decompress(strings[2], hf_indexes, means=hf_means_hat)
        hf_hat = self._reconstruct_hf_from_residual(hf_residual_hat, pred_bands)

        x_hat = self._decode_image(y_hat, hf_hat)

        return {"x_hat": x_hat.clamp(0, 1)}