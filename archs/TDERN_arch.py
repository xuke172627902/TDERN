import math
import os
import sys

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        self.scale = scale
        self.num_feat = num_feat
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                # m.append(nn.Conv2d(num_feat, 4 * num_feat, 5, 1, 2))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            # m.append(nn.Conv2d(num_feat, 9 * num_feat, 5, 1, 2))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        x, y = input_resolution
        if (self.scale & (self.scale - 1)) == 0:
            flops += self.num_feat * 4 * self.num_feat * 9 * x * y * int(math.log(self.scale, 2))
            # flops += self.num_feat * 4 * self.num_feat * 25 * x * y * int(math.log(self.scale, 2))
        else:
            flops += self.num_feat * 9 * self.num_feat * 9 * x * y
            # flops += self.num_feat * 9 * self.num_feat * 25 * x * y
        return flops

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        h, w = self.patches_resolution if input_resolution is None else input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.img_size if input_resolution is None else input_resolution
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops

class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self, input_resolution=None):
        flops = 0
        return flops



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self, input_resolution=None):
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops = h * w * self.dim
        flops += (h // 2) * (w // 2) * 4 * self.dim * 2 * self.dim
        return flops


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r"""
    Shifted Window-based Multi-head Self-Attention (MSA).

    Args:0
        dim (int): Number of input channels.
        layer_id (int): Index of the current layer in the network.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        num_topk (tuple[int]): Number of top-k attention values retained for sparsity.
        qkv_bias (bool, optional): If True, add a learnable bias to the query, key, and value tensors. Default: True.
    """

    def __init__(self, dim, layer_id, window_size, num_heads, num_topk, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.num_topk = num_topk
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.eps = 1e-20

        # define a parameter table of relative position bias
        if dim > 100:
            # for classical SR
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        else:
            # for lightweight SR
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), 1))  # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.topk = self.num_topk[self.layer_id]

    def forward(self, qkvp, pfa_values, rpi, mask=None, shift=0):
        r"""
        Args:
        qkvp (Tensor): Input tensor containing query, key, value tokens, and LePE positional encoding matrix with shape (num_windows * b, n, c * 4),
        pfa_values (Tensor or None): Precomputed attention values for Progressive Focusing Attention (PFA). If None, standard attention is applied.
        pfa_indices (Tensor or None): Index tensor for Progressive Focusing Attention (PFA), indicating which attention values should be retained or discarded.
        rpi (Tensor): Relative position index tensor, encoding positional information for tokens.
        mask (Tensor or None, optional): Attention mask tensor.
        shift (int, optional): Indicates whether window shifting is applied (e.g., 0 for no shift, 1 for shifted windows). Default: 0.
        """
        b_, n, c4 = qkvp.shape
        c = c4 // 4
        qkvp = qkvp.reshape(b_, n, 4, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v, v_lepe = qkvp[0], qkvp[1], qkvp[2], qkvp[3]

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))  # b_, self.num_heads, n, n
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # nH, Wh*Ww, Wh*Ww
        if not self.training:  # Check if in inference mode
            attn.add_(relative_position_bias)  # only in inference
        else:
            attn = attn + relative_position_bias  # Non-inplace if training

        if shift:
            nw = mask.shape[0]

            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        # Use in-place operations where possible (only in inference mode)
        if not self.training:  # Check if in inference mode
            attn = torch.softmax(attn, dim=-1, out=attn)  # 原地softmax
        else:
            attn = self.softmax(attn)  # Non-inplace if training

        if pfa_values[shift] is not None:
            if not self.training: # only in inference
                attn.mul_(pfa_values[shift])
                attn.add_(self.eps)
                denom = attn.sum(dim=-1, keepdim=True).add_(self.eps)
                attn.div_(denom)
            else:

                attn = (attn * pfa_values[shift])
                attn = (attn + self.eps) / (attn.sum(dim=-1, keepdim=True) + self.eps)


        pfa_values[shift] = attn
        x = ((attn @ v) + v_lepe).transpose(1, 2).reshape(b_, n, c)
        # only in inference. After use, delete unnecessary variables to free memory
        if not self.training:
            del q, k, v, relative_position_bias
            torch.cuda.empty_cache()  # Clear the unused cache
        x = self.proj(x)
        return x, pfa_values

# =============================================================================
# Coverage Loss
# =============================================================================

def coverage_loss(attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the coverage loss for a single TDU attention matrix.

    The loss is the negative mutual information between input tokens and
    dictionary tokens:
        L_cov = -I(j; k) = H(k|j) - H(k)
               = L_ent + L_div

    Args:
        attn: (b, m, n) row-wise softmax attention from TDU.
              Each row sums to 1 over the n input tokens.
        eps:  numerical stability constant.

    Returns:
        Scalar loss value (averaged over batch).
    """
    b, m, n = attn.shape

    # Step 1: column-wise normalisation -> soft assignment distribution
    # col_sum: (b, 1, n)
    A_tilde = F.softmax(attn, dim=1)                  # (b, m, n)

    # Step 2: assignment entropy H(k|j)  -- minimise
    # For each input token j, entropy of its assignment distribution over k
    L_ent = -(A_tilde * torch.log(A_tilde + eps)).sum(dim=1)  # (b, n)
    L_ent = L_ent.mean()                              # scalar

    # Step 3: usage rate p_k and diversity H(k)  -- maximise
    # p_k: average assignment weight for dictionary token k
    # p_k = A_tilde.mean(dim=2)                         # (b, m)
    # L_div = (p_k * torch.log(p_k + eps)).sum(dim=1)  # (b,)  negative entropy
    # L_div = L_div.mean()                              # scalar

    # L_cov = L_ent + L_div
    # L_ent >= 0, L_div <= 0; minimising L_cov minimises entropy and
    # maximises diversity simultaneously.
    return L_ent


def coverage_loss_multi_block(
    attn_list: list,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Aggregate coverage loss across all BasicBlocks.

    Args:
        attn_list: list of (b, m, n) attention tensors, one per BasicBlock.
        eps: numerical stability constant.

    Returns:
        Scalar loss averaged over blocks.
    """
    losses = [coverage_loss(a, eps) for a in attn_list]
    return torch.stack(losses).mean()



# =============================================================================
# ConvFFN
# =============================================================================

class ConvFFN(nn.Module):
    def __init__(self, dim, expansion_factor, ks):
        super().__init__()
        self.dim = dim
        self.ks = ks
        self.expansion_factor = expansion_factor

        hidden_features = int(dim * expansion_factor)
        self.pw1 = nn.Conv2d(dim, hidden_features, 1, 1)
        self.pw2 = nn.Conv2d(hidden_features, dim, 1, 1)
        self.bsdw = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, 1),
            nn.Conv2d(hidden_features, hidden_features, 5, 1, 2,
                      groups=hidden_features))
        self.strip = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features,
                      (ks, 1), 1, ((ks-1)//2, 0), groups=hidden_features),
            nn.Conv2d(hidden_features, hidden_features,
                      (1, ks), 1, (0, (ks-1)//2), groups=hidden_features))
        self.mixer = nn.Conv2d(hidden_features, hidden_features, 1, 1)

    def forward(self, x, x_size=None):
        b, n, c = x.shape
        if x_size is not None:
            h, w = x_size
        else:
            h = w = int(n ** 0.5)
        x2d = x.transpose(1, 2).reshape(b, c, h, w).contiguous()
        x2d = self.pw1(x2d)
        x2d = self.bsdw(F.gelu(x2d))
        x2d = x2d * self.mixer(x2d + self.strip(x2d))
        x2d = self.pw2(x2d)
        return x2d.flatten(2).transpose(1, 2)


# =============================================================================
# ATD_CA_lite
# =============================================================================

class ATD_CA_lite(nn.Module):
    """Lightweight cross-attention: Q from features, K/V from token dictionary."""

    def __init__(self, dim, num_tokens=64, reducted_dim=10, qkv_bias=True):
        super().__init__()
        self.num_tokens = num_tokens
        self.reducted_dim = reducted_dim
        self.wq = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wp = nn.Linear(reducted_dim, dim, bias=qkv_bias)
        self.scale = nn.Parameter(torch.ones(num_tokens) * 0.5,
                                  requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, td):
        # x: (b, n, c)   td: (b, m, c)
        b, n, c = x.shape
        q = F.normalize(self.wq(x),  dim=-1)   # (b, n, rc)
        k = F.normalize(self.wk(td), dim=-1)   # (b, m, rc)
        v = self.wv(td)                         # (b, m, rc)

        scale = torch.clamp(self.scale, 0, 1)
        attn = q @ k.transpose(-2, -1)          # (b, n, m)
        attn = attn * (1 + scale * math.log(self.num_tokens))
        attn = self.softmax(attn)               # (b, n, m)

        out = self.wp(attn @ v)                 # (b, n, c)
        return out


# =============================================================================
# TDU  -- returns (updated_td, attn) so attn can be collected for coverage loss
# =============================================================================

class TDU(nn.Module):
    """
    Token Dictionary Update via scaled dot-product cross-attention.
    Returns both the updated dictionary and the attention matrix so that
    the coverage loss can be computed externally.
    """

    def __init__(self, dim, reducted_dim=10, qkv_bias=True):
        super().__init__()
        self.reducted_dim = reducted_dim
        self.wq = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, td):
        # x: (b, n, c)   td: (b, m, c)
        q = self.wq(td)                                          # (b, m, rd)
        k = self.wk(x)                                           # (b, n, rd)
        v = self.wv(x)                                           # (b, n, c)
        attn = (q @ k.transpose(-2, -1)) / (self.reducted_dim ** 0.5)  # (b, m, n)
        attn = self.softmax(attn)
        td_new = attn @ v                                        # (b, m, c)
        return self.norm(td + td_new), attn   # return attn for coverage loss


# =============================================================================
# TDRefresh
# =============================================================================

class TDRefresh(nn.Module):
    """Affine modulation of dictionary tokens using current-layer feature mean."""

    def __init__(self, dim, num_tokens):
        super().__init__()
        self.scale = nn.Linear(dim, num_tokens, bias=False)
        self.shift = nn.Linear(dim, num_tokens, bias=False)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

    def forward(self, x, td):
        gap = x.mean(dim=1)                    # (b, c)
        s = self.scale(gap).unsqueeze(-1)      # (b, m, 1)
        t = self.shift(gap).unsqueeze(-1)      # (b, m, 1)
        return td * (1 + s) + t



class PFTransformerLayer(nn.Module):
    def __init__(self, dim, block_id, layer_id, input_resolution, num_heads,
                 num_topk, window_size, shift_size, convffn_kernel_size,
                 mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm,
                 num_tokens=64, reducted_dim=10):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.convlepe_kernel_size = convffn_kernel_size
        self.td_refresh = TDRefresh(dim=dim, num_tokens=num_tokens)
        self.v_LePE = dwconv(hidden_features=dim, kernel_size=convffn_kernel_size)

        self.attn_win = WindowAttention(
            dim, layer_id=layer_id, window_size=to_2tuple(window_size),
            num_heads=num_heads, num_topk=num_topk, qkv_bias=qkv_bias)

        self.atd_ca = ATD_CA_lite(
            dim, num_tokens=num_tokens, reducted_dim=reducted_dim,
            qkv_bias=qkv_bias)

        self.mlp = ConvFFN(dim, expansion_factor=mlp_ratio,
                           ks=convffn_kernel_size)

        self.hf_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1))
        self.hf_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, td, pfa_values, x_size, params):
        h, w = x_size
        b, n, c = x.shape
        c4 = 4 * c
        shortcut = x
        x_norm = self.norm1(x)

        # wqkv + LePE (same as v_tdu_sdp)
        x_qkv = self.wqkv(x_norm)
        s = torch.split(x_qkv, c, dim=-1)[-1]
        v_lepe = self.v_LePE(s, x_size)
        x_qkv_s1, x_qkv_s2, x_qkv_s3 = torch.chunk(x_qkv, 3, dim=-1)
        x_qkvp = torch.cat([x_qkv_s1, x_qkv_s3, x_qkv_s2, v_lepe], dim=-1)

        # --- 1. Window attention ---
        if self.shift_size > 0:
            shift = 1
            shifted_x = torch.roll(x_qkvp.reshape(b, h, w, c4),
                                   shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            shift = 0
            shifted_x = x_qkvp.reshape(b, h, w, c4)
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c4)
        attn_win, pfa_values = self.attn_win(
            x_windows, pfa_values=pfa_values,
            rpi=params["rpi_sa"], mask=params["attn_mask"], shift=shift)
        attn_win = attn_win.view(-1, self.window_size, self.window_size, c)
        attn_win = window_reverse(attn_win, self.window_size, h, w)
        if self.shift_size > 0:
            attn_win = torch.roll(attn_win,
                                  shifts=(self.shift_size, self.shift_size),
                                  dims=(1, 2))
        x_win = attn_win.reshape(b, n, c)

        # --- 2. Dictionary cross-attention (with per-layer TDRefresh) ---
        td_fresh = self.td_refresh(x_norm, td)
        x_global = self.atd_ca(self.norm3(shortcut), td_fresh)

        # --- 3. Fusion + FFN ---
        x = shortcut + x_win + x_global
        x = x + self.mlp(self.norm2(x), x_size)

        # --- 4. HF enhancement ---
        x_2d = x.transpose(1, 2).reshape(b, c, h, w)
        hf = self.hf_conv(x_2d).flatten(2).transpose(1, 2)
        x = x + torch.tanh(self.hf_alpha) * hf

        return x, td, pfa_values


# =============================================================================
# BasicBlock  -- collects TDU attention for coverage loss
# =============================================================================

class BasicBlock(nn.Module):
    def __init__(self, dim, input_resolution, idx, layer_id, depth, num_heads,
                 num_topk, window_size, convffn_kernel_size, mlp_ratio=4.,
                 qkv_bias=True, norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, num_tokens=64, reducted_dim=10):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint

        self.layers = nn.ModuleList([
            PFTransformerLayer(
                dim=dim, block_id=idx, layer_id=layer_id + i,
                input_resolution=input_resolution,
                num_heads=num_heads, num_topk=num_topk,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                norm_layer=norm_layer, num_tokens=num_tokens,
                reducted_dim=reducted_dim)
            for i in range(depth)
        ])

        self.tdu = TDU(dim, reducted_dim=reducted_dim, qkv_bias=qkv_bias)

        self.downsample = downsample(
            input_resolution, dim=dim, norm_layer=norm_layer
        ) if downsample is not None else None

    def forward(self, x, td, pfa_values, x_size, params):
        for layer in self.layers:
            if self.use_checkpoint:
                import checkpoint
                x, td, pfa_values = checkpoint.checkpoint(
                    layer, x, td, pfa_values, x_size, params)
            else:
                x, td, pfa_values = layer(x, td, pfa_values, x_size, params)

        # TDU: update dictionary, collect attention for coverage loss
        td, tdu_attn = self.tdu(x, td)   # tdu_attn: (b, m, n)

        if self.downsample is not None:
            x = self.downsample(x)

        return x, td, pfa_values, tdu_attn



class PFTB(nn.Module):
    def __init__(self, dim, idx, layer_id, input_resolution, depth, num_heads,
                 num_topk, window_size, convffn_kernel_size, mlp_ratio,
                 qkv_bias=True, norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, img_size=224, patch_size=4,
                 resi_connection="1conv", num_tokens=64, reducted_dim=10):
        super(PFTB, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0,
            embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0,
            embed_dim=dim, norm_layer=None)

        self.residual_group = BasicBlock(
            dim=dim, input_resolution=input_resolution,
            idx=idx, layer_id=layer_id, depth=depth,
            num_heads=num_heads, num_topk=num_topk,
            window_size=window_size,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            norm_layer=norm_layer, downsample=downsample,
            use_checkpoint=use_checkpoint,
            num_tokens=num_tokens, reducted_dim=reducted_dim)

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, td, pfa_values, x_size, params):
        # residual_group now returns tdu_attn as well
        res, td, pfa_values, tdu_attn = self.residual_group(
            x, td, pfa_values, x_size, params)
        res = self.patch_embed(
            self.conv(self.patch_unembed(res, x_size)))
        return x + res, td, pfa_values, tdu_attn



# =============================================================================
# Main Network
# =============================================================================

@ARCH_REGISTRY.register()
class TDERN(nn.Module):
    """


    Key differences from v_tdu_sdp:
      - TDU.forward() returns (td, attn) so the attention matrix is accessible.
      - BasicBlock.forward() returns (x, td, pfa_values, tdu_attn).
      - PFTB.forward() propagates tdu_attn upward.
      - forward_features() returns (x, attn_list).
      - forward() returns (sr, attn_list) during training, sr during inference.
      - compute_coverage_loss(attn_list, lam) is a convenience wrapper.
    """

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=90,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 num_topk=None,
                 window_size=8,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=4,
                 img_range=1.,
                 upsampler="pixelshuffle",
                 resi_connection="1conv",
                 num_tokens=64,
                 reducted_dim=10,
                 num_feat=64,
                 **kwargs):
        super(TDERN, self).__init__()

        if num_topk is None:
            num_topk = [256, 256, 128, 128, 128, 128,
                        64,  64,  64,  64,  64,  64,
                        32,  32,  32,  32,  32,  32,
                        16,  16,  16,  16,  16,  16]

        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.ape = ape
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        num_out_ch = in_chans

        # 1. Shallow feature extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 2. Deep feature extraction
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # relative position index for SW-MSA (required by WindowAttention)
        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        self.layers = nn.ModuleList()
        self.layer_id = 0
        for i_layer in range(self.num_layers):
            layer = PFTB(
                dim=embed_dim, idx=i_layer, layer_id=self.layer_id,
                input_resolution=(patches_resolution[0],
                                  patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                num_topk=num_topk,
                window_size=window_size,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                norm_layer=norm_layer, downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size, patch_size=patch_size,
                resi_connection=resi_connection,
                num_tokens=num_tokens, reducted_dim=reducted_dim)
            self.layers.append(layer)
            self.layer_id += depths[i_layer]

        self.norm = norm_layer(embed_dim)

        # Global token dictionary
        self.td_init = nn.Parameter(
            torch.randn(num_tokens, embed_dim), requires_grad=True)

        # 3. Reconstruction head
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        if self.upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            self.upsample = UpsampleOneStep(
                upscale, embed_dim, num_out_ch,
                (patches_resolution[0], patches_resolution[1]))
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -(self.window_size // 2)),
                    slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -(self.window_size // 2)),
                    slice(-(self.window_size // 2), None))
        cnt = 0
        for h_s in h_slices:
            for w_s in w_slices:
                img_mask[:, h_s, w_s, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(
            -1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)                   # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        return relative_coords.sum(-1)

    def forward_features(self, x, params):
        """
        Returns:
            x         : (b, n, c) deep features
            attn_list : list[Tensor(b, m, n)], one per PFTB
        """
        x_size = (x.shape[2], x.shape[3])
        pfa_values = [None, None]

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed

        b = x.shape[0]
        td = self.td_init.unsqueeze(0).expand(b, -1, -1).contiguous()

        attn_list = []
        for layer in self.layers:
            x, td, pfa_values, tdu_attn = layer(
                x, td, pfa_values, x_size, params)
            attn_list.append(tdu_attn)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x, attn_list

    def forward(self, x):
        """
        Training  : returns (sr_image, attn_list)
        Inference : returns sr_image
        """
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        self.mean = self.mean.type_as(x)
        x = x * self.img_range

        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {"attn_mask": attn_mask, "rpi_sa": self.relative_position_index_SA}

        feat = self.conv_first(x)
        deep, attn_list = self.forward_features(feat, params)

        if self.upsampler == "pixelshuffle":
            out = self.conv_before_upsample(
                self.conv_after_body(deep) + feat)
            out = self.conv_last(self.upsample(out))
        elif self.upsampler == "pixelshuffledirect":
            out = self.upsample(self.conv_after_body(deep) + feat)
        else:
            out = self.conv_last(self.conv_after_body(deep) + feat)

        out = out / self.img_range
        out = out[:, :, :h_ori * self.upscale, :w_ori * self.upscale]

        if self.training:
            return out, attn_list
        return out

    def compute_coverage_loss(self, attn_list, lam=1e-3):
        """
        Convenience wrapper.

        Args:
            attn_list : list of (b, m, n) tensors from forward().
            lam       : loss weight lambda.
        Returns:
            Scalar tensor: lam * L_cov.
        """
        return lam * coverage_loss_multi_block(attn_list)


if __name__ == '__main__':
  import torch


  base_kwargs = dict(
      img_size=64, patch_size=1, in_chans=3, embed_dim=40,
      depths=[3,3,3,3], num_heads=[4, 4, 4, 4],
      window_size=24, convffn_kernel_size=5, mlp_ratio=1,
      upsampler='pixelshuffledirect', resi_connection='1conv',
      num_tokens=16, reducted_dim=8,
  )

  # ── 参数量对比 ────────────────────────────────────────────────────────────
  for scale in [2]:
      model = TDERN(upscale=scale, **base_kwargs)
      model.eval()
      x = torch.rand(1, 3, 64, 64)
      with torch.no_grad():
          out = model(x)
      total = sum(p.nelement() for p in model.parameters())
      print(f'x{scale}  output: {out.shape}   params: {total:,}')

  # ── 模块级参数分解 ────────────────────────────────────────────────────────

  # td_init 单独统计
  print(f'  {"td_init":<25} {model.td_init.nelement():>10,}')


