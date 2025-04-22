import os
from utils import extract_lines, plot_multi_channel, collate_subsample_points, align, update_mol_positions, \
    get_mol_reps, collate_fn_compact_expl_ah,collate_fn_compact_expl_h, create_gaussian_batch_pdf_values, collate_fn_general, collate_fn_general_noatmNo,transform_training_batch_into_visualizable, visualize_mol
from pysmiles import read_smiles
from rdkit.Chem import AllChem
import networkx as nx
import pickle
from  torch.distributions import Normal, kl_divergence
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import plotly.express as px
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from test_utils import fit_pred_field,fit_pred_field_sep_chn_batch

import logging
logging.getLogger('some_logger')

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
import numpy as np
from sklearn.mixture import GaussianMixture
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from denoising_diffusion_pytorch.version import __version__
import datetime


def get_current_datetime():
    return str(datetime.datetime.now()).split(".")[0][2:]


ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def unnormalize_also_noise(imgs):
    print("I should not be here?")
    imgs = [(img - img.min())/(img.max()-img.min()) for img in imgs[0]]
    return imgs

# small helper modules



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Upsample3D(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv3d(dim, default(dim_out, dim), 3, padding = 1)
    )
def Downsample3D(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) (l p3) -> b (c p1 p2 p3) h w l', p1 = 2, p2 = 2, p3 = 2),
        nn.Conv3d(dim * 8, default(dim_out, dim), 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv3d(nn.Conv3d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv3d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        if torch.isinf(torch.sum(F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups))) or \
                torch.isneginf(torch.sum(F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups))):
            print("inf/-inf encountered in PURPORTEDLY;")
            breakpoint()


        if torch.isnan(torch.sum(F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups))):
            print("nan encountered in WeightStandardizedConv2d")
            breakpoint()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm3D(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm3D(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
class WeightStandardizedConv3D(nn.Conv3d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv3d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# building block modules
class Block3D(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv3D(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.normal_conv = torch.nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x, scale_shift = None):
        # x_proj = self.proj(x)
        x_proj=self.normal_conv(x)
        x_norm = self.norm(x_proj)

        if torch.isnan(torch.sum(x_norm)):
            print("Nan encountered in Block from GroupNorm")
            breakpoint()

        if exists(scale_shift):
            scale, shift = scale_shift
            x_norm = x_norm * (scale + 1) + shift

        x = self.act(x_norm)
        return x

class ResnetBlock3D(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block3D(dim, dim_out, groups = groups)
        self.block2 = Block3D(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention3D(nn.Module):
    # taken from https://arxiv.org/pdf/2007.14902.pdf I think (original DDPM does not  cite this though)

    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv3d(hidden_dim, dim, 1),
            LayerNorm3D(dim)
        )

    def forward(self, x):
        b, c, h, w, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h c (x y z)', h = self.heads), qkv)
        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)


        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y z) -> b (h c) x y z', h = self.heads, x = h, y = w, z=l)
        return self.to_out(out)

class LinearAttention(nn.Module):
    # taken from https://arxiv.org/pdf/2007.14902.pdf I think (original DDPM does not  cite this though)

    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention3D_legacy(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w, l = x.shape

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h c (x y z)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x = h, y = w, z=l)
        return self.to_out(out)
    
def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W, D/2)
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])
    emb = np.concatenate([emb_h, emb_w, emb_d], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_d = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h, grid_d, indexing='ij')  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    pos_embed = Rearrange('(h w l) c -> 1 c h w l',h=grid_size,w=grid_size,l=grid_size)(torch.tensor(pos_embed))

    return pos_embed.cpu().numpy()


class Attention3D(nn.Module):
    def __init__(self, dim, heads = 12, dim_head = 32, res_dim=8, wo_pe=False):
        super().__init__()
        dim_head = dim // heads
     
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, res_dim, res_dim, res_dim), requires_grad=False)
        if not wo_pe:
            pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[1], res_dim)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())
        self.wo_pe = wo_pe


    def forward(self, x):
        b, c, h, w, l = x.shape
        x = x + self.pos_embed if not self.wo_pe else x
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h c (x y z)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x = h, y = w, z=l)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)

        if torch.isnan(torch.sum(self.to_out(out))):
            print("Nan encountered in Attention")
            breakpoint()

        return self.to_out(out)

# model

class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 4,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        legacy_attention=False,
        attention_grid=None,
        add_pe=False
    ):
        super().__init__()

        # determine dimensions
        if not exists(attention_grid): attention_grid = 32// (2**(len(dim_mults)-1))

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d(input_channels, init_dim, 5, padding = 2)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock3D, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm3D(dim_in, LinearAttention3D(dim_in))),
                Downsample3D(dim_in, dim_out) if not is_last else nn.Conv3d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        if legacy_attention:
            self.mid_attn = Residual( PreNorm3D(mid_dim, Attention3D_legacy(mid_dim))    )
        else:
            self.mid_attn = Residual( PreNorm3D(mid_dim, Attention3D(mid_dim, res_dim=attention_grid, wo_pe=not add_pe))    )
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm3D(dim_out, LinearAttention3D(dim_out))),
                Upsample3D(dim_out, dim_in) if not is_last else  nn.Conv3d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv3d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        full_att_16x=False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            if full_att_16x and dim_in == 16:
                self.downs.append(nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                    Residual(PreNorm(dim_in, Attention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
                ]))
            elif full_att_16x and dim_in != 16:
                self.downs.append(nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    torch.nn.Identity(dim_in),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                ]))
            else:
                self.downs.append(nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            if full_att_16x and dim_in == 16:
                self.ups.append(nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, Attention(dim_out))),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                ]))
            elif full_att_16x and dim_in == 16:
                self.ups.append(nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    torch.nn.Identity(dim_in),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                ]))
            else:
                self.ups.append(nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
                ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class CNN(nn.Module):
    def __init__(
        self,
        channels,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        kernel_sizes=[3,3,3,3]
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition

        layers_cnn = []
        activation_fns = []
        layers_cnn.append(nn.Conv3d(channels+1, 64, kernel_size = kernel_sizes[0], stride = 1, padding = 'same'))
        layers_cnn.append(nn.Conv3d(64, 128, kernel_size = kernel_sizes[1], stride = 1, padding = 'same'))
        layers_cnn.append(nn.Conv3d(128,64, kernel_size = kernel_sizes[2], stride = 1, padding = 'same'))
        layers_cnn.append(nn.Conv3d(64, channels, kernel_size = kernel_sizes[3], stride = 1, padding = 'same'))
        activation_fns.append(nn.Sigmoid())
        activation_fns.append(nn.ELU(alpha=1.0))
        activation_fns.append(nn.ELU(alpha=1.0))
        self.layer_cnn = nn.ModuleList(layers_cnn)
        self.activation_cnn = nn.ModuleList(activation_fns)



    def forward(self, x, time, x_self_cond = None):
        tt = time.view(-1,1,1,1,1)
        tt = tt.expand((len(time),1,x.shape[2],x.shape[3],x.shape[4]))
        dx_final = torch.cat([tt.float(), x.float()], 1)
        for l,layer in enumerate(self.layer_cnn):
            dx_final = layer(dx_final.float())
            if l !=len(self.layer_cnn)-1:
                dx_final = self.activation_cnn[l](dx_final)
        return dx_final.float()
# gaussian diffusion trainer class

def extract_multithrsh(a, t, x_shape, inds, channel_size):
    # used when different shecdules go for different channels
    # a (which can be alpha cumprod (\Bar{alpha}) is different for different channels, and inds is a list of the form
    # [[i11, i12, ..]. [i21, i22, ...]]), where i11, i12 correspond to first alpha-cumprod
    b, *_ = t.shape
    empty_tens = torch.zeros((b, channel_size,*((1,) * (len(x_shape) - 2))), device=t.device, dtype=a.dtype) 
    for inds_, a_ in zip (inds, a):
        out = a_.gather(-1, t).reshape(b, *((1,) * (len(x_shape) - 1)))
        empty_tens[:,inds_,...] = out
    return empty_tens

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def cosine_beta_schedule_nu(timesteps, s = 0.008, nu=1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) ** nu / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusionDiffSchds(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        twoD=False,
        blur=False,
        noise_scheduler_conf=None,
        multi_gpu=False

    ):
        super().__init__()
        #assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        #assert not model.random_or_learned_sinusoidal_cond
        self.model = model
        self.twoD = twoD
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        if twoD:
            self.C,self.H,self.W = image_size[0],image_size[1],image_size[2]
        else:
            self.C,self.H,self.W,self.D = image_size[0],image_size[1],image_size[2],image_size[3]

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        self.channel_inds = []
        all_betas = []
        
        for channel, schedule in noise_scheduler_conf.cahnnel_sched.items():
            self.channel_inds.append(schedule.inds)
            if schedule.beta_sched.type == 'linear':
                betas = linear_beta_schedule(timesteps)
            elif schedule.beta_sched.type == 'cosine':
                betas = cosine_beta_schedule(timesteps)
            elif schedule.beta_sched.type == 'cosine_nu':
                betas = cosine_beta_schedule_nu(timesteps, nu=schedule.beta_sched.beta_sched_params.nu)
            else:
                breakpoint()
                raise ValueError(f'unknown beta schedule {schedule.beta_sched.type}')
            all_betas.append(betas)

        alphas = [1. - betas for betas in all_betas]
        alphas_cumprod = [torch.cumprod(alphas_, dim=0) for alphas_ in alphas]
        alphas_cumprod_prev = [F.pad(alphas_cumprod_[:-1], (1, 0), value = 1.) for alphas_cumprod_ in alphas_cumprod]

        all_betas = torch.stack([betas.to(torch.float32) for betas in all_betas])
        alphas = torch.stack([alphas_.to(torch.float32) for alphas_ in alphas])
        alphas_cumprod = torch.stack([alphas_cumprod_.to(torch.float32) for alphas_cumprod_ in alphas_cumprod])
        alphas_cumprod_prev = torch.stack([alphas_cumprod_prev_.to(torch.float32) for alphas_cumprod_prev_ in alphas_cumprod_prev])

        timesteps, = all_betas[0].shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val)

        register_buffer('all_betas', all_betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = all_betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', all_betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity
        self.unnormalize_also_noise=unnormalize_also_noise

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_multithrsh(self.sqrt_recip_alphas_cumprod, t, x_t.shape, self.channel_inds, self.C) * x_t -
            extract_multithrsh(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, self.channel_inds, self.C) * noise
        )
    
    def get_loss_by_time(self, img, *args, **kwargs):
        if self.twoD:
            b, c, h, w,device = img.shape[0],img.shape[1],img.shape[2],img.shape[3], img.device
        else:
            b, c, h, w,d, device = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4], img.device
        #assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()


        # take only 10 images as I need to test for many time steps
        time = torch.linspace(5,self.num_timesteps-1,5).floor().to(device, dtype=torch.long) if self.twoD \
            else torch.linspace(5,self.num_timesteps-1,5).floor().to(device, dtype=torch.long)


        no_of_imgs = min(img.shape[0], 10)
        img = img [:no_of_imgs]
        img = img.unsqueeze(0).repeat(len(time), *[1]*len(img.shape))
        
        time = time.unsqueeze(-1).repeat(1, no_of_imgs)
        img = self.normalize(img)
        img = rearrange(img, 't b c h w -> (t b) c h w' if self.twoD else 't b c h w d -> (t b) c h w d')

        time = rearrange(time, 't b -> (t b)')
        if self.blur: time = time/self.num_timesteps # move between [0,1] for blur-diff compatibility
        loss = self.p_losses(img, time, no_reduction=True, *args, **kwargs)
        loss = rearrange(loss, '(t b) c h w -> t b c h w' if self.twoD else '(t b) c h w d -> t b c h w d', b=no_of_imgs)
        loss_foreach_time = reduce(loss, 't b c h w -> t ' if self.twoD else 't b c h w d-> t ', 'mean')
        return loss_foreach_time
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_multithrsh(self.posterior_mean_coef1, t, x_t.shape, self.channel_inds, self.C) * x_start +
            extract_multithrsh(self.posterior_mean_coef2, t, x_t.shape, self.channel_inds, self.C) * x_t
        )
        posterior_variance = extract_multithrsh(self.posterior_variance, t, x_t.shape, self.channel_inds, self.C)
        posterior_log_variance_clipped = extract_multithrsh(self.posterior_log_variance_clipped, t, x_t.shape, self.channel_inds, self.C)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):

        model_output = self.model(x, t, x_self_cond)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_specific_timesteps=None, show_every_100=False, save_all_imgs_failed=False):
        # batch_size,channels,h,W,D,device = shape, self.betas.device
        if self.twoD:
            batch_size,channels,h,W,device = *shape, self.all_betas.device
        else:
            batch_size,channels,h,W,D,device = *shape, self.all_betas.device
        #batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        fig, axs = plt.subplots(1, 14,figsize=(14, 2))
        index = 0
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            if exists(return_specific_timesteps) and return_specific_timesteps and t in return_specific_timesteps:
                imgs.append(img)
            if show_every_100 and (t % 200 == 0 or (t<100 and t%10 ==0)):
                img_ = img[0]
                if len(img_.shape) != 3: print("You need to implement 3d visualization while sampling from mdl")
                min_, max_ = torch.min(img_), torch.max(img_)
                img_ = (img_ - min_) / (max_-min_)
                axs[index].imshow(rearrange(img_.cpu(), 'c h w -> h w c'))
                axs[index].set_title(r"$x_{" + str(t) + r"}$")
                axs[index].set_xlabel("min:{}\nmax:/{}".format(round(torch.min(img).item(),3), round(torch.max(img).item(), 3)))
                index += 1
        if show_every_100:
            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            plt.tight_layout()
            plt.savefig("p_2d_7.pdf")
            exit(1)

        # TODO: right now I only unnormalize the first image from the batch in case I return multiple timesteps
        # ret = self.unnormalize(img) if (not exists(return_specific_timesteps) or not return_specific_timesteps) else torch.stack(self.unnormalize_also_noise(torch.stack(imgs, dim = 1)))
        img = unnormalize_to_zero_to_one(img)
        if save_all_imgs_failed: return img
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, return_specific_timesteps = None, show_every_100=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        imgs = [img]

        x_start = None
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None

            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                # memory efficiency consideration/ don't retain all intermediary steps images
                if exists(return_specific_timesteps) and time in return_specific_timesteps:
                    imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            # memory efficiency consideration/ don't retain all intermediary steps images
            if return_specific_timesteps:
                imgs.append(img)

        ret = img if not exists(return_specific_timesteps) else torch.stack(imgs, dim = 1)

        #ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, batch_size = 16, return_specific_timesteps = False, show_every_100=False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        shape = (batch_size, image_size[0],image_size[1],image_size[2]) if self.twoD else (batch_size, image_size[0], image_size[1], image_size[2], image_size[3])
        return sample_fn(shape, return_specific_timesteps = return_specific_timesteps,show_every_100=show_every_100)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract_multithrsh(self.sqrt_alphas_cumprod, t, x_start.shape, self.channel_inds, self.C) * x_start +
            extract_multithrsh(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape, self.channel_inds, self.C) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def plot_section(self,x_start,noise):
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.min(x_start) == -1:
            x_start = self.unnormalize(x_start)

        some_img = x_start[0]
        indices = torch.argwhere(some_img[0] == 1).reshape(-1)

        # 14
        fig, axs = plt.subplots(1, 14,figsize=(14, 2))
        section = some_img if self.twoD else some_img[0, indices[0], :, :]
        axs[0].imshow(rearrange(section.detach().cpu().numpy(), 'c h w -> h w c')) if self.twoD else \
            axs[0].imshow(section.detach().cpu().numpy())
        axs[0].set_title("x0")
        for ind, t in enumerate([5,10,20,30,50,60,70,80,90,200,400,600,800]):
            # self.sqrt_alphas_cumprod self.sqrt_one_minus_alphas_cumprod
            # self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod
            q_mean, q_var = self.sqrt_alphas_cumprod[t] * x_start[0], self.sqrt_one_minus_alphas_cumprod[t]**2
            q_mean = q_mean.flatten()
            q_var = q_var * torch.ones_like(q_mean)
            normal_q = Normal(q_mean, q_var)
            normal_01 = Normal(torch.zeros_like(q_mean), torch.ones_like(q_mean))
            x_noised = self.q_sample(x_start=x_start, t=torch.tensor([t]*x_start.shape[0]).to(device), noise=noise)
            x_noised = (x_noised-torch.min(x_noised))/(torch.max(x_noised)-torch.min(x_noised))
            axs[ind+1].imshow(rearrange(x_noised[0].detach().cpu().numpy(), 'c h w -> h w c')) if self.twoD else axs[ind+1].imshow(x_noised[0,0,indices[0], :, :].detach().cpu().numpy())
            axs[ind+1].set_title(r"x$_{"+str(t)+"}$ ")
            axs[ind+1].set_xlabel("\nKL(q,N(0,1)):\n{}".format(round(torch.mean(kl_divergence(normal_q, normal_01)).item(), 3)))
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.savefig("q_2d.pdf")

        exit(1)
        # plt.show()


    def p_losses(self, x_start, t, train_indexes=None, noise = None, grids=None, no_reduction=False):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if grids is not None:
            self.plot_section(x_start, noise)

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')

        if no_reduction:
            return loss
        if train_indexes is not None: 
            loss = loss[train_indexes[:,0],train_indexes[:,1],train_indexes[:,2],train_indexes[:,3],train_indexes[:,4]]
        else: loss = reduce(loss, 'b ... -> b (...)', 'mean')


        # mdl_params = [p for p in self.model.parameters()]
        # max_param = torch.max(torch.stack([torch.max(mdl_params_) for mdl_params_ in mdl_params]))
        # min_params = torch.min(torch.stack([torch.min(mdl_params_) for mdl_params_ in mdl_params]))
        loss = loss * extract(self.p2_loss_weight[0], t[train_indexes[:,0]] if exists(train_indexes) else t, loss.shape)
        # return loss.float().mean()
        return loss.mean()

    def forward(self, img, train_indexes=None, *args, **kwargs):
        if "validation" in kwargs and kwargs["validation"]:
            return self.get_loss_by_time(img)
        if self.twoD:
            b, c, h, w,device = img.shape[0],img.shape[1],img.shape[2],img.shape[3], img.device
        else:
            b, c, h, w,d, device = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4], img.device
        #assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, train_indexes, *args, **kwargs)

    def get_loss_by_time(self, img, *args, **kwargs):
        if self.twoD:
            b, c, h, w,device = img.shape[0],img.shape[1],img.shape[2],img.shape[3], img.device
        else:
            b, c, h, w,d, device = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4], img.device
        #assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()


        # take only 10 images as I need to test for many time steps
        time = torch.linspace(5,self.num_timesteps-1,5).floor().to(device, dtype=torch.long) if self.twoD \
            else torch.linspace(5,self.num_timesteps-1,5).floor().to(device, dtype=torch.long)


        no_of_imgs = min(img.shape[0], 10)
        img = img [:no_of_imgs]
        img = img.unsqueeze(0).repeat(len(time), *[1]*len(img.shape))
        time = time.unsqueeze(-1).repeat(1, no_of_imgs)
        img = self.normalize(img)
        img = rearrange(img, 't b c h w -> (t b) c h w' if self.twoD else 't b c h w d -> (t b) c h w d')

        time = rearrange(time, 't b -> (t b)')
        loss = self.p_losses(img, time, no_reduction=True, *args, **kwargs)
        loss = rearrange(loss, '(t b) c h w -> t b c h w' if self.twoD else '(t b) c h w d -> t b c h w d', b=no_of_imgs)
        loss_foreach_time = reduce(loss, 't b c h w -> t ' if self.twoD else 't b c h w d-> t ', 'mean')
        return loss_foreach_time

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        twoD=False
    ):
        super().__init__()
        #assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        #assert not model.random_or_learned_sinusoidal_cond
        self.model = model
        self.twoD = twoD
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        if twoD:
            self.C,self.H,self.W = image_size[0],image_size[1],image_size[2]
        else:
            self.C,self.H,self.W,self.D = image_size[0],image_size[1],image_size[2],image_size[3]

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity
        self.unnormalize_also_noise=unnormalize_also_noise

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):

        model_output = self.model(x, t, x_self_cond)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_specific_timesteps=None, show_every_100=False):
        # batch_size,channels,h,W,D,device = shape, self.betas.device
        if self.twoD:
            batch_size,channels,h,W,device = *shape, self.betas.device
        else:
            batch_size,channels,h,W,D,device = *shape, self.betas.device
        #batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        fig, axs = plt.subplots(1, 14,figsize=(14, 2))
        index = 0
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            if exists(return_specific_timesteps) and return_specific_timesteps and t in return_specific_timesteps:
                imgs.append(img)
            if show_every_100 and (t % 200 == 0 or (t<100 and t%10 ==0)):
                img_ = img[0]
                if len(img_.shape) != 3: print("You need to implement 3d visualization while sampling from mdl")
                min_, max_ = torch.min(img_), torch.max(img_)
                img_ = (img_ - min_) / (max_-min_)
                axs[index].imshow(rearrange(img_.cpu(), 'c h w -> h w c'))
                axs[index].set_title(r"$x_{" + str(t) + r"}$")
                axs[index].set_xlabel("min:{}\nmax:/{}".format(round(torch.min(img).item(),3), round(torch.max(img).item(), 3)))
                index += 1
        if show_every_100:
            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            plt.tight_layout()
            plt.savefig("p_2d_7.pdf")
            exit(1)

        # TODO: right now I only unnormalize the first image from the batch in case I return multiple timesteps
        ret = self.unnormalize(img) if (not exists(return_specific_timesteps) or not return_specific_timesteps) else torch.stack(self.unnormalize_also_noise(torch.stack(imgs, dim = 1)))
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, return_specific_timesteps = None, show_every_100=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        imgs = [img]

        x_start = None
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None

            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                # memory efficiency consideration/ don't retain all intermediary steps images
                if exists(return_specific_timesteps) and time in return_specific_timesteps:
                    imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            # memory efficiency consideration/ don't retain all intermediary steps images
            if return_specific_timesteps:
                imgs.append(img)

        ret = img if not exists(return_specific_timesteps) else torch.stack(imgs, dim = 1)

        #ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, batch_size = 16, return_specific_timesteps = False, show_every_100=False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        shape = (batch_size, image_size[0],image_size[1],image_size[2]) if self.twoD else (batch_size, image_size[0], image_size[1], image_size[2], image_size[3])
        return sample_fn(shape, return_specific_timesteps = return_specific_timesteps,show_every_100=show_every_100)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def plot_section(self,x_start,noise):
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.min(x_start) == -1:
            x_start = self.unnormalize(x_start)

        some_img = x_start[0]
        indices = torch.argwhere(some_img[0] == 1).reshape(-1)

        # 14
        fig, axs = plt.subplots(1, 14,figsize=(14, 2))
        section = some_img if self.twoD else some_img[0, indices[0], :, :]
        axs[0].imshow(rearrange(section.detach().cpu().numpy(), 'c h w -> h w c')) if self.twoD else \
            axs[0].imshow(section.detach().cpu().numpy())
        axs[0].set_title("x0")
        for ind, t in enumerate([5,10,20,30,50,60,70,80,90,200,400,600,800]):
            # self.sqrt_alphas_cumprod self.sqrt_one_minus_alphas_cumprod
            # self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod
            q_mean, q_var = self.sqrt_alphas_cumprod[t] * x_start[0], self.sqrt_one_minus_alphas_cumprod[t]**2
            q_mean = q_mean.flatten()
            q_var = q_var * torch.ones_like(q_mean)
            normal_q = Normal(q_mean, q_var)
            normal_01 = Normal(torch.zeros_like(q_mean), torch.ones_like(q_mean))
            x_noised = self.q_sample(x_start=x_start, t=torch.tensor([t]*x_start.shape[0]).to(device), noise=noise)
            x_noised = (x_noised-torch.min(x_noised))/(torch.max(x_noised)-torch.min(x_noised))
            axs[ind+1].imshow(rearrange(x_noised[0].detach().cpu().numpy(), 'c h w -> h w c')) if self.twoD else axs[ind+1].imshow(x_noised[0,0,indices[0], :, :].detach().cpu().numpy())
            axs[ind+1].set_title(r"x$_{"+str(t)+"}$ ")
            axs[ind+1].set_xlabel("\nKL(q,N(0,1)):\n{}".format(round(torch.mean(kl_divergence(normal_q, normal_01)).item(), 3)))
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.savefig("q_2d.pdf")

        exit(1)
        # plt.show()


    def p_losses(self, x_start, t, train_indexes=None, noise = None, grids=None, no_reduction=False):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if grids is not None:
            self.plot_section(x_start, noise)

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')

        if no_reduction:
            return loss
        if train_indexes is not None: 
            loss = loss[train_indexes[:,0],train_indexes[:,1],train_indexes[:,2],train_indexes[:,3],train_indexes[:,4]]
        else: loss = reduce(loss, 'b ... -> b (...)', 'mean')


        # mdl_params = [p for p in self.model.parameters()]
        # max_param = torch.max(torch.stack([torch.max(mdl_params_) for mdl_params_ in mdl_params]))
        # min_params = torch.min(torch.stack([torch.min(mdl_params_) for mdl_params_ in mdl_params]))

        loss = loss * extract(self.p2_loss_weight, t[train_indexes[:,0]] if exists(train_indexes) else t, loss.shape)
        # return loss.float().mean()
        return loss.mean()

    def forward(self, img, train_indexes=None, *args, **kwargs):
        if self.twoD:
            b, c, h, w,device = img.shape[0],img.shape[1],img.shape[2],img.shape[3], img.device
        else:
            b, c, h, w,d, device = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4], img.device
        #assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, train_indexes, *args, **kwargs)

    def get_loss_by_time(self, img, *args, **kwargs):
        if self.twoD:
            b, c, h, w,device = img.shape[0],img.shape[1],img.shape[2],img.shape[3], img.device
        else:
            b, c, h, w,d, device = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4], img.device
        #assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()


        # take only 10 images as I need to test for many time steps
        time = torch.linspace(5,self.num_timesteps-1,5).floor().to(device, dtype=torch.long) if self.twoD \
            else torch.linspace(5,self.num_timesteps-1,5).floor().to(device, dtype=torch.long)


        no_of_imgs = min(img.shape[0], 10)
        img = img [:no_of_imgs]
        img = img.unsqueeze(0).repeat(len(time), *[1]*len(img.shape))
        time = time.unsqueeze(-1).repeat(1, no_of_imgs)
        img = self.normalize(img)
        img = rearrange(img, 't b c h w -> (t b) c h w' if self.twoD else 't b c h w d -> (t b) c h w d')

        time = rearrange(time, 't b -> (t b)')
        loss = self.p_losses(img, time, no_reduction=True, *args, **kwargs)
        loss = rearrange(loss, '(t b) c h w -> t b c h w' if self.twoD else '(t b) c h w d -> t b c h w d', b=no_of_imgs)
        loss_foreach_time = reduce(loss, 't b c h w -> t ' if self.twoD else 't b c h w d-> t ', 'mean')
        return loss_foreach_time

# dataset classes

'''
class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)
'''
bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]

# Taken from https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/60fdffab78005c4e4a2759d3179dd562d7479cd9/qm9/rdkit_functions.py



def extract_number_of_atoms_and_initial_estimate(dens, cutoff):
    # TODO implement this
    return 1



# create a torch model containing N number of mean atom positions (as learnable parameters), with equal and learnable weight parameter (for the gaussian components) and a fixed variance parameter


def check_pipelin(sample):
    sample = sample * 0.2
    sample[0,1,1,1] = 1
    sample[1,1,1,2] = 1
    sample[2,1,3,1] = 1
    sample[3,1,3,1] = 1
    return sample

# trainer class 
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        data,
        results_folder,
        unique_elements,
        x,y,z,
        smiles_list,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100_000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 100,
        num_samples = 100,
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        grids=None,
        run_name = "",
        twoD=False,
        sep_bond_chn=False,
        valid_data=None,
        load_name=None,
        explicit_aromatic=False,
        explicit_hydrogen=False,
        subsample_points=-1,
        mixed_prec=False,
        compact_batch=False,
        remove_thresholding_pdf_vals=False,
        arom_cycle_channel,
        val_batch_size=10,
        no_fields=8,
        atoms_considered=None,
        backward_mdl_compat=False,
        augment_rotations=False,
        center_atm_to_grids=False,
        multi_gpu=False,
        std_atoms=0.05,
        data_type="QM9",
        threshold_bond=0.75,
        threshold_atm=0.75,
        optimize_bnd_gmm_weights=False,
        data_generator=None,
        train_dataset_args=None,
        val_dataset_args=None
    ):
        super().__init__()
        self.train_dataset_args=train_dataset_args
        self.val_dataset_args=val_dataset_args

        self.data_generator=data_generator
        self.optimize_bnd_gmm_weights=optimize_bnd_gmm_weights
        self.threshold_atm=threshold_atm
        self.threshold_bond=threshold_bond
        self.data_type=data_type
        self.no_fields=no_fields
        self.std_atoms=std_atoms


        self.model = diffusion_model
        if torch.cuda.device_count() > 1 and multi_gpu: self.model = nn.DataParallel(self.model)
        elif torch.cuda.device_count() <= 1 and multi_gpu: print("!!!WARNING!!! - multi_gpu=True but only one GPU detected. Ignoring multi_gpu flag.")
        self.multi_gpu = torch.cuda.device_count() > 1 and multi_gpu

        self.center_atm_to_grids =center_atm_to_grids
        self.augment_rotations=augment_rotations
        self.backward_mdl_compat=backward_mdl_compat
        self.run_name=run_name
        self.explicit_aromatic=explicit_aromatic
        self.grids=grids
        self.sep_bond_chn=sep_bond_chn
        self.explicit_hydrogen=explicit_hydrogen
        self.mixed_prec=mixed_prec
        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
        )

        self.accelerator.native_amp = amp

        # model

        self.channel_atoms = unique_elements

        self.x_grid = x
        self.y_grid = y
        self.z_grid = z
        
        self.dataset_smiles = smiles_list
        self.twoD=twoD
        self.remove_thresholding_pdf_vals=remove_thresholding_pdf_vals

        # InceptionV3 for fid-score computation

        self.inception_v3 = None

        # if calculate_fid and self.twoD:
        #    breakpoint()
        #    assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        #    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        #    self.inception_v3 = InceptionV3([block_idx])
        #    self.inception_v3.to(self.device)

        # sampling and training hyperparameters

        #assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        self.compact_batch=compact_batch
        if compact_batch:
            x_flat,y_flat,z_flat = torch.tensor(self.x_grid.flatten(),device=self.device, dtype=torch.float32), \
            torch.tensor(self.y_grid.flatten(),device=self.device,dtype=torch.float32), torch.tensor(self.z_grid.flatten(),device=self.device,dtype=torch.float32)
            self.grid_inputs = torch.stack([x_flat,y_flat,z_flat],dim=1)        

        self.remove_thresholding_pdf_vals=remove_thresholding_pdf_vals
        #self.image_size = diffusion_model.image_size

        # dataset and dataloader

        #self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        # if torch.cuda.is_available():
        #     dl = DataLoader(data, batch_size = train_batch_size, shuffle = True, pin_memory = False, num_workers = 0)
        # else:
        self.subsample_points=subsample_points


        no_atms, no_bonds, aromatic_circles = len(atoms_considered), 3 + explicit_aromatic, arom_cycle_channel*2
        if self.compact_batch:
            self.dl = DataLoader(data, batch_size=train_batch_size, shuffle=True, pin_memory=False, num_workers=1, collate_fn=lambda x: collate_fn_general_noatmNo(x, no_channels=no_bonds+no_atms+aromatic_circles, atms_last_ind=no_atms-1), drop_last=True)
            self.dl_val = DataLoader(valid_data, batch_size=val_batch_size, shuffle=True, pin_memory=False, num_workers=1, collate_fn=lambda x: collate_fn_general_noatmNo(x, no_channels=no_bonds+no_atms+aromatic_circles, atms_last_ind=no_atms-1), drop_last=True)
            # self.dl = DataLoader(data, batch_size=train_batch_size, shuffle=True, pin_memory=False, num_workers=1, collate_fn=collate_fn_compact_expl_ah if self.explicit_aromatic else collate_fn_compact_expl_h)
            # self.dl_val = DataLoader(valid_data, batch_size=10, shuffle=True, pin_memory=False, num_workers=1, collate_fn=collate_fn_compact_expl_ah if self.explicit_aromatic else collate_fn_compact_expl_h)
        else:
            if subsample_points != -1: self.dl = DataLoader(data, batch_size=train_batch_size, shuffle=True, pin_memory=False, num_workers=8 if exists(valid_data) else 9, collate_fn=collate_subsample_points)
            else: self.dl = DataLoader(data, batch_size=train_batch_size, shuffle=True, pin_memory=False, num_workers=8 if exists(valid_data) else 9)
            self.dl_val = DataLoader(valid_data, batch_size=10, shuffle=True, pin_memory=False, num_workers=1) if exists(valid_data) else None

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        # if self.accelerator.is_main_process:
        self.ema = EMA(diffusion_model.cpu(), beta = ema_decay, update_every = ema_update_every)
        self.ema.to(self.device)

        # self.results_folder = Path(results_folder)
        self.results_folder = results_folder
        # self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        # self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        if load_name is not None: self.load(load_name)

    @property
    def device(self):
        return self.accelerator.device
   
    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder + "/" + 'model-{}'.format(milestone) + self.run_name + '.pt'))
        if exists(self.data_generator) and self.data_generator.epoch_conformers_train:
            save_smpls_path = os.path.join(self.data_path, self.run_name +"_sample_indinces.bin")
            pickle.dump([self.data_generator.all_conf_inds, self.data_generator.current_conf_index], open(save_smpls_path, "wb"))

    def load(self, model_name):
        accelerator = self.accelerator
        device = accelerator.device

        model_name = f'{model_name}.pt' if 'pt' not in model_name else model_name

        data = torch.load(str(f'{self.results_folder}/{model_name}'), map_location=device)

        # model = self.accelerator.unwrap_model(self.model)
        if self.multi_gpu and type(self.model) != nn.DataParallel: self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        print(f"Model {model_name} has been loaded")
        logging.info(f"Model {model_name} has been loaded")


    @torch.no_grad()
    def calculate_activation_statistics(self, samples):
        features = self.inception_v3(samples)[0]
        features = rearrange(features, '... 1 1 -> ...')

        mu = torch.mean(features, dim = 0).cpu()
        sigma = torch.cov(features).cpu()
        return mu, sigma

    def fid_score(self, real_samples, fake_samples):
        min_batch = min(real_samples.shape[0], fake_samples.shape[0])
        real_samples, fake_samples = map(lambda t: t[:min_batch], (real_samples, fake_samples))

        m1, s1 = self.calculate_activation_statistics(real_samples)
        m2, s2 = self.calculate_activation_statistics(fake_samples)

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

    def train(self):
        loss_by_time = {t: 0 for t in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800]} if self.twoD \
            else {t: 0 for t in [5, 50, 100, 400, 800]}
        accelerator = self.accelerator
        # device = accelerator.device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        total_loss = 0.
        while self.step < self.train_num_steps:
            for data in self.dl:
                if self.compact_batch:
                    coords, inds, N_list, bnds = data
                    if self.augment_rotations: coords, inds, N_list, bnds, removed = self.rotate_coords(coords, inds, N_list, bnds)
                    if self.center_atm_to_grids: coords = self.center_positions_to_grids(coords, inds, N_list, bnds)
                    grid_shapes = [self.model.module.H, self.model.module.W, self.model.module.D] if self.multi_gpu else [self.model.H, self.model.W, self.model.D]
                    inp = create_gaussian_batch_pdf_values(x=self.grid_inputs, coords=coords, N_list=N_list, std=self.std_atoms, device=device, gaussian_indices=inds,
                                                            no_fields =self.no_fields, grid_shapes=grid_shapes,
                                                            threshold_vlals= not self.remove_thresholding_pdf_vals, backward_mdl_compat=self.backward_mdl_compat)
                else: inp = data

                # * visualize inputs
                # atm_symb_vis, atm_pos_vis, actual_bnds_vis = transform_training_batch_into_visualizable(coords, inds, N_list,bnds, self.explicit_aromatic,self.explicit_hydrogen,self.no_fields, data_type=self.data_type)
                # for i in range(len(inp)):
                #     visualize_mol(plot_bnd=4, field=inp[i].cpu(), atm_pos=atm_pos_vis[i].cpu().numpy(), atm_symb=atm_symb_vis[i],actual_bnd=actual_bnds_vis[i], x_grid=self.x_grid,y_grid=self.y_grid, z_grid=self.z_grid, threshold=0.3)
                if self.subsample_points != -1: inp, train_indexes = inp
                else: train_indexes = None
                if self.mixed_prec:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        inp = inp.to(device)
                        loss = self.model(inp,grids=self.grids, train_indexes=train_indexes)/self.gradient_accumulate_every
                else:
                    inp = inp.to(device)
                    loss = self.model(inp,grids=self.grids, train_indexes=train_indexes)/self.gradient_accumulate_every
                loss = loss.mean()
                total_loss += loss.item()
                loss.backward()

                if self.step % self.gradient_accumulate_every == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.opt.step()
                    self.opt.zero_grad()

                if self.step % (20*self.gradient_accumulate_every) == 0:
                    loss_by_time = {t: 0 for t in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800]} \
                        if self.twoD else {t: 0 for t in [5, 50, 100, 400, 800]}
                    # data_val= next(iter(self.dl_val)).to(self.device) if exists(self.dl_val) else data
                    data_val= next(iter(self.dl_val)) if exists(self.dl_val) else data
                    if self.compact_batch:
                        coords, inds, N_list,bnds = data_val
                        grid_shapes = [self.model.module.H, self.model.module.W, self.model.module.D] if self.multi_gpu else [self.model.H, self.model.W, self.model.D]
                        inp_val = create_gaussian_batch_pdf_values(x=self.grid_inputs, coords=coords, N_list=N_list, std=self.std_atoms, device=device, gaussian_indices=inds,
                                                                no_fields = self.no_fields, grid_shapes=grid_shapes,
                                                                threshold_vlals= not self.remove_thresholding_pdf_vals,backward_mdl_compat=self.backward_mdl_compat)
                    else: inp_val = data_val
                    with torch.no_grad():
                        # TODO should this be also self.model.eval/train before/after (for block norm)? - mb not/no batch stats
                        loss_by_time_ = self.model(inp_val, validation=True)
                        loss_by_time = {k:v+loss_by_time_[ind] for ind, (k,v) in enumerate(loss_by_time.items())}

                    print("{}: Step {}: loss {:.6f}".format(get_current_datetime(), self.step, total_loss/10 if self.step != 0 else total_loss * self.gradient_accumulate_every))
                    logging.info("{}: Step {}: loss {:.6f}".format(get_current_datetime(), self.step, total_loss/10 if self.step != 0 else total_loss * self.gradient_accumulate_every))
                    loss_by_time_string = "" if not exists(self.dl_val) else "val loss: "
                    for k,v in loss_by_time.items():
                        loss_by_time_string += "{}:{:.6f} |".format(k,v.item())
                    print(loss_by_time_string)
                    logging.info(loss_by_time_string)
                    total_loss = 0.

                self.step += 1
                self.ema.update()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    milestone = self.step // self.save_and_sample_every
                    self.save(milestone)

                    self.ema.ema_model.eval()

                    with torch.no_grad():
                        # batches = num_to_groups(self.num_samples, self.batch_size)
                        batches = num_to_groups(10, 10)
                        if self.multi_gpu: all_images_list = list(map(lambda n: self.model.module.sample(batch_size=n), batches))
                        else: all_images_list = list(map(lambda n: self.model.sample(batch_size=n), batches))
                    training_images = self.model.module.twoD if self.multi_gpu else self.model.twoD
                    if not training_images:
                        all_images = torch.cat(all_images_list, dim=0)
                        try:
                            if self.sep_bond_chn:
                                val, unq, nov, _ = fit_pred_field_sep_chn_batch(all_images, 0.1, self.x_grid, self.y_grid, self.z_grid,
                                            len(self.channel_atoms), self.channel_atoms, self.dataset_smiles,
                                            noise_std=0.0, normalize01=True,explicit_aromatic=self.explicit_aromatic,
                                            explicit_hydrogen=self.explicit_hydrogen, optimize_bnd_gmm_weights=self.optimize_bnd_gmm_weights,
                                            threshold_bond=self.threshold_bond, threshold_atm=self.threshold_atm, atm_symb_bnd_by_channel=self.unique_atms_considered if "GEOM" in self.data_type else None,)
                            else:
                                val, unq, nov, _ = fit_pred_field(all_images, 0.1, self.x_grid, self.y_grid, self.z_grid,
                                                            len(self.channel_atoms), self.channel_atoms, self.dataset_smiles)
                        except Exception as e:
                            print("!!!Error in fitting field!!!:\n{}".format(str(e)))
                            logging.info("!!!Error in fitting field!!!:\n{}".format(str(e)))
                            val, unq, nov = 0, 0, 0

                        print(f"Validity of molecules: {val * 100 :.2f}%")
                        logging.info(f"Validity of molecules: {val * 100 :.2f}%")
                        if val>0:
                            print(f"Uniqueness of valid molecules: {unq * 100 :.2f}%")
                            logging.info(f"Uniqueness of valid molecules: {unq * 100 :.2f}%")
                            print(f"Novelty of valid molecules: {nov * 100 :.2f}%")
                            logging.info(f"Novelty of valid molecules: {nov * 100 :.2f}%")
                    
                    

        # with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
        #
        #     while self.step < self.train_num_steps:
        #         for data in self.dl:
        #             total_loss = 0.
        #             for _ in range(self.gradient_accumulate_every):
        #                 # data = next(self.dl).to(device)
        #                 data = data.to(device)
        #
        #
        #                 with self.accelerator.autocast():
        #                     loss = self.model(data)
        #                     loss = loss / self.gradient_accumulate_every
        #                     total_loss += loss.item()
        #
        #                 self.accelerator.backward(loss)
        #
        #             accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
        #             pbar.set_description(f'loss: {total_loss:.4f}')
        #             accelerator.wait_for_everyone()
        #
        #             self.opt.step()
        #             self.opt.zero_grad()
        #
        #             accelerator.wait_for_everyone()
        #
        #             if self.step % 100 == 0:
        #                 print("Step {}: loss {}".format(self.step, total_loss))
        #
        #             self.step += 1
        #             if accelerator.is_main_process:
        #                 self.ema.update()
        #
        #                 if self.step != 0 and self.step % self.save_and_sample_every == 0:
        #                     self.ema.ema_model.eval()
        #
        #                     with torch.no_grad():
        #                         milestone = self.step // self.save_and_sample_every
        #                         batches = num_to_groups(self.num_samples, self.batch_size)
        #                         all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
        #
        #                     #print(len(all_images_list))
        #                     all_images = torch.cat(all_images_list, dim = 0)
        #                     val,unq,nov = fit_pred_field(all_images,0.1,self.x_grid,self.y_grid,self.z_grid,len(self.channel_atoms),self.channel_atoms,self.dataset_smiles)
        #                     print(f"Validity of molecules: {val * 100 :.2f}%")
        #                     if val>0:
        #                         print(f"Uniqueness of valid molecules: {unq * 100 :.2f}%")
        #                         print(f"Novelty of valid molecules: {nov * 100 :.2f}%")
        #
        #                     #utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
        #                     self.save(milestone)
        #
        #                     # whether to calculate fid
        #
        #                     #if exists(self.inception_v3):
        #                     #    fid_score = self.fid_score(real_samples = data, fake_samples = all_images)
        #                     #    accelerator.print(f'fid_score: {fid_score}')
        #
        #             pbar.update(1)

        accelerator.print('training complete')
