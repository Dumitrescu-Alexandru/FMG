import os, pickle
from scipy.spatial.transform import Rotation
import numpy as np
from visualize_utils import visualize_mol
from pytorch_fid.inception import InceptionV3
from test_utils import fit_pred_field,fit_pred_field_sep_chn_batch
from pytorch_fid.fid_score import calculate_frechet_distance
from utils import create_gaussian_batch_pdf_values, collate_fn_compact_expl_h_clsfreeguid,collate_fn_compact_expl_ah_clsfreeguid, collate_subsample_points, plot_channel, transform_training_batch_into_visualizable, get_next_ep_data, collate_fn_compact_expl_h_clsfreeguid_geom, collate_fn_general, collate_fn_general_cond_var
from ema_pytorch import EMA
import logging
logging.getLogger('some_logger')
from torch_dct import dct_3d, idct_3d
from collections import OrderedDict
import math
from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader
import copy

from accelerate import Accelerator
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from denoising_diffusion_pytorch.version import __version__

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm
import datetime

def get_current_datetime():
    return str(datetime.datetime.now()).split(".")[0][2:]

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions




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

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def unnormalize_also_noise(imgs):
    print("I should not be here?")
    imgs = [(img - img.min())/(img.max()-img.min()) for img in imgs[0]]
    return imgs


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

# classifier free guidance functions

def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

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
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


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

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

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

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock3D(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, classes_emb_dim = None, cond_var_dim=None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim) + int(default(cond_var_dim, 0)), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        self.block1 = Block3D(dim, dim_out, groups = groups)
        self.block2 = Block3D(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, class_emb = None, cond_var = None):

        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb, cond_var)))
            cond_emb = torch.cat(cond_emb, dim = -1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 1 1')
            scale_shift = cond_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, classes_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, class_emb = None):

        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim = -1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 1')
            scale_shift = cond_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LayerNorm3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

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

class Attention(nn.Module):
    def __init__(self, dim, heads = 12, dim_head = 32):
        super().__init__()
        dim_head = dim // heads
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


# model

class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        num_classes,
        cond_drop_prob = 0.5,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        attention_grid=None,
        add_pe=False,
        legacy_attention=False,
        cond_var=None,
        norm_factors=None,
        remove_null_cond_emb=False
    ):
        super().__init__()

        # classifier free guidance stuff
        self.cond_drop_prob = cond_drop_prob

        self.cond_var = cond_var
        self.norm_factors=norm_factors

        # determine dimensions

        if not exists(attention_grid): attention_grid = 32// (2**(len(dim_mults)-1))
        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)

        # TODO is 7 too large as initi conv? what about 5?
        self.init_conv = nn.Conv3d(input_channels, init_dim, 7, padding = 3)

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

        if not remove_null_cond_emb:
            self.null_conditioning_emb = nn.Parameter(torch.randn(dim)) #if norm_factors is not None else None
        if self.cond_var:
            cond_var_dim = 4 * dim * len(cond_var)
            self.cond_var_mlp = nn.Sequential(sinu_pos_emb, nn.Linear(fourier_dim, cond_var_dim), nn.GELU(), nn.Linear(cond_var_dim, cond_var_dim))
        else:
            cond_var_dim = None

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # class embeddings

        self.classes_emb = nn.Embedding(num_classes, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))

        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim, cond_var_dim=cond_var_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim, cond_var_dim=cond_var_dim),
                Residual(PreNorm3D(dim_in, LinearAttention3D(dim_in))),
                Downsample3D(dim_in, dim_out) if not is_last else nn.Conv3d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim, cond_var_dim=cond_var_dim)
        if legacy_attention:
            self.mid_attn = Residual( PreNorm3D(mid_dim, Attention3D_legacy(mid_dim))    )
        else:
            self.mid_attn = Residual( PreNorm3D(mid_dim, Attention3D(mid_dim, res_dim=attention_grid, wo_pe=not add_pe))    )
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim, cond_var_dim=cond_var_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim, cond_var_dim=cond_var_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim, cond_var_dim=cond_var_dim),
                Residual(PreNorm3D(dim_out, LinearAttention3D(dim_out))),
                Upsample3D(dim_out, dim_in) if not is_last else  nn.Conv3d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim, cond_var_dim=cond_var_dim)
        self.final_conv = nn.Conv3d(dim, self.out_dim, 1)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits
        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        classes,
        cond_var,
        cond_drop_prob = None
    ):
        batch, device = x.shape[0], x.device


        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # derive condition, with condition dropout for classifier free guidance        
        classes_emb = self.classes_emb(classes)

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)
            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )
            # * TODO: Remove some embeddings from cond_var

        if cond_var is not None:
            cond_var = cond_var.to(device)
            cond_var = self.cond_var_mlp(cond_var).squeeze()
        c = self.classes_mlp(classes_emb)
        # unet

        x = self.init_conv(x)
        r = x.clone()
        if hasattr(self, 'train_inf_tmstp_fact') and self.train_inf_tmstp_fact is not None:
            time  = time / self.train_inf_tmstp_fact
        t = self.time_mlp(time)

        h = []



        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c, cond_var)
            h.append(x)

            x = block2(x, t, c, cond_var)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c, cond_var)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c, cond_var)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, c, cond_var)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, c, cond_var)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, c, cond_var)
        return self.final_conv(x)

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        num_classes,
        cond_drop_prob = 0.5,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
    ):
        super().__init__()

        # classifier free guidance stuff

        self.cond_drop_prob = cond_drop_prob

        # determine dimensions

        self.channels = channels
        input_channels = channels

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

        # class embeddings

        self.classes_emb = nn.Embedding(num_classes, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))

        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        classes,
        cond_drop_prob = None
    ):
        batch, device = x.shape[0], x.device

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # derive condition, with condition dropout for classifier free guidance        

        classes_emb = self.classes_emb(classes)

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )

        c = self.classes_mlp(classes_emb)

        # unet

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)

# gaussian diffusion trainer class

def normalized_noise(img):
    max_vals = img.flatten(1).max(dim=1)[0]
    min_vals = img.flatten(1).min(dim=1)[0]
    img_ = (img - min_vals[:,None,None,None,None]) / (max_vals[:,None,None,None,None] - min_vals[:,None,None,None,None])
    return img_.detach().cpu().numpy()


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
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
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
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.,
        twoD=False,
        auto_normalize=True,
        blur=False,
        noise_scheduler_conf=None,
        multi_gpu=False
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.twoD = twoD
        if twoD:
            self.C, self.H, self.W = image_size[0],image_size[1],image_size[2]
        else:
            self.C, self.H, self.W, self.D = image_size[0],image_size[1],image_size[2],image_size[3]

        self.model = model
        self.channels = self.model.channels
        self.blur = blur

        self.image_size = image_size

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


        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity
        self.unnormalize_also_noise=unnormalize_also_noise



    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_multithrsh(self.sqrt_recip_alphas_cumprod, t, x_t.shape, self.channel_inds, self.C) * x_t -
            extract_multithrsh(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, self.channel_inds, self.C) * noise
        )
    
    def get_loss_by_time(self, img, classes=None, cond_var=None, *args, **kwargs):
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
        classes = classes.unsqueeze(0).repeat(len(time), 1)
        if exists(cond_var): cond_var = cond_var.unsqueeze(0).repeat(len(time), 1, 1)
        
        time = time.unsqueeze(-1).repeat(1, no_of_imgs)
        img = self.normalize(img)
        img = rearrange(img, 't b c h w -> (t b) c h w' if self.twoD else 't b c h w d -> (t b) c h w d')

        time = rearrange(time, 't b -> (t b)')
        classes = rearrange(classes, 't b -> (t b)')
        if exists(cond_var): cond_var = rearrange(cond_var, 't b c -> (t b) c')
        if self.blur: time = time/self.num_timesteps # move between [0,1] for blur-diff compatibility
        loss = self.p_losses(img, time, classes=classes, no_reduction=True, cond_var=cond_var, *args, **kwargs)
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

    def model_predictions(self, x, t, classes, cond_scale = 3., clip_x_start = False, cond_var=None):
        model_output = self.model.forward_with_cond_scale(x, t, classes, cond_scale = cond_scale, cond_var=cond_var)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

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

    def p_mean_variance(self, x, t, classes, cond_scale, clip_denoised = True, cond_var=None):
        preds = self.model_predictions(x, t, classes, cond_scale, cond_var=cond_var)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample_blur(self, x, t: int, classes, cond_scale = 3., clip_denoised = True, delta=1e-8):
        b, c, h, w, d, device, img_size, = *x.shape, x.device, self.image_size
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        alpha_s, sigma_s = self.get_alpha_sigma((batched_times-1)/self.num_timesteps, h, device)
        alpha_t, sigma_t = self.get_alpha_sigma(batched_times/self.num_timesteps, h, device)
        sigma_s, sigma_t =sigma_s[:, None, None, None, None], sigma_t[:, None, None, None, None]

        # comput helpful coefficients
        alpha_ts = alpha_t / alpha_s
        alpha_st = 1 / alpha_ts
        sigma2_ts = ( sigma_t ** 2 - alpha_ts ** 2 * sigma_s ** 2)

        # compute sigma_{t->s}/ below like in pseudocode, commented like in paper
        sigma_denoise = 1 / torch.clip( 1/torch.clip(sigma_s **2, min=delta) + 1/torch.clip(sigma_t ** 2/alpha_ts **2 - sigma_s **2, min=delta), min=delta)
        # inv_sigma_denoise = torch.clip(1/sigma_s ** 2,min=delta) + alpha_ts** 2/torch.clip(sigma2_ts, min=delta)
        # sigma_denoise_2 = (torch.clip(1/inv_sigma_denoise, min=delta))
        # coefficients
        coeff_term1 = alpha_ts * sigma_denoise / torch.clip(sigma2_ts, min=delta)
        coeff_term2 = alpha_s * sigma_denoise / torch.clip(sigma_s **2, min=delta)

        # hat_eps = self.model.forward_with_cond_scale(x, t, classes, cond_scale = cond_scale)

        hat_eps = self.model.forward_with_cond_scale(x, batched_times, classes, cond_scale = cond_scale)

        u_t = dct_3d(x)
        term1 = idct_3d(coeff_term1 * u_t)
        term2 = idct_3d(coeff_term2 * u_t - sigma_t * dct_3d(hat_eps))
        mu_denoise = term1 + term2

        eps = torch.randn_like(mu_denoise)
        return mu_denoise + idct_3d(torch.sqrt(sigma_denoise) * eps), None

    @torch.no_grad()
    def p_sample(self, x, t: int, classes, cond_scale = 3., clip_denoised = True, cond_var=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, classes = classes, cond_scale = cond_scale, clip_denoised = clip_denoised, cond_var=cond_var)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_scale = 3., save_all_imgs=False,cond_var=None):
        batch, device = shape[0], self.all_betas.device

        img = torch.randn(shape, device=device)

        x_start = None
        all_imgs = []
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, x_start = self.p_sample_blur(img, t, classes, cond_scale) if self.blur else self.p_sample(img, t, classes, cond_scale, cond_var=cond_var)
            if save_all_imgs: all_imgs.append(normalized_noise(img))

        img = unnormalize_to_zero_to_one(img)
        if save_all_imgs: return img, all_imgs
        return img

    @torch.no_grad()
    def ddim_sample(self, classes, shape, cond_scale = 3., clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, cond_scale = cond_scale, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, classes, cond_scale = 3., save_all_imgs=False, cond_var=None):
        batch_size, image_size, channels = classes.shape[0], self.image_size, self.channels
        image_size = list(image_size)
        image_size.insert(0, batch_size)
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(classes, tuple(image_size), cond_scale, save_all_imgs=save_all_imgs, cond_var=cond_var)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

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

    def p_losses(self, x_start, t, *, classes, noise = None, grids=None, train_indexes=None, no_reduction=False, cond_var=None):
        if self.twoD: b, c, h, w = x_start.shape
        else: b, c, h, w, d  = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise) if not self.blur else self.diffuse_blur(x_start, t, noise)[0]
        # ! in def train from Trainer set x_grid, .. to global to visualize below
        # x_ = [[x_b.unsqueeze(0), torch.randn_like(x_b).unsqueeze(0)] for x_b in x]
        # noised_imgs.insert(0,[inp[0].unsqueeze(0), torch.zeros_like(inp[0])])
        # visualize_noisy_fields(x_, [0]*100, False, True, coords=torch.tensor([1,2,3]*100).reshape(100,3), x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, t=t)

        if self.blur: t = (t * self.num_timesteps).long() # move from [0,1] to indices

        # predict and take gradient step
        model_out = self.model(x, t, classes, cond_var)

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
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # TODO right now I may use different loss weights for different channels (w different beta schedules),
        # but impl. below doesn't care about that (does it make sense to have it separate anyways?)
        loss = loss * extract(self.p2_loss_weight[0], t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        if "validation" in kwargs and kwargs["validation"]:
            return self.get_loss_by_time(img, kwargs['classes'], kwargs['cond_var'])

        if self.twoD:
            b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
            assert h == self.H and w == self.W, f'height and width of image must be {img_size}'
        else:
            b, c, h, w, d, device, img_size, = *img.shape, img.device, self.image_size
            assert h == self.H and w == self.W and d == self.D, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        if self.blur: t = t/self.num_timesteps # move between [0,1] for blur-diff compatibility
        img = normalize_to_neg_one_to_one(img)

        return self.p_losses(img, t, *args, **kwargs)
    


    def get_frequency_scaling (self, t , min_scale =0.001, image_size=32, device=None):
        sigma_blur_max = 20
        # compute dissipation time
        sigma_blur = sigma_blur_max * torch.sin ( t * np.pi / 2) ** 2
        dissipation_time = sigma_blur ** 2 / 2
        # compute frequencies
        # freq = torch.arange(1,33).reshape(32,1) ** 2 / 32 **2
        freqs = np.pi*torch.linspace(0,image_size-1,image_size, device=device)/(image_size-1)
        L = freqs[None, None, :, None, None]**2 + freqs[None, None, None, :, None]**2 + freqs[None, None, None, None, :]**2
        L = L.repeat(t.shape[0], 1, 1, 1, 1)
        dissipation_time = dissipation_time[:, None, None, None, None]
        # compute scaling for frequencies
        scaling = torch.exp(-L * dissipation_time) * (1 - min_scale)
        return scaling + min_scale

    def get_alpha_sigma(self, t, image_size, device):
        freq_scaling = self.get_frequency_scaling(t, image_size=image_size, device=device)
        a, sigma = self.get_noise_scaling_cosine(t, device=device)
        alpha = a[:, None, None, None, None] * freq_scaling
        return alpha, sigma

    def get_noise_scaling_cosine (self, t , logsnr_min = -10 , logsnr_max =10, device=None):
        limit_max = torch.arctan ( torch.exp ( torch.tensor([-0.5 * logsnr_max], device=device) ))
        limit_min = torch.arctan ( torch.exp ( torch.tensor([-0.5 * logsnr_min], device=device) )) - limit_max
        logsnr = -2 * torch.log ( torch.tan ( limit_min * t + limit_max ))
        # Transform logsnr to a , sigma .
        return torch.sqrt ( torch.sigmoid ( logsnr )) , torch.sqrt ( torch.sigmoid ( - logsnr ))


    def diffuse_blur(self, x,t, noise=None):
        x_freq = dct_3d(x)
        b, c, h, w, d, device, img_size, = *x.shape, x.device, self.image_size
        alpha, sigma = self.get_alpha_sigma(t, h, device) # ! i for now assuem all dims are equal = h; 
        eps = noise if exists(noise) else torch.randn_like(x) 
        z_t = idct_3d(x_freq * alpha) +  sigma[:, None, None, None, None] * eps
        return z_t, eps




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
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.,
        twoD=False,
        auto_normalize=True,
        blur=False
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.twoD = twoD
        if twoD:
            self.C, self.H, self.W = image_size[0],image_size[1],image_size[2]
        else:
            self.C, self.H, self.W, self.D = image_size[0],image_size[1],image_size[2],image_size[3]

        self.model = model
        self.channels = self.model.channels
        self.blur = blur

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

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


        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity
        self.unnormalize_also_noise=unnormalize_also_noise

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def get_loss_by_time(self, img, classes=None, *args, **kwargs):
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
        classes = classes.unsqueeze(0).repeat(len(time), 1)
        
        time = time.unsqueeze(-1).repeat(1, no_of_imgs)
        img = self.normalize(img)
        img = rearrange(img, 't b c h w -> (t b) c h w' if self.twoD else 't b c h w d -> (t b) c h w d')

        time = rearrange(time, 't b -> (t b)')
        classes = rearrange(classes, 't b -> (t b)')
        if self.blur: time = time/self.num_timesteps # move between [0,1] for blur-diff compatibility
        loss = self.p_losses(img, time, classes=classes, no_reduction=True, *args, **kwargs)
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
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, classes, cond_scale = 3., clip_x_start = False, cond_var = None):
        model_output = self.model.forward_with_cond_scale(x, t, classes, cond_scale = cond_scale, cond_var=cond_var)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

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

    def p_mean_variance(self, x, t, classes, cond_scale, clip_denoised = True, cond_var=None):
        preds = self.model_predictions(x, t, classes, cond_scale, cond_var=cond_var)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)

        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample_blur(self, x, t: int, classes, cond_scale = 3., clip_denoised = True, delta=1e-8):
        b, c, h, w, d, device, img_size, = *x.shape, x.device, self.image_size
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        alpha_s, sigma_s = self.get_alpha_sigma((batched_times-1)/self.num_timesteps, h, device)
        alpha_t, sigma_t = self.get_alpha_sigma(batched_times/self.num_timesteps, h, device)
        sigma_s, sigma_t =sigma_s[:, None, None, None, None], sigma_t[:, None, None, None, None]

        # comput helpful coefficients
        alpha_ts = alpha_t / alpha_s
        alpha_st = 1 / alpha_ts
        sigma2_ts = ( sigma_t ** 2 - alpha_ts ** 2 * sigma_s ** 2)

        # compute sigma_{t->s}/ below like in pseudocode, commented like in paper
        sigma_denoise = 1 / torch.clip( 1/torch.clip(sigma_s **2, min=delta) + 1/torch.clip(sigma_t ** 2/alpha_ts **2 - sigma_s **2, min=delta), min=delta)
        # inv_sigma_denoise = torch.clip(1/sigma_s ** 2,min=delta) + alpha_ts** 2/torch.clip(sigma2_ts, min=delta)
        # sigma_denoise_2 = (torch.clip(1/inv_sigma_denoise, min=delta))
        # coefficients
        coeff_term1 = alpha_ts * sigma_denoise / torch.clip(sigma2_ts, min=delta)
        coeff_term2 = alpha_s * sigma_denoise / torch.clip(sigma_s **2, min=delta)

        # hat_eps = self.model.forward_with_cond_scale(x, t, classes, cond_scale = cond_scale)

        hat_eps = self.model.forward_with_cond_scale(x, batched_times, classes, cond_scale = cond_scale)

        u_t = dct_3d(x)
        term1 = idct_3d(coeff_term1 * u_t)
        term2 = idct_3d(coeff_term2 * u_t - sigma_t * dct_3d(hat_eps))
        mu_denoise = term1 + term2

        eps = torch.randn_like(mu_denoise)
        return mu_denoise + idct_3d(torch.sqrt(sigma_denoise) * eps), None

    @torch.no_grad()
    def p_sample(self, x, t: int, classes, cond_scale = 3., clip_denoised = True, cond_var=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, classes = classes, cond_scale = cond_scale, clip_denoised = clip_denoised, cond_var=cond_var)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_scale = 3., save_all_imgs=False, cond_var=None):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None
        all_imgs = []
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, x_start = self.p_sample_blur(img, t, classes, cond_scale) if self.blur else self.p_sample(img, t, classes, cond_scale,cond_var=cond_var)
            if save_all_imgs: all_imgs.append(normalized_noise(img))

        img = unnormalize_to_zero_to_one(img)
        if save_all_imgs: return img, all_imgs
        return img

    @torch.no_grad()
    def ddim_sample(self, classes, shape, cond_scale = 3., clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, cond_scale = cond_scale, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, classes, cond_scale = 3., save_all_imgs=False, cond_var=None):
        batch_size, image_size, channels = classes.shape[0], self.image_size, self.channels
        image_size = list(image_size)
        image_size.insert(0, batch_size)
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(classes, tuple(image_size), cond_scale, save_all_imgs=save_all_imgs, conv_var=cond_var)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

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

    def unpack_mol_info(self, coords_pded=None, inds_pded=None, N_list_pded=None, device=None, rotations=None):

        # * order will be [batch_dim, rotations, ...]
        no_rotations = rotations[0].shape[-1] if rotations is not None else 1
        N_list_repeated = []
        if exists(coords_pded) and exists(inds_pded) and exists(N_list_pded):
            N_list_ = [n_list[n_list>=0].detach().cpu().numpy().tolist() for n_list in N_list_pded]
            for batch_elem in range(len(N_list_)): N_list_repeated.extend([N_list_[batch_elem]] * no_rotations)
            molecule_limits = np.cumsum([sum(n_list) for n_list in N_list_])
            molecule_limits = np.insert(molecule_limits, 0, 0)

            inds = inds_pded[inds_pded>-1]
            # inds might be on e.g. gpu 3, so starting index needs to be reverted to 0
            starting_ind = np.ceil(inds.min().item()/self.C) * self.C
            inds -= starting_ind
            inds_per_molec = [inds[molecule_limits[i]:molecule_limits[i+1]]%self.C for i in range(len(molecule_limits)-1)]
            inds_per_molec_repeated = []
            for i,inds_ in enumerate(inds_per_molec):
                current_inds = inds_.reshape(1,-1).repeat(no_rotations,1) + torch.range(0, no_rotations-1, device=device).reshape(-1,1) * self.C + i * self.C * no_rotations
                inds_per_molec_repeated.append(current_inds.reshape(-1))
            inds_per_molec_repeated = torch.cat(inds_per_molec_repeated).to(dtype=torch.int)

            coords = coords_pded[coords_pded > -100].reshape(-1,3)
            coords_per_molec = [coords[molecule_limits[i]:molecule_limits[i+1]] for i in range(len(molecule_limits)-1)]
            coords_pre_molec_repeated = []
            for batch_elem in range(len(N_list_)): coords_pre_molec_repeated.append(coords_per_molec[batch_elem].reshape(1,-1,3).repeat(no_rotations,1,1))

            return coords_pre_molec_repeated, inds_per_molec_repeated, N_list_repeated, no_rotations
        else:
            return None, None, None, None

    def p_losses(self, x_start, t, *, classes, noise = None, grids=None, train_indexes=None, no_reduction=False,coords_pded=None, inds_pded=None, N_list_pded=None, rotations=None, target=None, only_min_activ=False):
        if type(grids) == np.ndarray: grids = torch.tensor(grids, device=x_start.device)
        coords, inds, N_list, no_rotations = self.unpack_mol_info(coords_pded, inds_pded, N_list_pded, device=x_start.device, rotations=rotations)


        if self.twoD: b, c, h, w = x_start.shape
        else: b, c, h, w, d  = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise) if not self.blur else self.diffuse_blur(x_start, t, noise)[0]
        # ! in def train from Trainer set x_grid, .. to global to visualize below
        # x_ = [[x_b.unsqueeze(0), torch.randn_like(x_b).unsqueeze(0)] for x_b in x]
        # noised_imgs.insert(0,[inp[0].unsqueeze(0), torch.zeros_like(inp[0])])
        # visualize_noisy_fields(x_, [0]*100, False, True, coords=torch.tensor([1,2,3]*100).reshape(100,3), x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, t=t)

        if self.blur: t = (t * self.num_timesteps).long() # move from [0,1] to indices

        # predict and take gradient step

        model_out = self.model(x, t, classes)

        if self.objective == 'pred_noise':
            target = noise if target is None else target
        elif self.objective == 'pred_x0':
            target = x_start if target is None else target
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise) if target is None else target
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        if rotations is None:
            loss = self.loss_fn(model_out, target, reduction = 'none')
            if no_reduction: return loss
        if rotations is not None:


            coords_rotated = []
            rotations = torch.tensor(np.stack(rotations))[t.cpu()].to(device=x_start.device, dtype=torch.float32).permute(0,3,1,2)
            # * since there's a variable no of coords/molecule i for loop - probably not a bottleneck anyways but could switch to some form of padding
            for crds, rot in zip(coords, rotations):
                coords_rotated.append(torch.bmm(rot, crds.permute(0,2,1)).permute(0,2,1).reshape(-1,3))
            coords_rotated = torch.cat(coords_rotated)
            rot_target = create_gaussian_batch_pdf_values(x=grids, coords=coords_rotated, N_list=N_list, std=0.05, device=x_start.device, gaussian_indices=inds,no_fields =self.C, grid_shapes=[self.H, self.W, self.D], threshold_vlals=True,  backward_mdl_compat=False, gmm_mixt_weight=1/5.679043443503446)
            rot_target = (rot_target-1/2)*2
            rot_target = rot_target.reshape(-1, no_rotations,self.C, self.H, self.W, self.D)

            loss = (rot_target - model_out[:, None, :, :, :, :])**2
            loss = torch.mean(loss, dim=(2,3,4,5))
            if only_min_activ: return torch.mean(torch.min(loss, dim=1)[0])
            with torch.no_grad():
                soft_weights =torch.nn.functional.softmin(loss)
            return torch.mean(torch.sum(loss*soft_weights,dim=1)) # weighted average across rotations, average across batch samples

            with torch.no_grad():
                # try just selecting first
                loss = (rot_target - model_out_[:, None, :, :, :, :])**2
                loss = torch.mean(loss, dim=(2,3,4,5))
                loss, inds = torch.min(loss, dim=1) # ! min along rotations dimension (maybe some softmin?)
                target=rot_target[torch.arange(inds.shape[0]), inds, :, :, :, :]
                del rot_target
                # torch.cuda.empty_cache() # ! this slows down by a huge amount (although it's more mem efficient)
                # return target
            # loss = loss * extract(self.p2_loss_weight, t, loss.shape)
            # return loss.mean()
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        if "validation" in kwargs and kwargs["validation"]:
            return self.get_loss_by_time(img, kwargs['classes'])

        if self.twoD:
            b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
            assert h == self.H and w == self.W, f'height and width of image must be {img_size}'
        else:
            b, c, h, w, d, device, img_size, = *img.shape, img.device, self.image_size
            assert h == self.H and w == self.W and d == self.D, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        if self.blur: t = t/self.num_timesteps # move between [0,1] for blur-diff compatibility
        img = normalize_to_neg_one_to_one(img)

        return self.p_losses(img, t, *args, **kwargs)
    


    def get_frequency_scaling (self, t , min_scale =0.001, image_size=32, device=None):
        sigma_blur_max = 20
        # compute dissipation time
        sigma_blur = sigma_blur_max * torch.sin ( t * np.pi / 2) ** 2
        dissipation_time = sigma_blur ** 2 / 2
        # compute frequencies
        # freq = torch.arange(1,33).reshape(32,1) ** 2 / 32 **2
        freqs = np.pi*torch.linspace(0,image_size-1,image_size, device=device)/(image_size-1)
        L = freqs[None, None, :, None, None]**2 + freqs[None, None, None, :, None]**2 + freqs[None, None, None, None, :]**2
        L = L.repeat(t.shape[0], 1, 1, 1, 1)
        dissipation_time = dissipation_time[:, None, None, None, None]
        # compute scaling for frequencies
        scaling = torch.exp(-L * dissipation_time) * (1 - min_scale)
        return scaling + min_scale

    def get_alpha_sigma(self, t, image_size, device):
        freq_scaling = self.get_frequency_scaling(t, image_size=image_size, device=device)
        a, sigma = self.get_noise_scaling_cosine(t, device=device)
        alpha = a[:, None, None, None, None] * freq_scaling
        return alpha, sigma

    def get_noise_scaling_cosine (self, t , logsnr_min = -10 , logsnr_max =10, device=None):
        limit_max = torch.arctan ( torch.exp ( torch.tensor([-0.5 * logsnr_max], device=device) ))
        limit_min = torch.arctan ( torch.exp ( torch.tensor([-0.5 * logsnr_min], device=device) )) - limit_max
        logsnr = -2 * torch.log ( torch.tan ( limit_min * t + limit_max ))
        # Transform logsnr to a , sigma .
        return torch.sqrt ( torch.sigmoid ( logsnr )) , torch.sqrt ( torch.sigmoid ( - logsnr ))


    def diffuse_blur(self, x,t, noise=None):
        x_freq = dct_3d(x)
        b, c, h, w, d, device, img_size, = *x.shape, x.device, self.image_size
        alpha, sigma = self.get_alpha_sigma(t, h, device) # ! i for now assuem all dims are equal = h; 
        eps = noise if exists(noise) else torch.randn_like(x) 
        z_t = idct_3d(x_freq * alpha) +  sigma[:, None, None, None, None] * eps
        return z_t, eps



# def sanity_check_coords(coords, gaussian_indices):
#     for i in range(np.max(gaussian_indices) // 9):
#         current_ind = 9* i
#         chosen_coords = coords[[gi in range(current_ind, current_ind + 5) for gi in gaussian_indices]]
#         dist_x, dist_y, dist_z = torch.max(chosen_coords[:, 0]) - torch.min(chosen_coords[:, 0]), \
#             torch.max(chosen_coords[:, 1]) - torch.min(chosen_coords[:, 1]),  \
#             torch.max(chosen_coords[:, 2]) - torch.min(chosen_coords[:, 2])     
#         print(dist_x, dist_y, dist_z)
    # for ind in np.unique(gaussian_indices):
    #     current_ind = 9* ind
    #     for i in range(current_ind, current_ind + 5)
    #     chosen_coords = coords[gaussian_indices == ind]
    #     dist_x, dist_y, dist_z = torch.max(chosen_coords[:, 0]) - torch.min(chosen_coords[:, 0]), \
    #         torch.max(chosen_coords[:, 1]) - torch.min(chosen_coords[:, 1]),  \
    #         torch.max(chosen_coords[:, 2]) - torch.min(chosen_coords[:, 2])     
    #     print(dist_x, dist_y, dist_z)



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
        std_atoms=None,
        backward_mdl_compat=False,
        optimize_bnd_gmm_weights=False,
        threshold_bond=0.75,
        threshold_atm=0.75,
        augment_rotations=False,
        center_atm_to_grids=False,
        model_type='unet',
        data_generator=None,
        data_type='QM9',
        train_dataset_args=None,
        val_dataset_args=None,
        no_fields=8, 
        unique_atms_considered=None,
        val_batch_size=10,
        arom_cycle_channel=False,
        multi_gpu=False,
        data_path=None,
        fix_pi_values=False,
        data_file=None,
        gmm_mixt_weight=1/5.679043443503446,
        remove_bonds=False,
        inv_rot_loss_angles=-1,
        only_min_activ=False,
        cond_variables=None,
        atomno_conditioned_variable_sampler=None,
        norm_factors=None,
        discrete_conditioning=False
    ):
        super().__init__()
        self.discrete_conditioning=discrete_conditioning
        self.norm_factors=norm_factors
        self.atomno_conditioned_variable_sampler=atomno_conditioned_variable_sampler
        self.cond_variables=cond_variables
        self.remove_bonds=remove_bonds
        self.only_min_activ=only_min_activ
        self.arom_cycle_channel=arom_cycle_channel
        self.data_type=data_type
        self.data_path=data_path
        self.unique_atms_considered=unique_atms_considered
        self.no_fields=no_fields
        self.train_dataset_args=train_dataset_args
        self.val_dataset_args=val_dataset_args
        self.train_batch_size=train_batch_size
        self.data_generator=data_generator
        self.model_type = model_type
        self.run_name=run_name
        self.explicit_aromatic=explicit_aromatic
        self.grids=grids
        self.sep_bond_chn=sep_bond_chn
        self.explicit_hydrogen=explicit_hydrogen
        self.mixed_prec=mixed_prec
        self.std_atoms=std_atoms
        self.backward_mdl_compat=backward_mdl_compat
        self.optimize_bnd_gmm_weights=("GEOM" in data_type) or optimize_bnd_gmm_weights
        self.threshold_bond=threshold_bond if "GEOM" not in data_type else 0.4
        self.threshold_atm=threshold_atm if "GEOM" not in data_type else 0.4
        self.center_atm_to_grids=center_atm_to_grids
        # accelerator

        self.augment_rotations=augment_rotations

        self.accelerator = Accelerator(
            split_batches = split_batches,
        )

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model
        if torch.cuda.device_count() > 1 and multi_gpu: self.model = nn.DataParallel(self.model)
        elif torch.cuda.device_count() <= 1 and multi_gpu: print("!!!WARNING!!! - multi_gpu=True but only one GPU detected. Ignoring multi_gpu flag.")
        self.multi_gpu = torch.cuda.device_count() > 1 and multi_gpu
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

        if self.compact_batch:
            if self.cond_variables:
                self.dl = DataLoader(data, batch_size=train_batch_size, shuffle=True, pin_memory=False, num_workers=1, collate_fn=lambda x: collate_fn_general_cond_var(x, no_channels=self.no_fields, atms_last_ind=3 + explicit_hydrogen), drop_last=self.multi_gpu)
                self.dl_val = DataLoader(valid_data, batch_size=val_batch_size, shuffle=True, pin_memory=False, num_workers=1, collate_fn=lambda x: collate_fn_general_cond_var(x, no_channels=self.no_fields, atms_last_ind=3 + explicit_hydrogen), drop_last=self.multi_gpu)

                # self.dl = DataLoader(data, batch_size=train_batch_size, shuffle=True, pin_memory=False, num_workers=1, collate_fn=lambda x: collate_fn_general_cond_var(x, no_channels=len(self.unique_atms_considered) + 3 + 2 * arom_cycle_channel, atms_last_ind=len(self.unique_atms_considered)-1), drop_last=True)
                # self.dl_val = DataLoader(valid_data, batch_size=3, shuffle=True, pin_memory=False, num_workers=1, collate_fn=lambda x: collate_fn_general_cond_var(x, no_channels=len(self.unique_atms_considered) + 3 + 2 * arom_cycle_channel, atms_last_ind=len(self.unique_atms_considered)-1), drop_last=True)
            elif data_type == "GEOM":
                self.dl = DataLoader(data, batch_size=train_batch_size, shuffle=True, pin_memory=False, num_workers=1, collate_fn=lambda x: collate_fn_general(x, no_channels=len(self.unique_atms_considered) + 3 + 2 * arom_cycle_channel + self.explicit_aromatic, atms_last_ind=len(self.unique_atms_considered)-1), drop_last=True)
                self.dl_val = DataLoader(valid_data, batch_size=3, shuffle=True, pin_memory=False, num_workers=1, collate_fn=lambda x: collate_fn_general(x, no_channels=len(self.unique_atms_considered) + 3 + 2 * arom_cycle_channel  + self.explicit_aromatic, atms_last_ind=len(self.unique_atms_considered)-1), drop_last=True)
            else:
                # self.dl = DataLoader(data, batch_size=train_batch_size, shuffle=True, pin_memory=False, num_workers=1, collate_fn=collate_fn_compact_expl_ah_clsfreeguid if self.explicit_aromatic else collate_fn_compact_expl_h_clsfreeguid)
                # self.dl_val = DataLoader(valid_data, batch_size=val_batch_size, shuffle=True, pin_memory=False, num_workers=1, collate_fn=collate_fn_compact_expl_ah_clsfreeguid if self.explicit_aromatic else collate_fn_compact_expl_h_clsfreeguid)
                self.dl = DataLoader(data, batch_size=train_batch_size, shuffle=True, pin_memory=False, num_workers=1, collate_fn=lambda x: collate_fn_general(x, no_channels=self.no_fields, atms_last_ind=3 + explicit_hydrogen), drop_last=self.multi_gpu)
                self.dl_val = DataLoader(valid_data, batch_size=val_batch_size, shuffle=True, pin_memory=False, num_workers=1, collate_fn=lambda x: collate_fn_general(x, no_channels=self.no_fields, atms_last_ind=3 + explicit_hydrogen), drop_last=self.multi_gpu)
        else:
            if subsample_points != -1: self.dl = DataLoader(data, batch_size=train_batch_size, shuffle=True, pin_memory=False, num_workers=8 if exists(valid_data) else 9, collate_fn=collate_subsample_points)
            else: self.dl = DataLoader(data, batch_size=train_batch_size, shuffle=True, pin_memory=False, num_workers=8 if exists(valid_data) else 9)
            self.dl_val = DataLoader(valid_data, batch_size=val_batch_size, shuffle=True, pin_memory=False, num_workers=1) if exists(valid_data) else None

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
        if gmm_mixt_weight is not None and fix_pi_values: self.gmm_mixt_weight = gmm_mixt_weight
        elif fix_pi_values:
            print("!!!WARNING!!! determining pi values from data was never finished - using default value of 5.679043443503446")
            self.gmm_mixt_weight = 1/5.679043443503446
            # pi_value_file = os.path.join(data_path, data_file.replace(".bin","pi_values.bin"))
            # if fix_pi_values and not os.path.exists(pi_value_file): self.determine_pi_values(pi_value_file)
            # elif fix_pi_values: self.pi_values, _ = pickle.load(open(os.path.load(pi_value_file, allow_pickle=True), "rb"))
        else: self.gmm_mixt_weight = None
        self.inv_rot_loss_angles=inv_rot_loss_angles
        self.create_rot_matrices()

    def create_rot_matrices(self):
        if self.inv_rot_loss_angles == -1: self.rotations = None; return
        # if self.inv_rot_loss_angles%2 ==0: print("!!!WARNING!!! - inv_rot_loss_angles should be odd - adding 1"); self.inv_rot_loss_angles+=1
        add_identity = self.inv_rot_loss_angles%2 ==0 # identity matrix rot is added automatiically for odd
        alphas_cumprod = self.model.module.alphas_cumprod if self.multi_gpu else self.model.alphas_cumprod
        angle_intervals = (1-alphas_cumprod) * torch.tensor([[-np.pi], [np.pi]], device=self.device) # 0 rotations need to always be contained as they are GT
        rotations_by_time = []
        for lower, upper in zip(angle_intervals[0], angle_intervals[1]):
            lower, upper = lower.item(), upper.item()
            alpha,beta,gamma = np.linspace(lower, upper, self.inv_rot_loss_angles), np.linspace(lower, upper, self.inv_rot_loss_angles), np.linspace(lower, upper, self.inv_rot_loss_angles)

            alpha, beta, gamma = np.meshgrid(alpha, beta, gamma, indexing='ij')
            alpha, beta, gamma = alpha.flatten(), beta.flatten(), gamma.flatten()

            rotations = np.array([[np.cos(beta)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma), np.cos(alpha) * np.sin(beta)*np.cos(gamma)+ np.sin(alpha)*np.sin(gamma)],
                                [np.cos(beta)*np.sin(gamma), np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma), np.cos(alpha) * np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)],
                                [-np.sin(beta), np.sin(alpha)*np.cos(beta), np.cos(alpha)*np.cos(beta)]])
            if add_identity: rotations = np.concatenate([rotations,np.eye(3)[:,:,None]], axis=2)
            rotations_by_time.append(rotations)
        self.rotations = rotations_by_time

    def determine_pi_values(self, pi_value_file):
        abs_max, abs_min = 0, 100
        current_max = torch.zeros(self.no_fields, device=self.device)
        for data in self.dl:
            if self.compact_batch:
                    coords, inds, N_list, classes, bnds = data
                    if self.augment_rotations: coords, inds, N_list, classes, bnds, removed = self.rotate_coords(coords, inds, N_list, classes, bnds)
                    if self.center_atm_to_grids: coords = self.center_positions_to_grids(coords, inds, N_list, classes, bnds)
                    grid_shapes = [self.model.module.H, self.model.module.W, self.model.module.D] if self.multi_gpu else [self.model.H, self.model.W, self.model.D]
                    inp = create_gaussian_batch_pdf_values(x=self.grid_inputs, coords=coords, N_list=N_list, std=self.std_atoms, device=self.device, gaussian_indices=inds,
                                                            no_fields =self.no_fields, grid_shapes=grid_shapes,
                                                            threshold_vlals= not self.remove_thresholding_pdf_vals, backward_mdl_compat=self.backward_mdl_compat, pi_values=torch.ones(self.no_fields))
            else: inp, classes = data
            max_inp = inp.reshape(-1, np.prod(grid_shapes)).max(dim=1)[0]
            min_max_inp = max_inp.flatten()[max_inp.flatten()>0].min()
            max_inp = max_inp.reshape(-1, self.no_fields)
            # if torch.max(max_inp) > 5.679043443503446: 
            #     atm_symb_vis, atm_pos_vis, actual_bnds_vis = transform_training_batch_into_visualizable(coords, inds, N_list,bnds, self.explicit_aromatic,self.explicit_hydrogen,self.no_fields, data_type=self.data_type)
            #     index_ = torch.where(max_inp>6)[0].item()
            #     for i in range(len(inp)):
            #         if i != index_: continue
            #         visualize_mol(plot_bnd=4, field=inp[i].cpu(), atm_pos=atm_pos_vis[i].cpu().numpy(), atm_symb=atm_symb_vis[i],actual_bnd=actual_bnds_vis[i], x_grid=self.x_grid,y_grid=self.y_grid, z_grid=self.z_grid, threshold=0.3, )
            max_inp = max_inp.max(dim=0)[0]

            max_max_inp = max_inp.max()
            
            max_max_inp, min_max_inp = max_max_inp.item(), min_max_inp.item()
            abs_min = (min_max_inp < abs_min)*min_max_inp + (min_max_inp >= abs_min)*abs_min
            abs_max = (max_max_inp > abs_max)*max_max_inp + (max_max_inp <= abs_max)*abs_max

            current_max = (max_inp > current_max)*max_inp + (max_inp <= current_max)*current_max


        breakpoint()
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
        # scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        # self.opt.state_dict()
        print("saved model {}".format('model-{}'.format(milestone) + self.run_name + '.pt'))
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
        elif not self.multi_gpu and "module" in list(data['model'].keys())[0]:
            state_dict_ = {}
            for k, v in data['model'].items():
                state_dict_.update({k.replace("module.", ""): v})
            state_dict_ = OrderedDict(state_dict_)
            data['model'] = state_dict_
        try:
            self.model.load_state_dict(data['model'])
        except:
            self.model.load_state_dict(data['model'], strict=False)
            print("!!!WARNING!!! Found unmatching keys when loading model. Proceed with caution")
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

    # def rotate_coords_nobnds(self, coords, inds, N_list, classes, bnds, no_channels):
    #     new_coords, new_inds, new_N_list, new_classes, new_bnds = [], [], [], [], []
    #     removed = 0
    #     removed_coords = 0
    #     current_ind = 0
    #     listed_bnds = []
    #     for _ in range(len(N_list)): listed_bnds.append([])
    #     limits = [sum([sum(n_list) for n_list in N_list[:i]]) for i in range(len(N_list) + 1)]
    #     limits.append(limits[-1]+1)
    #     limits = np.array(limits)
    #     for bnd in bnds:
    #         index = np.argwhere(bnd[1] < limits).reshape(-1)[0] -1
    #         listed_bnds[index].append(bnd)
    #     listed_bnds = [np.array(lb) for lb in listed_bnds]
    #     for i in range(len(N_list)):
    #         coords_ = coords[current_ind:current_ind+sum(N_list[i])]
    #         r = Rotation.random()
    #         coords_ = r.apply(coords_)

    #         lim1 = coords_.min() - self.std_atoms > self.x_grid.min() and coords_.min() - self.std_atoms > self.y_grid.min() and coords_.min() - self.std_atoms > self.z_grid.min()
    #         lim2 = coords_.max() + self.std_atoms < self.x_grid.max() and coords_.max() + self.std_atoms < self.y_grid.max() and coords_.max() + self.std_atoms < self.z_grid.max()
    #         if i == 2: lim1 = False
    #         if lim1 and lim2:
    #             new_coords.extend(coords_)
    #             new_inds.extend(inds[current_ind:current_ind+sum(N_list[i])] - no_channels * removed)
    #             new_N_list.append(N_list[i])
    #             new_classes.append(classes[i])

    #             # subtract removed atoms from atom indices listed_bnds
    #             current_bonds = listed_bnds[i]; current_bonds[:, 1:] = current_bonds[:, 1:] - removed_coords
    #             new_bnds.extend(current_bonds)
    #         else: removed+=1; removed_coords += sum(N_list[i])

    #         current_ind += sum(N_list[i])
    #     return torch.tensor(np.stack(new_coords), device=self.device, dtype=torch.float32), \
    #         np.array(new_inds), new_N_list, torch.stack(new_classes), np.stack(new_bnds), removed

    def rotate_coords(self, coords, inds, N_list, classes, bnds, no_channels, remove_bonds):
        new_coords, new_inds, new_N_list, new_classes, new_bnds = [], [], [], [], []
        removed = 0
        removed_coords = 0
        current_ind = 0
        listed_bnds = []
        for _ in range(len(N_list)): listed_bnds.append([])
        limits = [sum([sum(n_list) for n_list in N_list[:i]]) for i in range(len(N_list) + 1)]
        limits.append(limits[-1]+1)
        limits = np.array(limits)
        for bnd in bnds:
            index = np.argwhere(bnd[0] < limits).reshape(-1)[0] -1 if not remove_bonds else np.argwhere(bnd[1] < limits).reshape(-1)[0] -1
            listed_bnds[index].append(bnd)

        listed_bnds = [np.array(lb) for lb in listed_bnds]
        for i in range(len(N_list)):
            coords_ = coords[current_ind:current_ind+sum(N_list[i])]
            r = Rotation.random()
            coords_ = r.apply(coords_)

            lim1 = coords_.min() - self.std_atoms > self.x_grid.min() and coords_.min() - self.std_atoms > self.y_grid.min() and coords_.min() - self.std_atoms > self.z_grid.min()
            lim2 = coords_.max() + self.std_atoms < self.x_grid.max() and coords_.max() + self.std_atoms < self.y_grid.max() and coords_.max() + self.std_atoms < self.z_grid.max()
            if lim1 and lim2:
                new_coords.extend(coords_)
                new_inds.extend(inds[current_ind:current_ind+sum(N_list[i])] - no_channels * removed)
                new_N_list.append(N_list[i])
                new_classes.append(classes[i])

                if remove_bonds:
                    current_bonds = listed_bnds[i]; current_bonds[:, 1:] = current_bonds[:, 1:] - removed_coords
                    new_bnds.extend(current_bonds)
                else: new_bnds.extend(listed_bnds[i] - removed_coords)
            else: removed+=1; removed_coords += sum(N_list[i])
            current_ind += sum(N_list[i])
        return torch.tensor(np.stack(new_coords), device=self.device, dtype=torch.float32), \
            np.array(new_inds), new_N_list, torch.stack(new_classes), np.stack(new_bnds), removed

    def center_positions_to_grids(self, coords, inds, N_list, classes, bnds):
        all_grid_crds = torch.tensor(np.stack([self.x_grid.reshape(-1), self.y_grid.reshape(-1), self.z_grid.reshape(-1)]).T, dtype=torch.float32, device=coords.device)
        tot_channels = 7 + self.explicit_aromatic + self.explicit_hydrogen
        tot_atms = 4 + self.explicit_hydrogen
        atm_inds = [ind for ind, i in enumerate(inds) if i%tot_channels < tot_atms]
        atm_crds = coords[atm_inds]
        closest_grids = torch.argmin(torch.cdist(atm_crds, all_grid_crds), dim=1)
        coords[atm_inds] = all_grid_crds[closest_grids]

        # update bond coordinate to the new, discretized atm positions
        coords[bnds[:,0]] = (coords[bnds[:, 1]] + coords[bnds[:, 2]])/2
        return coords

    def prep_mol_inf_for_rot(self, coords, inds, N_list):
        N_list_padded, coords_padded, inds_padded = [], [], []
        max_atm_no = max([sum(n_list) for n_list in N_list])
        current_index = 0
        coords = coords.to(self.device)
        for n_list in N_list:
            n_list_padded = copy.deepcopy(n_list)
            n_list_padded.extend([-1]*(self.no_fields-len(n_list))) 
            N_list_padded.append(n_list_padded)

            current_coords = coords[current_index:current_index+sum(n_list)]
            padding = torch.ones((max_atm_no - sum(n_list), 3), device=self.device) * -100
            pad_coords = torch.vstack([current_coords,padding])
            coords_padded.append(pad_coords)

            current_inds =  inds[current_index:current_index+sum(n_list)]
            pad_inds = np.hstack([current_inds, np.ones(max_atm_no - sum(n_list)) * -1])
            inds_padded.append(pad_inds)
            current_index += sum(n_list)
        return torch.tensor(N_list_padded, device=self.device), torch.stack(coords_padded), torch.tensor(np.stack(inds_padded), device=self.device)

    def train(self):
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        loss_by_time = {t: 0 for t in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800]} if self.twoD \
            else {t: 0 for t in [5, 50, 100, 400, 800]}
        accelerator = self.accelerator
        # device = accelerator.device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        total_loss = 0.
        first_ep = True

        while self.step < self.train_num_steps:
            if first_ep: first_ep = False
            elif not first_ep and exists(self.data_generator): # data_generator exists when dataset is too large to fit in memory
                # data, data_val = get_next_ep_data(self.data_generator,self.train_dataset_args,self.val_dataset_args)
                data, data_val = self.data_generator.get_next_epoch_dl(self.train_dataset_args,self.val_dataset_args)
                self.dl = DataLoader(data, batch_size=self.train_batch_size, shuffle=True, pin_memory=False, num_workers=1, collate_fn=lambda x: collate_fn_general(x, no_channels=len(self.unique_atms_considered) + 3 + 2 * self.arom_cycle_channel + self.explicit_aromatic, atms_last_ind=len(self.unique_atms_considered)-1), drop_last=True)
                self.dl_val = DataLoader(data_val, batch_size=10, shuffle=True, pin_memory=False, num_workers=1, collate_fn=lambda x: collate_fn_general(x, no_channels=len(self.unique_atms_considered) + 3 + 2 * self.arom_cycle_channel + self.explicit_aromatic, atms_last_ind=len(self.unique_atms_considered)-1), drop_last=True)                
            for data in self.dl:
                if self.compact_batch:
                    if not self.cond_variables: coords, inds, N_list, classes, bnds = data; cond_var = None
                    else: 
                        coords, inds, N_list, classes, bnds, cond_var = data
                        cond_var = (torch.tensor(cond_var) - self.norm_factors[0])/(self.norm_factors[1]-self.norm_factors[0])
                        cond_var = cond_var.to(dtype=torch.float32)


                    if self.augment_rotations: coords, inds, N_list, classes, bnds, removed = self.rotate_coords(coords, inds, N_list, classes, bnds, self.no_fields, self.remove_bonds) #if not self.remove_bonds else self.rotate_coords_nobnds(coords, inds, N_list, classes, bnds, self.no_fields)
                    if self.center_atm_to_grids: coords = self.center_positions_to_grids(coords, inds, N_list, classes, bnds)
                    grid_shapes = [self.model.module.H, self.model.module.W, self.model.module.D] if self.multi_gpu else [self.model.H, self.model.W, self.model.D]
                    inp = create_gaussian_batch_pdf_values(x=self.grid_inputs, coords=coords, N_list=N_list, std=self.std_atoms, device=device, gaussian_indices=inds,
                                                            no_fields =self.no_fields, grid_shapes=grid_shapes,
                                                            threshold_vlals= not self.remove_thresholding_pdf_vals, backward_mdl_compat=self.backward_mdl_compat, gmm_mixt_weight=self.gmm_mixt_weight)
                    if self.discrete_conditioning: cond_var=None
                else: inp, classes = data

                # * #### visualize inputs
                # atm_symb_vis, atm_pos_vis, actual_bnds_vis = transform_training_batch_into_visualizable(coords, inds, N_list,bnds, self.explicit_aromatic,self.explicit_hydrogen,self.no_fields, data_type=self.data_type, non_expl_bonds=self.no_fields<6)
                # for i in range(len(inp)):
                #     visualize_mol(plot_bnd=0, field=inp[i].cpu(), atm_pos=atm_pos_vis[i].cpu().numpy(), 
                #                  atm_symb=atm_symb_vis[i],actual_bnd=actual_bnds_vis[i], 
                #                   x_grid=self.x_grid,y_grid=self.y_grid, z_grid=self.z_grid,
                #                     threshold=0.1, title=-1, set_legend=True, data="GEOM")
                # * #### visualize inputs

                # ! needed for visualization from within forward loop of GaussianDiffusion model
                global x_grid; global y_grid; global z_grid; x_grid, y_grid, z_grid = self.x_grid, self.y_grid, self.z_grid
                

                if self.subsample_points != -1: inp, train_indexes = inp
                classes = classes.to(self.device)
                if self.subsample_points != -1: data, train_indexes = data
                else: train_indexes = None
                inp = inp.to(device)
                if self.mixed_prec:
                    with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
                        loss = self.model(inp,grids=self.grids, train_indexes=train_indexes, classes=classes)/self.gradient_accumulate_every
                        loss = loss.mean()
                        total_loss += loss.item()
                        scaler.scale(loss).backward()

                        if self.step % self.gradient_accumulate_every == 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                            scaler.step(self.opt)
                            scaler.update()
                            self.opt.zero_grad()
                else:
                    if self.rotations is not None:
                        N_list_pded, coords_pded, inds_pded  = self.prep_mol_inf_for_rot(coords, inds, N_list)
                        loss = self.model(inp,grids=self.grid_inputs.detach().cpu().numpy(), train_indexes=train_indexes, classes=classes,coords_pded=coords_pded, inds_pded=inds_pded, N_list_pded=N_list_pded, rotations=self.rotations, only_min_activ=self.only_min_activ)/self.gradient_accumulate_every
                        # with torch.no_grad():
                        #     N_list_pded, coords_pded, inds_pded  = self.prep_mol_inf_for_rot(coords, inds, N_list)
                        #     targets = self.model(inp,grids=self.grid_inputs.detach().cpu().numpy(), train_indexes=train_indexes, classes=classes,coords_pded=coords_pded, inds_pded=inds_pded, N_list_pded=N_list_pded, rotations=self.rotations)/self.gradient_accumulate_every
                        # loss = self.model(inp,grids=self.grids, train_indexes=train_indexes, classes=classes, target=targets)/self.gradient_accumulate_every
                        # N_list_pded, coords_pded, inds_pded  = self.prep_mol_inf_for_rot(coords, inds, N_list)
                        # loss = self.model(inp,grids=self.grid_inputs.detach().cpu().numpy(), train_indexes=train_indexes, classes=classes,coords_pded=coords_pded, inds_pded=inds_pded, N_list_pded=N_list_pded, rotations=self.rotations)/self.gradient_accumulate_every

                    else:
                        loss = self.model(inp,grids=self.grids, train_indexes=train_indexes, classes=classes, cond_var=cond_var)/self.gradient_accumulate_every
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
                    data_val= next(iter(self.dl_val)) if exists(self.dl_val) else data
                    if self.compact_batch:
                        if not self.cond_variables: coords, inds, N_list, classes, bnds = data_val; cond_var = None
                        else: coords, inds, N_list, classes, bnds, cond_var = data_val; cond_var = (torch.tensor(cond_var) - self.norm_factors[0])/(self.norm_factors[1]-self.norm_factors[0]); cond_var=cond_var.to(dtype=torch.float32)
                        if self.discrete_conditioning: cond_var=None
                        # coords, inds, N_list,classes,bnds = data_val
                        grid_shapes = [self.model.module.H, self.model.module.W, self.model.module.D] if self.multi_gpu else [self.model.H, self.model.W, self.model.D]
                        inp_val = create_gaussian_batch_pdf_values(x=self.grid_inputs, coords=coords, N_list=N_list, std=self.std_atoms, device=device, gaussian_indices=inds,
                                                                no_fields = self.no_fields, grid_shapes=grid_shapes,
                                                                threshold_vlals= not self.remove_thresholding_pdf_vals,backward_mdl_compat=self.backward_mdl_compat)
                    else: inp_val, classes = data_val
                    data_val, classes = inp_val.to(self.device), classes.to(self.device)
                    with torch.no_grad():
                        # TODO should this be also self.model.eval/train before/after (for block norm)? - mb not/no batch stats
                        loss_by_time_ = self.model(data_val, classes=classes, validation=True, cond_var=cond_var)
                        loss_by_time_ = loss_by_time_.reshape(-1, len(loss_by_time.keys())).mean(dim=0)
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
                        if self.model_type =='unet': 
                            no_cls = self.model.model.classes_emb.num_embeddings if not self.multi_gpu else self.model.module.model.classes_emb.num_embeddings
                        elif self.model_type == 'DiT': no_cls = self.model.model.y_embedder.num_classes
                        if exists(self.atomno_conditioned_variable_sampler) and not self.discrete_conditioning: # for discrete conditioning on con_var, cond_va is already a class
                            classes, cond_var = self.atomno_conditioned_variable_sampler.sample(10)
                            classes, cond_var = torch.tensor(classes).to(device), (torch.tensor(cond_var) - self.norm_factors[0])/(self.norm_factors[1]-self.norm_factors[0])
                            cond_var=cond_var.to(device=device, dtype=torch.float32)
                        else: classes = torch.randint(0, no_cls, (10,)).to(self.device); cond_var = None
                        # all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
                        if self.multi_gpu: all_images_list = list(map(lambda n: self.model.module.sample(classes=n[0], cond_var=n[1]), [[classes, cond_var]]))
                        else: all_images_list = list(map(lambda n: self.model.sample(classes=n[0], cond_var=n[1]), [[classes, cond_var]]))
                        # all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))  this gest nowehere for a long time so mb test on mdl

                        #
                        # all_images_list = list(map(lambda n: self.model.sample(batch_size=n), batches))
                        # below might fix it a sampling issue?? alternatively, fix p_sample_loop
                        # all_images_list = list(map(lambda n: self.model.module.sample(batch_size=n), batches))

                    #print(len(all_images_list))
                    training_images = self.model.module.twoD if self.multi_gpu else self.model.twoD
                    if not training_images:
                        all_images = torch.cat(all_images_list, dim=0)
                        try:
                            if self.sep_bond_chn:
                                val, unq, nov, _ = fit_pred_field_sep_chn_batch(all_images, 0.1, self.x_grid, self.y_grid, self.z_grid,
                                            len(self.channel_atoms), self.channel_atoms, self.dataset_smiles,
                                            noise_std=0.0, normalize01=True,explicit_aromatic=self.explicit_aromatic,
                                            explicit_hydrogen=self.explicit_hydrogen, optimize_bnd_gmm_weights=self.optimize_bnd_gmm_weights,
                                            threshold_bond=self.threshold_bond, threshold_atm=self.threshold_atm, atm_symb_bnd_by_channel=self.unique_atms_considered if "GEOM" in self.data_type else None,
                                            atm_dist_based=self.remove_bonds)
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


        accelerator.print('training complete')




# example

if __name__ == '__main__':
    num_classes = 10




    model = Unet3D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        num_classes = num_classes,
        cond_drop_prob = 0.5
    )
    training_images = torch.randn(2, 3, 32,32,32).cuda() # images are normalized from 0 to 1

    diffusion = GaussianDiffusion(
        model,
        image_size = training_images[0].shape,
        timesteps = 10
    ).cuda()

    image_classes = torch.randint(0, num_classes, (2,)).cuda()    # say 10 classes

    loss = diffusion(training_images, classes = image_classes)
    loss.backward()

    # do above for many steps

    sampled_images = diffusion.sample(
        classes = image_classes,
        cond_scale = 3.                # condition scaling, anything greater than 1 strengthens the classifier free guidance. reportedly 3-8 is good empirically
    )

    sampled_images.shape # (8, 3, 128, 128)



    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        num_classes = num_classes,
        cond_drop_prob = 0.5
    )
    training_images = torch.randn(8, 3, 128, 128).cuda() # images are normalized from 0 to 1

    diffusion = GaussianDiffusion(
        model,
        image_size = tuple(training_images[0].shape),
        timesteps = 10,
        twoD=True
    ).cuda()

    image_classes = torch.randint(0, num_classes, (8,)).cuda()    # say 10 classes

    loss = diffusion(training_images, classes = image_classes)
    loss.backward()

    # do above for many steps

    sampled_images = diffusion.sample(
        classes = image_classes,
        cond_scale = 3.                # condition scaling, anything greater than 1 strengthens the classifier free guidance. reportedly 3-8 is good empirically
    )

    sampled_images.shape # (8, 3, 128, 128)

