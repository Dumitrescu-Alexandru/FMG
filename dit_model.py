
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------


# * import timm stuff that is not to define PatchEmbed3D
from timm.layers.format import Format
from timm.layers.helpers import _ntuple
import copy
from einops import rearrange, reduce
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from typing import Callable, List, Optional, Tuple, Union
from torch import _assert
from torch.nn.utils.rnn import pad_sequence
from typing import Final
# code taken from https://arxiv.org/pdf/2212.09748.pdf
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_neighbod_inds(size=16, att_size=2):
    """
        look 2 inds away in every direction (for diagonals, they will be further, and will have e.g. [x+y,y+2,z+y] index)
    """
    sparse_neighbor_inds = []
    normalization_indices = []
    pad_ind = size ** 3 
    grid_x, grid_y, grid_z = np.meshgrid(np.arange(size), np.arange(size), np.arange(size), indexing='ij')
    indices = []
    current_index = 0
    for x,y,z in zip(grid_x.flatten(), grid_y.flatten(), grid_z.flatten()):
        x_gr,y_gr,z_gr = np.meshgrid(np.arange(x-att_size,x+att_size+1), np.arange(y-att_size,y+att_size+1), np.arange(z-att_size,z+att_size+1), indexing='ij')
        x_gr,y_gr,z_gr = x_gr.flatten(), y_gr.flatten(), z_gr.flatten()
        inds =  (x_gr >= 0) * (y_gr >= 0) * (z_gr >= 0) *(x_gr < size) * (y_gr < size) * (z_gr < size)
        x_gr,y_gr,z_gr = x_gr[inds], y_gr[inds], z_gr[inds]
        flatten_inds = x_gr * size ** 2 + y_gr * size + z_gr
        indices.append(torch.tensor(flatten_inds, dtype=torch.long))

        current_inds = np.zeros(size**3)
        current_inds[flatten_inds] = np.array(list(range(current_index, current_index+len(flatten_inds))))
        sparse_neighbor_inds.extend(list(current_inds.astype(int)))

        normalization_indices.extend(list(range(current_index, current_index+len(flatten_inds))))
        normalization_indices.extend([0] * ((2*att_size +1) ** 3 - len(flatten_inds)))
        current_index += len(flatten_inds)
    padded_inds = pad_sequence(indices, batch_first=True, padding_value=pad_ind)
    return padded_inds, pad_ind, sparse_neighbor_inds, normalization_indices



class PatchEmbed3D(nn.Module):
    """ 3D Field to Patch embedding
    """
    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        if img_size is not None:
            self.img_size = to_3tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW


        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W, D = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
                _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
                _assert(D == self.img_size[2], f"Input width ({D}) doesn't match model ({self.img_size[2]}).")
            elif not self.dynamic_img_pad:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                )
                _assert(
                    D % self.patch_size[2] == 0,
                    f"Input width ({D}) should be divisible by patch size ({self.patch_size[2]})."
                )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None, cond_drop_prob=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        cond_drop_prob = cond_drop_prob if cond_drop_prob is not None else self.dropout_prob
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < cond_drop_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None, cond_drop_prob=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids, cond_drop_prob=cond_drop_prob)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################
class SmallAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            padded_neigh_inds=[], 
            pad_ind=4096,
            sparse_neighbor_inds=None,
            normalization_indices=None
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.padded_neigh_inds=padded_neigh_inds
        self.pad_ind=pad_ind
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask = padded_neigh_inds == pad_ind
        self.mask = self.mask.float().masked_fill(self.mask == 1, float('-inf')).to(self.device)
        self.sparse_neighbor_inds=sparse_neighbor_inds
        self.normalization_indices=normalization_indices
        self.block_dim = round(pad_ind ** (1/3))


        padded_inds = (self.padded_neigh_inds != self.block_dim**3) * self.padded_neigh_inds
        self.padded_row_inds = torch.arange(padded_inds.shape[0]).unsqueeze(1).repeat(1, padded_inds.shape[1])
        self.padded_col_inds = padded_inds.reshape(-1)


    def forward(self, x, att_matrix):
        # * Currently need to expand LxL. Perhaps there is some better alternative in link below.
        # https://stackoverflow.com/questions/67231563/pytorch-memory-efficient-implementation-of-indexed-matrix-multiplication
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale


        # TODO is the implementation below correct? Should I need BnhL x d/H matrices?
        unnormalized_att_weights = torch.sparse.sampled_addmm(att_matrix.cuda(), q.reshape(B * self.num_heads * self.pad_ind, -1), k.reshape(B * self.num_heads * self.pad_ind, -1).transpose(-2,-1), beta=0).values()
        unnormalized_att_weights = unnormalized_att_weights.reshape(B, self.num_heads, -1)
        # unnormalized_att_weights = torch.randn_like(unnormalized_att_weights)
        # unnormalized_att_weights = torch.sparse.sampled_addmm(att_matrix, q.reshape(B * self.num_heads * self.pad_ind, -1), k.reshape(B * self.num_heads * self.pad_ind, -1).transpose(-2,-1), beta=0).values()

        padded_inds = (self.padded_neigh_inds != self.block_dim**3) * self.padded_neigh_inds # 

        unnormalized_att_weights = unnormalized_att_weights[:, :, self.normalization_indices].reshape(B,self.num_heads,self.block_dim**3,-1)
        mask = self.mask.unsqueeze(0).unsqueeze(0)
        att_weights = torch.softmax(mask + unnormalized_att_weights, dim=-1)

        # ! from here just do sparse matrix multiplication
        # TODO make sure that selecting [row_ind, 0] as pad indices doesn't do anything - corresp vals are zero but...

        pad_inds_row = self.padded_row_inds.reshape(1,-1).repeat(B*self.num_heads, 1).reshape(-1)
        pad_inds_col = self.padded_col_inds.reshape(1,-1).repeat(B*self.num_heads, 1).reshape(-1)

        head_inds = torch.arange(self.num_heads)
        head_inds = head_inds[None, :, None].repeat(B, 1, len(self.padded_col_inds)) * N

        batch_inds = torch.arange(B)
        batch_inds = batch_inds[:,None, None].repeat(1,self.num_heads, len(self.padded_col_inds)) * N * self.num_heads



        head_inds, batch_inds = head_inds.reshape(-1), batch_inds.reshape(-1)
        pad_inds_row, pad_inds_col = pad_inds_row.reshape(-1), pad_inds_col.reshape(-1)
        pad_inds_row = pad_inds_row + batch_inds + head_inds
        pad_inds_col = pad_inds_col + batch_inds + head_inds


        inds = torch.stack([pad_inds_row, pad_inds_col]).to(self.device)
        sparse_att_matrix = torch.sparse.FloatTensor(inds, att_weights.flatten(), torch.Size([B*self.num_heads*N, B*self.num_heads*N]))
        
        weighted_avg_vectors = torch.sparse.mm(sparse_att_matrix, v.reshape(B*self.num_heads*N, -1))
        weighted_avg_vectors = weighted_avg_vectors.reshape(B, self.num_heads, N, -1)
        # weighted_avg_vectors = rearrange(weighted_avg_vectors, 'b h l d -> b l (d h)')
        weighted_avg_vectors = weighted_avg_vectors.transpose(1, 2).reshape(B, N, C)
        weighted_avg_vectors = self.proj(weighted_avg_vectors)
        weighted_avg_vectors = self.proj_drop(weighted_avg_vectors)
        return weighted_avg_vectors





        breakpoint()

        pad_inds_row = pad_inds_row.reshape(-1)


        # batch_inds = batch_inds[:,None, None].repeat(1,self.num_heads, len(pad_inds_col))
        # pad_inds_row = pad_inds_row[None, None, :].repeat(B, self.num_heads, 1).reshape(-1).to(self.device)
        # pad_inds_col = pad_inds_col[None, None, :].repeat(B, self.num_heads, 1).reshape(-1).to(self.device)
        # inds = torch.stack([pad_inds_row.flatten(), pad_inds_col.flatten() ])
        # att_weight_sparse = torch.sparse.FloatTensor(inds,att_weights.flatten(), torch.Size([B * self.num_heads * selfb.block_dim**3,self.block_dim**3]))


        

        # ! sparse matrix mult only implemented for 2d tensors so below is just bad
        # pad_inds_row = pad_inds_row.unsqueeze(0).unsqueeze(0).repeat(B, self.num_heads, 1).to(self.device)
        # pad_inds_col = pad_inds_col.unsqueeze(0).unsqueeze(0).repeat(B, self.num_heads, 1).to(self.device)
        # head_index = torch.arange(self.num_heads).unsqueeze(0).unsqueeze(-1).repeat(B, 1, pad_inds_col.shape[-1]).to(self.device)
        # batch_index = torch.arange(B).unsqueeze(1).unsqueeze(-1).repeat(1, self.num_heads, pad_inds_col.shape[-1]).to(self.device)



        inds = torch.stack([batch_index.flatten(), head_index.flatten(), pad_inds_row.flatten(), pad_inds_col.flatten() ])
        att_weight_sparse = torch.sparse.FloatTensor(inds,att_weights.flatten(), torch.Size([B, self.num_heads, self.block_dim**3,self.block_dim**3]))

        breakpoint()
        







        att_weights = att_weights.reshape(B, self.num_heads, -1)
        att_weights = att_weights[:, :, self.sparse_neighbor_inds].reshape(B, self.num_heads, self.block_dim**3, -1)

        values = torch.sparse.sampled_addmm(att_matrix, q, k.transpose(-2,-1), beta=0).values()


        # torch.sparse.sampled_addmm
        mask = self.mask.unsqueeze(0).unsqueeze(0).repeat(B, self.num_heads, 1, 1).to(self.device)
        attn += mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        
        
        value_inds = copy.deepcopy(self.padded_neigh_inds)
        value_inds[value_inds == self.pad_ind] = 0
        value_inds = list(value_inds.flatten().cpu().numpy())

        v = v[:,:,value_inds,:].reshape(B,self.num_heads,N,-1,C//self.num_heads)
        x = torch.sum(attn.unsqueeze(-1) * v, dim=3)

        #  attn[:,:,[[0],[1],[2]],[[1,2],[4,5],[7,8]]].shape

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DiTBlockSmallAtt(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SmallAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, att_matrix):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), att_matrix)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x




class FinalLayer3D(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

def Upsample3D(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b (p1 p2 p3) c -> b c p1 p2 p3', p1 = 8, p2 = 8, p3 = 8),
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv3d(dim, default(dim_out, dim), 3, padding = 1),
        Rearrange('b c h w l -> b (h w l) c'),

    )

def Downsample3D(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b (p1 p2 p3) c -> b c p1 p2 p3', p1 = 16, p2 = 16, p3 = 16),
        Rearrange('b c (h p1) (w p2) (l p3) -> b (c p1 p2 p3) h w l', p1 = 2, p2 = 2, p3 = 2),
        nn.Conv3d(dim * 8, default(dim_out, dim), 1),
        Rearrange('b c h w l -> b (h w l) c'),
    )

class DiT3D(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        att_size=2,
        small_attention=False,
        dbl_resolution=False
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.dbl_resolution=dbl_resolution
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.channels = in_channels
        self.out_dim = in_channels
        self.random_or_learned_sinusoidal_cond = False
        padded_neigh_inds, pad_ind, sparse_neighbor_inds, normalization_indices = get_neighbod_inds(size=input_size//patch_size, att_size=att_size)
        self.sparse_neighbor_inds = sparse_neighbor_inds
        self.normalization_indices=normalization_indices
        self.padded_neigh_inds = padded_neigh_inds
        self.pad_ind = pad_ind
        self.x_embedder = PatchEmbed3D(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)


        if dbl_resolution:
            self.x_embedder = PatchEmbed3D(input_size, patch_size//2, in_channels, hidden_size, bias=True)
            num_patches = self.x_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

            modules = [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)]
            modules.append(DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio))
            modules.append(Downsample3D(hidden_size))
            modules.extend([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth) for _ in range(depth-2)])
            self.blocks = nn.ModuleList(modules)
            self.final_layer = FinalLayer3D(hidden_size, patch_size, self.out_channels)

        # if dbl_resolution: # * low resolution first then high seems bad
        #     modules = [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth) for _ in range(depth-2)]
        #     modules.append(Upsample3D(hidden_size))
        #     modules.append(DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio))
        #     modules.append(DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio))
        #     self.blocks = nn.ModuleList(modules)
        #     self.final_layer = FinalLayer3D(hidden_size, patch_size//2, self.out_channels)

        elif small_attention:
            self.blocks = nn.ModuleList([
                DiTBlockSmallAtt(hidden_size, num_heads, mlp_ratio=mlp_ratio, padded_neigh_inds=padded_neigh_inds, pad_ind=pad_ind,sparse_neighbor_inds=sparse_neighbor_inds, normalization_indices=normalization_indices) for _ in range(depth)
            ])
            self.final_layer = FinalLayer3D(hidden_size, patch_size, self.out_channels)

        else:
            self.blocks = nn.ModuleList([
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
            ])
            self.final_layer = FinalLayer3D(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()
        self.cond_drop_prob = 0.5
        self.small_attention = small_attention
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



    
    def form_mult_matrix(self,B):

        no_of_blocks = B * self.num_heads
        no_elems = 0
        row_inds, col_inds = [0],[]
        for i in range(self.pad_ind):
            current_col_inds = list(self.padded_neigh_inds[i][self.padded_neigh_inds[i] != self.pad_ind].detach().cpu().numpy())
            row_inds.append(row_inds[-1] + len(current_col_inds))
            col_inds.extend(current_col_inds)
            no_elems += len(current_col_inds)
        row_inds, col_inds = np.array(row_inds)[1:], np.array(col_inds)
        
        
        col_index_shift = (np.arange(no_of_blocks) * self.pad_ind)[:, None]
        row_index_shift = (np.arange(no_of_blocks) * no_elems)[:, None]
        
        
        row_inds = row_inds[None,:].repeat(no_of_blocks,axis=0) + row_index_shift
        col_inds = col_inds[None,:].repeat(no_of_blocks,axis=0) + col_index_shift
        
        row_inds = list(row_inds.flatten())
        col_inds = list(col_inds.flatten())
        row_inds.insert(0,0)
        
        col_inds = torch.tensor(col_inds)
        row_inds = torch.tensor(row_inds)
        values = torch.ones(len(col_inds))

        return torch.sparse_csr_tensor(row_inds, col_inds,values)
        # breakpoint()
        # torch.sparse_csr_tensor()

        # row_inds = row_inds[None,:].repeat(no_of_blocks,axis=0) + batch_noHeads_indx
        # col_inds = col_inds[None,:].repeat(no_of_blocks,axis=0) + batch_noHeads_indx
        
        



        # no_of_blocks = B * self.num_heads
        # row_inds, col_inds = np.array(row_inds), np.array(col_inds)
        # batch_noHeads_indx = (np.arange(no_of_blocks) * self.pad_ind)[:, None]

        # row_inds = row_inds[None,:].repeat(no_of_blocks,axis=0) + batch_noHeads_indx
        # col_inds = col_inds[None,:].repeat(no_of_blocks,axis=0) + batch_noHeads_indx
        # row_inds, col_inds = list(row_inds.flatten()), list(col_inds.flatten())
        # breakpoint()

        # mult_matrix = torch.zeros((self.pad_ind * no_of_blocks,self.pad_ind *  self.pad_ind), device=self.device)
        # mult_matrix[row_inds, col_inds] = 1
        # return mult_matrix.to_sparse_csr()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int(round(self.x_embedder.num_patches ** (1/3))))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        try:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        except:
            pass

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, D, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if not self.dbl_resolution else self.x_embedder.patch_size[0]*2
        h = w = d = round(x.shape[1] ** (1/3))
        assert h * w * d == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p, c))
        x = torch.einsum('nhwdpqlc->nchpwqdl', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p, h * p))
        return imgs

    def forward(self, x, t, y, cond_drop_prob=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        B = x.shape[0]
        att_matrix = self.form_mult_matrix(B)
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training, cond_drop_prob=cond_drop_prob)    # (N, D)
        c = t + y 
        for block in self.blocks:
            if type(block) == torch.nn.modules.container.Sequential:
                x = block(x)
            elif self.small_attention:
                x = block(x, c, att_matrix)                      # (N, T, D)    
            else:
                x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x


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

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)



class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.channels = in_channels
        self.out_dim = in_channels
        self.random_or_learned_sinusoidal_cond = False

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

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
    return pos_embed



def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W, D/2)
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])
    emb = np.concatenate([emb_h, emb_w, emb_d], axis=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)



def DiT_S_2_3D(**kwargs):
    if 'depth' not in kwargs: kwargs['depth'] = 4
    if 'hidden_size' not in kwargs: kwargs['hidden_size'] = 768
    if 'patch_size' not in kwargs: kwargs['patch_size'] = 2

    return DiT3D(num_heads=12, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}