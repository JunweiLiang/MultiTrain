#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from slowfast.models.common import DropPath, Mlp


def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None, pool2d=None):
    # junwei: pool2d is for TNN, it does not support [1,3,3] maxpool3d
    if pool is None:
        return tensor, thw_shape

    # tensor could be multi-head
    # # [B, num_head, thw, C // num_head]
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        # [B, THW, dim] -> [B, 1, THW, dim]
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    # N is the number of head
    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()


    """  # junwei: test for pool2d vs pool3d
    if pool2d is not None:
        tensor_test = pool2d(tensor.view(-1, H, W)).view(B * N, C, T, H//2, W//2)
    #print(tensor.shape) # N, C, T, H, W
    # pool kernel/stride/pad is # [1, 3, 3] [1, 2, 2] [0, 1, 1]
    tensor = pool(tensor)
    if pool2d is not None:
        print(torch.allclose(tensor, tensor_test))
    """
    # junwei: change to use pool2d
    if pool2d is not None:
        # https://discuss.pytorch.org/t/ceil-mode-in-pooling/54646/2
        # output_hw size: floor((W-K+2P)/S) + 1
        # if K=3, P=1, S=2
        # so floor[(W - 1)/2] + 1 == W/2 if W is even number,
        # for odd number it is W/2 + 1
        tensor = pool2d(tensor.view(-1, H, W)).view(B * N, C, T, H//2, W//2)
    else:
        tensor = pool(tensor)  # could be maxpool or conv

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, tensor.shape[1], L_pooled).transpose(2, 3)

    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
        L_pooled = torch.add(L_pooled, 1)

    if norm is not None:
        tensor = norm(tensor)
    # Assert tensor_dim in [3, 4]
    # 3 [8, 14, 14] torch.Size([1, 1, 1568, 384])
    # 4 [8, 7, 7] torch.Size([1, 4, 392, 96])
    #print(tensor_dim, thw_shape, tensor.shape)
    # pool: Conv3d or MaxPool3d
    #print(tensor_dim, thw_shape, isinstance(pool, nn.MaxPool3d))
    #if tensor_dim == 4:
    #    pass
    #else:  #  tensor_dim == 3:
    if tensor_dim == 3:
        # [B, 1, THW, dim] -> [B, THW, dim]
        # this with CLS_EMBED_ON=False, ONNX will generate onnx::If
        # and TensorRT v8.0 will complain
        #tensor = tensor.squeeze(1)
        tensor = tensor.reshape(B, L_pooled, tensor.shape[3])
    return tensor, thw_shape


def cal_rel_pos_spatial(
    attn,
    q,
    has_cls_embed,
    q_shape,
    k_shape,
    rel_pos_h, # (2*h-1, dim)
    rel_pos_w,
):
    """
    Spatial Relative Positional Embeddings.
     attn [B, num_head, thw, thw']
     q [B, num_head, thw, dim]
     k_shape thw', [T, H, W]
    """
    sp_idx = 1 if has_cls_embed else 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)

    # compute the lookup table for getting the positional embedding

    # if q_hw and k_hw the same
    # size is H/W
    # got a (H, W) matrix,
    # [0, -1, ... -size+1] size
    # ..
    # [size-1, .... 0]
    # value from -size+1 to size-1
    # we got [q_h, k_h] matrix here
    dist_h = (
        torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    )
    # then
    # [size, size-1, ... 0]
    # ..
    # [2*size-1, .... size-1]
    # value from 0 to 2*size -1
    dist_h += (k_h - 1) * k_h_ratio


    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    # rel_pos_h [2*size-1, dim] -> [size, size, D]
    # [H, K, D]
    # [q_h, k_h, dim]
    Rh = rel_pos_h[dist_h.long()]
    # [q_w, k_w, dim]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    # [B, head, q_h, q_w, D], [q_h, k_h, D] -> [B, head, q_h, q_w, k_h]
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    # [B, head, q_h, q_w, D], [q_w, k_w, D] -> [B, head, q_h, q_w, k_w]
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        # [B, head, H, W, H, W]
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None] # [B, head, H, W, H, 1]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


def skew(QEr, is_sym=False):
    """
        Given a tensor of [B, num_head, _, _, L, 2L-1], 
        (is_sym would be [B, num_head, _, _, L, L])
        get [B, num_head, _, _, L, L]
    """
    
    B, num_head, S, T, L, _ = QEr.shape

    if is_sym:
        # pad_size is from the last dim, (pad_left_dim-1, pad_right_dim-1, pad_left_dim-2, pad_right_dim-2, ..)
        # [B, num_head, S, T, L, L] -> # [B, num_head, S, T, L, L + 1]
        padded = F.pad(QEr, (1, 0))  # left pad
        # [B, num_head, S, T, L+1, L]
        reshaped = padded.reshape(B, num_head, S, T, L+1, L)

        sliced = reshaped[:, :, :, :, 1:, :]

        # TODO(junwei): consider q_len != k_len
        # https://github.com/jason9693/MusicTransformer-pytorch/blob/fe296678c99030adb073768112b54ee6cebef907/custom/layers.py#L121

    else:

        padded = F.pad(QEr, (0, 1))  # right padding to the last dim

        flatten = torch.flatten(padded, 4)  #[B, num_head, S, T, 2L*L]

        flatten_padded = F.pad(flatten, (0, L-1))  # [B, num_head, S, T, 2L*L + L-1]

        reshaped = flatten_padded.reshape(B, num_head, S, T, L+1, 2*L - 1)

        sliced = reshaped[:, :, :, :, :L, L-1:]

        #sliced = reshaped[:, :, :, :, :L, :L]
        #sliced = torch.flip(sliced, dims=[5])
    return sliced


def cal_rel_pos_spatialtemporal_skew(
    attn,
    q,
    q_shape,
    k_shape,
    rel_pos_h, # (2*h-1, dim)
    rel_pos_w, # (2*w-1, dim)
    rel_pos_t, # (2*t-1, dim)
    is_sym=False,  # if is_sym, the rel_pos should be L length
    debug=False,
):
    """
    Spatial-temporal Relative Positional Embeddings by junwei
     attn [B, num_head, thw, thw']
     q [B, num_head, thw, dim]
     k_shape thw', [T, H, W]
     # assuming no classification token

     # TODO(junwei): check out https://openreview.net/pdf?id=rJe4ShAcF7
     # to reduce space complexity, we could multiply Q and E and then avoid the indexing
    """

    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    assert q_shape == k_shape, "the skewing process needs the shape to be the same"
    B, n_head, q_N, dim = q.shape  # q_N: THW

    # assuming q_h and k_h the same
    r_q = q[:, :, :].reshape(B, n_head, q_t, q_h, q_w, dim)


    # [B, H, q_t, q_h, q_w, dim] -> [B, H, q_t, q_w, q_h, dim]
    r_q_h = r_q.permute(0, 1, 2, 4, 3, 5)
    # [B, num_head, q_t, q_w, q_h, dim] -> [dim, 2h-1] -> [B, num_head, q_t, q_w, q_h, 2h-1]
    QEr_h = torch.matmul(r_q_h, rel_pos_h.transpose(1, 0))

    # [B, num_head, q_t, q_w, q_h, q_h]
    rel_h = skew(QEr_h, is_sym=is_sym)

    # [B, num_head, q_t, q_h, q_w, q_h]
    rel_h = rel_h.permute(0, 1, 2, 4, 3, 5)
   

    # [B, num_head, q_t, q_h, q_w, dim] -> [dim, 2w-1] -> [B, num_head, q_t, q_h, q_w, 2w-1]
    QEr_w = torch.matmul(r_q, rel_pos_w.transpose(1, 0))
    # [B, num_head, q_t, q_h, q_w, q_w]
    rel_w = skew(QEr_w, is_sym=is_sym)

    # [B, H, q_t, q_h, q_w, dim] -> [B, H, q_h, q_w, q_t, dim]
    r_q_t = r_q.permute(0, 1, 3, 4, 2, 5)
    # [B, num_head, q_h, q_w, q_t, dim] -> [dim, 2t-1] -> [B, num_head, q_h, q_w, q_t, 2t-1]
    QEr_t = torch.matmul(r_q_t, rel_pos_t.transpose(1, 0))
    # [B, num_head, q_h, q_w, q_t, q_t]
    rel_t = skew(QEr_t, is_sym=is_sym)
    rel_t = rel_t.permute(0, 1, 4, 2, 3, 5) # [B, num_head, q_t, q_h, q_w, q_t]

    #print(attn.shape, q_shape, k_shape, rel_h.shape, rel_w.shape, rel_t.shape)
    attn[:, :, :, :] = (
        attn[:, :, :, :].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
        + rel_h[:, :, :, :, :, None, :, None]
        + rel_w[:, :, :, :, :, None, None, :]
        + rel_t[:, :, :, :, :, :, None, None]
    ).view(B, -1, q_t*q_h * q_w, k_t * k_h * k_w)

    if debug:
        return attn, rel_t, rel_h, rel_w
    else:
        return attn

def cal_rel_pos_spatialtemporal(
    attn,
    q,
    q_shape,
    k_shape,
    rel_pos_h, # (2*h-1, dim)
    rel_pos_w, # (2*w-1, dim)
    rel_pos_t, # (2*t-1, dim)
    no_einsum=False,
    is_sym=False,  # whether the relative position is symmetric (L length or 2L-1)
    debug=False,
    use_skew_ordering=False,  # this ordering would be consistent with Huang et. al and Shaw et. al.
):
    """
    Spatial-temporal Relative Positional Embeddings by junwei
     attn [B, num_head, thw, thw']
     q [B, num_head, thw, dim]
     k_shape thw', [T, H, W]
     # assuming no classification token

     # TODO(junwei): check out https://openreview.net/pdf?id=rJe4ShAcF7
     # to reduce space complexity, we could multiply Q and E and then avoid the indexing
    """
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    # assume the temporal does not change
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)

    # compute the lookup table for getting the positional embedding

    # if q_hw and k_hw the same
    # got a (H, W) matrix,
    # [0, -1, ... -size+1] size
    # ..
    # [size-1, .... 0]
    # value from -size+1 to size-1
    dist_h = (
        torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    )
    # then
    # [size-1, size-2, ... 0]
    # ..
    # [2*size-2, .... size-1]
    # value from 0 to 2*size -2
    if not is_sym: 
        dist_h += (k_h - 1) * k_h_ratio
        
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)

    # for q_w==56, k_w==14
    # [0, ..., -52]
    # ...
    # [55, ..., 3]]
    # (56, 14)
    dist_w = (
        torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    )
    if not is_sym:
        # then 
        # [52, ..., 0]
        # ...
        # [107,...,55]
        dist_w += (k_w - 1) * k_w_ratio

    # assuming temporal resolution does not change
    dist_t = (
        torch.arange(q_t)[:, None] - torch.arange(k_t)[None, :]
    )
    if not is_sym:
        dist_t += (k_t - 1) # value from 0 to 2*size -2
        #dist_t = 2*k_t - 2 - dist_t


    if use_skew_ordering:
        assert not is_sym, "not working yet"
        if q_shape != k_shape:

            dist_t = 2*(k_t - 1) - dist_t

            dist_h = 2*(k_h - 1) * k_h_ratio - dist_h

            dist_w = 2*(k_w - 1) * k_w_ratio - dist_w                

        else:
            # [size-1, size, ... 2*size-2]
            # ..
            # [0, .... size-1]
            dist_h = dist_h.T
            dist_w = dist_w.T
            dist_t = dist_t.T

    # rel_pos_h [2*size-1, dim] -> [size, size, dim]
    # here are the relative embedding of the attention matrix [LxL]
    # [q_H, K_h, head_dim]
    Rh = rel_pos_h[dist_h.long()]
    # [q_W, K_w, head_dim]
    Rw = rel_pos_w[dist_w.long()] # every block has one different embedding
    # [q_T, K_t, head_dim]
    Rt = rel_pos_t[dist_t.long()]

    B, n_head, q_N, dim = q.shape  # q_N: THW

    # q [B, num_head, thw, dim]
    # here we compute the E from the paper, by dot-product of Q and RPE
    r_q = q[:, :, :].reshape(B, n_head, q_t, q_h, q_w, dim)

    if no_einsum:
        # without einsum
        # https://github.com/facebookresearch/SlowFast/blob/main/slowfast/models/attention.py#L144

        # [B, H, q_t, q_h, q_w, dim] -> [q_h, B, H, q_t, q_w, dim] -> [q_h, B*H*q_t*q_w, dim]
        r_q_h = r_q.permute(3, 0, 1, 2, 4, 5).reshape(
            q_h, B * n_head * q_t * q_w, dim
        )
        # [q_h, B*H*q_t*q_w, dim] * [q_h, dim, k_h] = [q_h, B*H*q_t*q_w, k_h] -> [B*H*q_t*q_w, q_h, k_h]
        rel_h = torch.matmul(r_q_h, Rh.transpose(1, 2)).transpose(0, 1)
        # [B*H*q_t*q_w, q_h, k_h] -> [B, H, q_t, q_h, q_w, k_h]
        rel_h = rel_h.view(B, n_head, q_t, q_w, q_h, k_h).permute(0, 1, 2, 4, 3, 5)

        # [B, H, q_t, q_h, q_w, dim] -> [q_w, B, H, q_t, q_h, dim] -> [q_w, B*H*q_t*q_h, dim]
        r_q_w = r_q.permute(4, 0, 1, 2, 3, 5).reshape(
            q_w, B * n_head * q_t * q_h, dim
        )
        # [q_w, B*H*q_t*q_h, dim] * [q_w, dim, k_w] = [q_w, B*H*q_t*q_h, k_w] -> [B*H*q_t*q_h, q_w, k_w]
        rel_w = torch.matmul(r_q_w, Rw.transpose(1, 2)).transpose(0, 1)
        # [B*H*q_t*q_h, q_w, k_w] -> [B, H, q_t, q_h, q_w, k_w]
        rel_w = rel_w.view(B, n_head, q_t, q_h, q_w, k_w)


        # [B, H, q_t, q_h, q_w, dim] -> [q_t, B, H, q_h, q_w, dim] -> [q_t, B*H*q_h*q_w, dim]
        r_q_t = r_q.permute(2, 0, 1, 3, 4, 5).reshape(
            q_t, B * n_head * q_h * q_w, dim
        )
        # [q_t, B*H*q_h*q_w, dim] * [q_t, dim, k_t] = [q_t, B*H*q_h*q_w, k_t] -> [B*H*q_h*q_w, q_t, k_t]
        # here the complexity if k_t^2 * dim, but since Rt is of 2L-1, there are a lot of repeated computation
        rel_t = torch.matmul(r_q_t, Rt.transpose(1, 2)).transpose(0, 1)
        # [B*H*q_h*q_w, q_t, k_t] -> [B, H, q_t, q_h, q_w, k_t]
        rel_t = rel_t.view(B, n_head, q_h, q_w, q_t, k_t).permute(0, 1, 4, 2, 3, 5)

    else:
        # [B, head, T, H, W, K(H)]
        rel_h = torch.einsum("bythwc,hkc->bythwk", r_q, Rh)
        # [B, head, T, H, W, K(W)]
        rel_w = torch.einsum("bythwc,wkc->bythwk", r_q, Rw)
        # [B, head, T, H, W, K(T)]
        rel_t = torch.einsum("bythwc,tkc->bythwk", r_q, Rt)



    # q_shape, [1, 56, 56]
    # k_shape, [1, 14, 14]
    #print(attn.shape, q_shape, k_shape)
    attn[:, :, :, :] = (
        attn[:, :, :, :].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
        + rel_h[:, :, :, :, :, None, :, None]
        + rel_w[:, :, :, :, :, None, None, :]
        + rel_t[:, :, :, :, :, :, None, None]
    ).view(B, -1, q_t*q_h * q_w, k_t*k_h* k_w)

    if debug:
        return attn, rel_t, rel_h, rel_w
    else:
        return attn





class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        # Options include `conv`, `avg`, and `max`.
        mode="conv",
        pool_first=False,# from MeMViT paper
        # added by junwei for MViT version 2
        use_query_residual_pool=False,
        # we expand the channel in the last project?
        expand_channel=False,
        expand_to_dim=None,
        # for relativ pos emb
        use_rel_pos=False,
        rel_pos_zero_init=False,
        input_size=None,
        no_einsum=False,
        use_skew=False,
        use_sym_rel=False,
        ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        self.no_einsum = no_einsum

        self.use_skew = use_skew

        dim_in = dim
        dim_out = dim
        if expand_channel:
            dim_out = expand_to_dim
        self.dim_out = dim_out

        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.pool_first = pool_first
        if self.pool_first:
            self.lin_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
            self.lin_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
            self.lin_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim_in, dim_out * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)

        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()

        if mode == "avg":
            self.pool_q = (
                nn.AvgPool3d(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                nn.AvgPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                nn.AvgPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "max":
            self.pool_q = (
                nn.MaxPool3d(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                nn.MaxPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                nn.MaxPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )

        elif mode == "conv":  # this is default for MViT 16x4, 32x3 model from the paper
            if self.pool_first:
                dim_conv = dim_in // num_heads
            else:
                dim_conv = head_dim

            # input is # [B, num_head, thw, C // num_head]
            self.pool_q = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
            self.pool_v = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")


        # junwei: https://arxiv.org/pdf/2112.01526v1.pdf

        self.use_query_residual_pool = use_query_residual_pool

        # relative pos embedding
        self.use_rel_pos = use_rel_pos
        self.use_sym_rel = use_sym_rel
        if self.use_rel_pos:

            # this version consider relative position front and back are different
            # meaning L length will have 2L - 1 position
            # TODO(junwei): use L length L position, meaning front and back relative are the same
            # symmetric relative position
            # implemented by Junwei
            # input size is [T, H, W]
            assert input_size[1] == input_size[2]

            size = input_size[1]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            if self.use_sym_rel:
                # h == w, 2*h - 1
                rel_sp_dim = max(q_size, kv_size)
                rel_t_dim = input_size[0]  # assuming no temporal stride
            else:
                # h == w, 2*h - 1
                rel_sp_dim = 2 * max(q_size, kv_size) - 1
                rel_t_dim = 2 * input_size[0] - 1  # assuming no temporal stride

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_t = nn.Parameter(torch.zeros(rel_t_dim, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_t, std=0.02)


    def forward(self, x, thw_shape):

        # N is thw
        B, N, C = x.shape
        C = self.dim_out

        if self.pool_first:
            # [B, N, C] -> [B, num_heads, N, C // num_heads]
            x = x.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

            # [B, num_heads, N', C // num_heads]
            q, q_shape = attention_pool(
                x,
                self.pool_q,
                thw_shape,
                has_cls_embed=self.has_cls_embed,
                norm=self.norm_q if hasattr(self, "norm_q") else None,
            )
            k, k_shape = attention_pool(
                x,
                self.pool_k,
                thw_shape,
                has_cls_embed=self.has_cls_embed,
                norm=self.norm_k if hasattr(self, "norm_k") else None,
            )
            v, v_shape = attention_pool(
                x,
                self.pool_v,
                thw_shape,
                has_cls_embed=self.has_cls_embed,
                norm=self.norm_v if hasattr(self, "norm_v") else None,
            )

            # now we do the linear
            # [B, num_heads, N', C // num_heads] -> [B, N', D]
            q_N = np.prod(q_shape)
            k_N = np.prod(k_shape)
            v_N = np.prod(v_shape)

            # last dim: self.dim_conv * self.num_heads
            q = q.permute(0, 2, 1, 3).reshape(B, q_N, -1)
            k = k.permute(0, 2, 1, 3).reshape(B, k_N, -1)
            v = v.permute(0, 2, 1, 3).reshape(B, v_N, -1)

            # [B, THW', C'] -> [B, num_heads, THW', C' // num_heads]
            q = self.lin_q(q).reshape(B, q_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = self.lin_k(k).reshape(B, k_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.lin_v(v).reshape(B, v_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            # x linearly projected to query, key, value tensors
            # [B, N, C] -> [B, N, 3*C] -> each [B, N, num_head, C//num_head]
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            # [B, num_head, thw, C // num_head]
            q, k, v = qkv[0], qkv[1], qkv[2]

            # pool with Conv3D
            # [B, num_head, thw, C // num_head] -> [B, num_head, new_thw, C // num_head]
            # so the final output only depends on query's new_thw
            q, q_shape = attention_pool(
                q,
                self.pool_q,
                thw_shape,
                has_cls_embed=self.has_cls_embed,
                norm=self.norm_q if hasattr(self, "norm_q") else None,
            )
            k, k_shape = attention_pool(
                k,
                self.pool_k,
                thw_shape,
                has_cls_embed=self.has_cls_embed,
                norm=self.norm_k if hasattr(self, "norm_k") else None,
            )
            v, v_shape = attention_pool(
                v,
                self.pool_v,
                thw_shape,
                has_cls_embed=self.has_cls_embed,
                norm=self.norm_v if hasattr(self, "norm_v") else None,
            )

        # matmul
        # each token and head in query attends each token and head in key
        # q [B, num_head, new_thw, dim] -> k [B, num_head, dim, new_thw']
        # new_thw and new_thw' could be different

        # attn [B, num_head, new_thw, new_thw']
        # below should get the same results
        # in terms of flops, depend on q and k size vs the head_dim

        #if not self.fixed_scale:
            # pyslowfast old code
        #    attn = (q @ k.transpose(-2, -1)) * self.scale
        # MViT v2 official code:
        # https://github.com/facebookresearch/mvit
        # this should has less flops
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:

            if self.use_skew and q_shape == k_shape:
                # there will be four block that has different q_shape and k_shape

                """
                assert not self.use_sym_rel, "sym rel with skew not working yet"
                # for debugging
                attn_ori, rel_t, rel_h, rel_w = cal_rel_pos_spatialtemporal(
                    attn.clone(),
                    q,
                    q_shape,
                    k_shape,
                    self.rel_pos_h,
                    self.rel_pos_w,
                    self.rel_pos_t,
                    no_einsum=self.no_einsum,
                    is_sym=self.use_sym_rel,
                    use_skew_ordering=self.use_skew,  
                    debug=True,
                )
                
                
                
                attn_skew, rel_ts, rel_hs, rel_ws = cal_rel_pos_spatialtemporal_skew(
                    attn,
                    q,
                    q_shape,
                    k_shape,
                    self.rel_pos_h,
                    self.rel_pos_w,
                    self.rel_pos_t,
                    is_sym=self.use_sym_rel,
                    debug=True,
                )

                # [32, 4, 196, 196]
                # [B, num_head, THW, THW]
                #print(attn.dtype) # float32
                #print(attn.shape, attn_ori.shape, attn[0, 0, 1], attn_ori[0, 0, 1])                
                
                assert torch.allclose(rel_t, rel_ts)
                # [B, num_head, T, H, W, W]
                assert torch.allclose(rel_w, rel_ws)
                assert torch.allclose(rel_h, rel_hs)

                assert torch.allclose(attn_skew, attn_ori)

                attn = attn_ori
                """
                attn = cal_rel_pos_spatialtemporal_skew(
                    attn,
                    q,
                    q_shape,
                    k_shape,
                    self.rel_pos_h,
                    self.rel_pos_w,
                    self.rel_pos_t,
                    is_sym=self.use_sym_rel,
                )
                
                

            else:
                attn = cal_rel_pos_spatialtemporal(
                    attn,
                    q,
                    q_shape,
                    k_shape,
                    self.rel_pos_h,
                    self.rel_pos_w,
                    self.rel_pos_t,
                    no_einsum=self.no_einsum,
                    is_sym=self.use_sym_rel,
                    use_skew_ordering=self.use_skew,  # still need the correct ordering
                )

        # attn: [B, num_head, new_thw, new_thw']
        attn = attn.softmax(dim=-1)

        N = q.shape[2]
        # attn [B, num_head, new_thw, new_thw'] -> v [B, num_head, new_thw', dim]
        # -> [B, num_head, new_thw, dim]
        # -> [B, new_thw (q_shape), num_head*dim]
        # so we have attention from each token in the query to all of value's token,

        """
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.use_query_residual_pool:
            # q [B, num_head, new_thw, dim]
            x = x + q.transpose(1, 2).reshape(B, N, C)  # this does not add to FLOPs?
        """
        # [B, new_thw (q_shape), num_head*dim]
        x = attn @ v

        if self.use_query_residual_pool:
            x = x + q

        x = x.transpose(1, 2).reshape(B, N, C)


        # linear, dim -> dim
        x = self.proj(x)
        if self.drop_rate > 0.0:  # drop out, 0.0 by default
            x = self.proj_drop(x)
        return x, q_shape


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        up_rate=None,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        mode="conv",
        has_cls_embed=True,
        pool_first=False,
        use_query_residual_pool=False,
        channel_expand_front=False,
        pool_skip_use_conv=False,
        use_rel_pos=False,
        use_sym_rel=False,
        rel_pos_zero_init=False, # used in MViTv2 paper, from IN model to Kinetics for temporal
        no_einsum=False,
        use_skew=False,
        input_size=None,
        fixed_scale=False,
        # from the MeMViT paper
        use_mem=False,
        mem_compress_factor=(4, 2, 2),
        max_mem_len=2,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]

        # junwei: MViT Version 2, dim became 2*dim during attention compute
        expand_channel = False
        dim_in = dim
        if channel_expand_front:
            if dim != dim_out:
                expand_channel = True
        self.expand_channel = expand_channel
        self.pool_skip_use_conv = pool_skip_use_conv

        if use_mem:
            self.attn = MeMViTAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                kernel_q=kernel_q,
                kernel_kv=kernel_kv,
                stride_q=stride_q,
                stride_kv=stride_kv,
                norm_layer=nn.LayerNorm,
                has_cls_embed=has_cls_embed,
                mode=mode,
                use_query_residual_pool=use_query_residual_pool,
                expand_channel=expand_channel,
                expand_to_dim=dim_out,
                use_rel_pos=use_rel_pos,
                use_sym_rel=use_sym_rel,
                rel_pos_zero_init=rel_pos_zero_init,
                input_size=input_size,
                no_einsum=no_einsum,
                use_skew=use_skew,
                pool_first=pool_first,
                max_mem_len=max_mem_len,
                mem_compress_factor=mem_compress_factor,
            )
        else:
            self.attn = MultiScaleAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                kernel_q=kernel_q,
                kernel_kv=kernel_kv,
                stride_q=stride_q,
                stride_kv=stride_kv,
                norm_layer=nn.LayerNorm,
                has_cls_embed=has_cls_embed,
                mode=mode,
                use_query_residual_pool=use_query_residual_pool,
                expand_channel=expand_channel,
                expand_to_dim=dim_out,
                use_rel_pos=use_rel_pos,
                use_sym_rel=use_sym_rel,
                rel_pos_zero_init=rel_pos_zero_init,
                input_size=input_size,
                no_einsum=no_einsum,
                use_skew=use_skew,
                pool_first=pool_first,
            )
        if self.expand_channel:
            dim = dim_out
            self.dim = dim_out

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)  # normalize along last dimension
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed

        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )
        if dim != dim_out:
            # this is never used for MViTv2 full
            self.proj = nn.Linear(dim, dim_out)

        # this does not seem to be used in MViT V2
        """
        if pool_skip_use_conv:
            self.pool_skip = (
                nn.Conv3d(
                    dim_in, dim_out,
                    kernel_skip, stride_skip, padding_skip, bias=False
                )
                if len(kernel_skip) > 0
                else None
            )
            self.pool_skip_norm = nn.LayerNorm(dim_out)
        else:
        """
        if self.expand_channel:
            self.proj_max_pool = nn.Linear(dim_in, dim_out)

        # pooling for the skip connections
        self.pool_skip = (
            nn.MaxPool3d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
            if len(kernel_skip) > 0
            else None
        )
        self.pool_skip_norm = None
        #print(kernel_skip)  # empty for 16x4 and 32x3 model
        # junwei: 12/2021, use MaxPool2d instead
        # for both 16x4, 32x3, some blocks are [] some are [1, 3, 3] for kernel_skip
        # [1, 3, 3] [1, 2, 2] [0, 1, 1]
        #print(kernel_skip, stride_skip, padding_skip)
        """
        self.pool_skip2d = (
            nn.MaxPool2d(
                kernel_skip[1:], stride_skip[1:], padding_skip[1:], ceil_mode=False
            )
            if len(kernel_skip) > 0 and stride_skip[1] != 1
            else None
        )
        """

    def forward(self, x, thw_shape):
        # x: [B, (1+)thw, channel]
        # layernorm -> attention
        # [8, 56, 56] [8, 28, 28], changed 3 times for MViT 16x4
        # there are attention_pool within attn() as well

        # here the attentionblock will pool the feature from [B, THW, C]
        # to [B, new_THW, C]
        # the query of self attention block determines the output shape
        x_block, thw_shape_new = self.attn(self.norm1(x), thw_shape)
        # attention pool
        # x_res: torch.Size([1, 6273, 192])
        if not self.pool_skip_use_conv and self.expand_channel:
            # need to change the channel for the residual connection
            x = self.proj_max_pool(x)
        x_res, _ = attention_pool(
            x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed,
            norm=self.pool_skip_norm,
            # pool2d does not support (2, 3, 3) yet
            #pool2d=self.pool_skip2d  # not using it during training/testing
        )
        # x_res should be pooled to thw_shape_new as well
        x = x_res + self.drop_path(x_block)

        x_norm = self.norm2(x)

        x_mlp = self.mlp(x_norm)

        # in this case x is normed before +, but it may not be normed if dim equals
        if self.dim != self.dim_out:
            # residual connect for the MLP block
            # this is never used for MViTv2 full
            x = self.proj(x_norm)

        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new


# ------- for MeMViT impl

def compress_mem(tensor, pool, thw_shape, num_heads):

    # tensor could be multi-head
    # # [B, THW, C]  -> [B, num_head, THW, C//num_head] -> [B, num_head, THW', C//num_head] -> [B, THW', C]
    tensor_dim = tensor.ndim
    if tensor_dim == 3:
        pass
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    B, L, C = tensor.shape
    T, H, W = thw_shape
    conv_dim = C // num_heads
    tensor = tensor.reshape(B, T, H, W, num_heads, conv_dim).permute(0, 4, 1, 2, 3, 5)
    # [-1, C, T, H, W]
    tensor = tensor.reshape(B * num_heads, T, H, W, conv_dim).permute(0, 4, 1, 2, 3).contiguous()


    tensor = pool(tensor)  # could be maxpool or conv

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    
    tensor = tensor.reshape(B, num_heads, conv_dim, tensor.shape[2], tensor.shape[3], tensor.shape[4])
    # [B, N, conv_dim, T, H, W] -> [B, T, H, W, N, conv_dim] -> [B, THW', C]
    tensor = tensor.permute(0, 3, 4, 5, 1, 2).reshape(B, L_pooled, C)

    return tensor, thw_shape


class MeMViTAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        # Options include `conv`, `avg`, and `max`.
        mode="conv",
        pool_first=False,# from MeMViT paper
        # added by junwei for MViT version 2
        use_query_residual_pool=False,
        # we expand the channel in the last project?
        expand_channel=False,
        expand_to_dim=None,
        # for relativ pos emb
        use_rel_pos=False,
        rel_pos_zero_init=False,
        input_size=None,
        no_einsum=False,
        use_skew=False,
        use_sym_rel=False,
        # for MeMViT
        max_mem_len=2,
        mem_compress_factor=(4, 2, 2),
        ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        self.no_einsum = no_einsum

        self.use_skew = use_skew
        assert not use_skew
        assert not use_rel_pos, "not supported yet"

        dim_in = dim
        dim_out = dim
        if expand_channel:
            dim_out = expand_to_dim
        self.dim_out = dim_out

        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        assert not has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.pool_first = pool_first

        assert pool_first
        # always pool_first
        self.lin_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.lin_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.lin_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.proj = nn.Linear(dim_out, dim_out)

        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()

        assert mode == "conv"
        dim_conv = dim_in // num_heads
        self.dim_conv = dim_conv

        # input is # [B, num_head, thw, C // num_head]
        self.pool_q = (
            nn.Conv3d(
                dim_conv,
                dim_conv,
                kernel_q,
                stride=stride_q,
                padding=padding_q,
                groups=dim_conv,
                bias=False,
            )
            if len(kernel_q) > 0
            else None
        )
        self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None
        self.pool_k = (
            nn.Conv3d(
                dim_conv,
                dim_conv,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=dim_conv,
                bias=False,
            )
            if len(kernel_kv) > 0
            else None
        )
        self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
        self.pool_v = (
            nn.Conv3d(
                dim_conv,
                dim_conv,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=dim_conv,
                bias=False,
            )
            if len(kernel_kv) > 0
            else None
        )
        self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None


        self.use_query_residual_pool = use_query_residual_pool

        # relative pos embedding
        self.use_rel_pos = use_rel_pos
        self.use_sym_rel = use_sym_rel

        # this version consider relative position front and back are different
        # meaning L length will have 2L - 1 position
        # TODO(junwei): use L length L position, meaning front and back relative are the same
        # symmetric relative position
        # implemented by Junwei
        # input size is [T, H, W]
        assert input_size[1] == input_size[2]

        size = input_size[1]
        q_size = size // stride_q[1] if len(stride_q) > 0 else size
        kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size

        if self.use_rel_pos:
            if self.use_sym_rel:
                # h == w, 2*h - 1
                rel_sp_dim = max(q_size, kv_size)
                rel_t_dim = input_size[0]  # assuming no temporal stride
            else:
                # h == w, 2*h - 1
                rel_sp_dim = 2 * max(q_size, kv_size) - 1
                rel_t_dim = 2 * input_size[0] - 1  # assuming no temporal stride

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_t = nn.Parameter(torch.zeros(rel_t_dim, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_t, std=0.02)


        # for MeMViT
        self.max_mem_len = max_mem_len
        self.mem_compress_factor = mem_compress_factor
        if self.max_mem_len > 0:
            self.m_k = []
            self.m_v = []

            # module for compressing the memory (THW -> THW^m)
            # see MeMViT suppl for details
            self.f_k = nn.Conv3d(
                dim_conv,
                dim_conv,
                # this is according to paper, but will raise errors as the input size is only [8x7x7]
                #[2*o+1 for o in self.mem_compress_factor], 
                [3, 3, 3],
                stride=self.mem_compress_factor,
                #padding=padding_kv,
                groups=dim_conv,
                bias=False,
            )
            self.f_v = nn.Conv3d(
                dim_conv,
                dim_conv,
                [3, 3, 3],
                stride=self.mem_compress_factor,
                #padding=padding_kv,
                groups=dim_conv,
                bias=False,
            )


    def forward(self, x, thw_shape):

        # N is thw
        B, N, dim_in = x.shape
        C = self.dim_out

        # [B, N, C] -> [B, num_heads, N, C // num_heads]
        x = x.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        # [B, num_heads, N', C // num_heads]
        q, q_shape = attention_pool(
            x,
            self.pool_q,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        k, k_shape = attention_pool(
            x,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, v_shape = attention_pool(
            x,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )


        # now we do the linear
        # [B, num_heads, N', C // num_heads] -> [B, N', D]
        q_N = np.prod(q_shape)
        k_N = np.prod(k_shape)
        v_N = np.prod(v_shape)

        q = q.permute(0, 2, 1, 3).reshape(B, q_N, self.dim_conv * self.num_heads)
        k = k.permute(0, 2, 1, 3).reshape(B, k_N, self.dim_conv * self.num_heads)
        v = v.permute(0, 2, 1, 3).reshape(B, v_N, self.dim_conv * self.num_heads)

        if self.max_mem_len > 0:
            if self.m_k:
                # the tensor in m_k are [B, thw^m, C]

                # shape after compression
                #cm_thw_shape = [max(o//cm_o, 1) for o, cm_o in zip(k_shape, self.mem_compress_factor)]

                # [B, N^m, C]
                #print(self.m_k[-1].shape)
                # torch.Size([1, 392, 96])
                # torch.Size([1, 1568, 96])
                cm_k, cm_k_shape = compress_mem(
                    self.m_k[-1],
                    self.f_k,
                    k_shape,
                    num_heads=self.num_heads
                ) 
                # [1, 14, 14] -> [1, 7, 7]
                #print(cm_k_shape)  # [1, 7, 7]

                cm_v, cm_v_shape = compress_mem(
                    self.m_v[-1],
                    self.f_v,
                    v_shape,
                    num_heads=self.num_heads
                ) 

                # need to reshape for k and v concat
                # could be empty list
                # a list of [B, THW^m, C]
                this_m_k_list = self.m_k[:-1]
                this_m_v_list = self.m_v[:-1]
                

                # cache the memory
                self.m_k[-1] = cm_k.detach()
                self.m_v[-1] = cm_v.detach()

                self.m_k.append(k)
                self.m_v.append(v)  # [B, THW', C]

                if len(self.m_k) > self.max_mem_len:
                    self.m_k.pop(0)
                    self.m_v.pop(0)

                # [B, length, C]
                k = torch.cat(this_m_k_list + [cm_k, k], dim=1)
                v = torch.cat(this_m_v_list + [cm_v, v], dim=1)

                k_N = k.shape[1]
                v_N = v.shape[1]

            else:
                self.m_k.append(k)
                self.m_v.append(v)  # [B, THW', C]

        # [B, THW', C'] -> [B, num_heads, THW', C' // num_heads]
        q = self.lin_q(q).reshape(B, q_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        k = self.lin_k(k).reshape(B, k_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.lin_v(v).reshape(B, v_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # matmul

        attn = (q * self.scale) @ k.transpose(-2, -1)
        
        
        # junwei: how to determine t?
        # junwei: not supporting this yet,
        # [1, 56, 56] [1, 14, 14] torch.Size([1, 1, 1x56x56, 96]) torch.Size([1, 1, 1x56x56, 1x7x7 + 1x14x14])
        #print(q_shape, k_shape, q.shape, attn.shape)
        """  # THW, mem_length + THW'
        attn = cal_rel_pos_spatialtemporal(
            attn,
            q,
            q_shape,
            k_shape,
            self.rel_pos_h,
            self.rel_pos_w,
            self.rel_pos_t,
            no_einsum=self.no_einsum,
            is_sym=self.use_sym_rel,
            use_skew_ordering=False,  
        )
        """
        

        # attn: [B, num_head, new_thw, new_thw']
        attn = attn.softmax(dim=-1)

        N = q.shape[2]
        # attn [B, num_head, new_thw, new_thw'] -> v [B, num_head, new_thw', dim]
        # -> [B, num_head, new_thw, dim]
        # -> [B, new_thw (q_shape), num_head*dim]
        # so we have attention from each token in the query to all of value's token,

        # [B, new_thw (q_shape), num_head*dim]
        x = attn @ v

        if self.use_query_residual_pool:
            x = x + q

        x = x.transpose(1, 2).reshape(B, N, C)

        # linear, dim -> dim
        x = self.proj(x)
        if self.drop_rate > 0.0:  # drop out, 0.0 by default
            x = self.proj_drop(x)
        return x, q_shape

    def reset_mem(self):
        # reset the memory
        self.m_k = []
        self.m_v = []

