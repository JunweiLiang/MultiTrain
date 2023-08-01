# coding=utf-8


"""Video models."""

import math
from functools import partial
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.attention import MultiScaleBlock
from slowfast.models.utils import round_width, validate_checkpoint_wrapper_import
from slowfast.models.common import Mlp

# for contrastive learning
#from slowfast.models.text_models import Transformer

from . import head_helper
from .build import MODEL_REGISTRY

from copy import deepcopy
import numpy as np

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None



@MODEL_REGISTRY.register()
class MViT(nn.Module):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg

        # by junwei, Improved MViT not using CLS_EMB already
        #assert not cfg.MVIT.CLS_EMBED_ON, "dont use this!"
        if cfg.MVIT.CLS_EMBED_ON:
            print("warning, using CLS_EMBED_ON")

        # by junwei, MViT version 2
        # https://arxiv.org/pdf/2112.01526v1.pdf
        self.use_query_residual_pool = cfg.MVIT.Q_POOL_RESIDUAL
        self.q_pool_all = cfg.MVIT.Q_POOL_ALL
        self.channel_expand_front = cfg.MVIT.CHANNEL_EXPAND_FRONT
        self.pool_skip_use_conv = cfg.MVIT.POOL_SKIP_USE_CONV

        # whether to use x = x[0] for inputs
        self.direct_input = cfg.MVIT.DIRECT_INPUT
        if cfg.MVIT.USE_MEM:
            self.direct_input = True  # for long-term video loader, no multi-pathway packing

        self.fixed_attn_scale = cfg.MVIT.fixed_attn_scale

        # return a model for feature extraction
        self.feature_extraction = cfg.MODEL.GET_FEATURE


        pool_first = cfg.MVIT.POOL_FIRST

        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        use_2d_patch = cfg.MVIT.PATCH_2D
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if use_2d_patch:  # default False for 16x4 and 32x3 model
            self.patch_stride = [1] + self.patch_stride
        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        dim_out = embed_dim
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS  # 1
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        self.rel_pos_embed = cfg.MVIT.REL_POS_EMBED
        self.use_sym_rel = cfg.MVIT.USE_SYM_REL
        self.no_einsum = cfg.MVIT.NO_EINSUM  # not using einsum in rel pos cal
        self.use_skew = cfg.MVIT.USE_SKEW  # less mem solution
        self.rel_pos_zero_init = cfg.MVIT.REL_POS_ZERO_INIT

        # MeMViT
        self.use_mem = cfg.MVIT.USE_MEM
        if self.use_mem:
            assert not self.rel_pos_embed and self.sep_pos_embed, "only absolute emb supported yet"
            
        self.mem_compress_factor = cfg.MVIT.MEM_COMPRESS_FACTOR
        self.max_mem_len = cfg.MVIT.MAX_MEM_LEN

        if self.rel_pos_embed:
            assert not self.sep_pos_embed, "Using relative embedding will need to disable sep_pos_embed"
        if cfg.MVIT.NORM == "layernorm":
            # create a new function with this eps
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")

        # activation checkpointing to save GPU memory
        if cfg.MODEL.ACT_CHECKPOINT:
            # check for
            # from fairscale.nn.checkpoint import checkpoint_wrapper
            validate_checkpoint_wrapper_import(checkpoint_wrapper)

        self.num_classes = num_classes

        # simple convolutions,
        self.patch_embed = PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,  # [3, 7, 7]
            stride=cfg.MVIT.PATCH_STRIDE,  # [2, 4, 4]
            padding=cfg.MVIT.PATCH_PADDING,  #[1, 3, 3]
            conv_2d=use_2d_patch,
        )

        if cfg.MODEL.ACT_CHECKPOINT:
            self.patch_embed = checkpoint_wrapper(self.patch_embed)

        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]

        # 16x224x224 inputs with path_stride (2, 4, 4) -> 8x56x56
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        # 8*56*56
        num_patches = math.prod(self.patch_dims)

        """
        >>> torch.linspace(0, 0.5, 16)
        tensor([0.0000, 0.0333, 0.0667, 0.1000, 0.1333, 0.1667, 0.2000, 0.2333, 0.2667,
        0.3000, 0.3333, 0.3667, 0.4000, 0.4333, 0.4667, 0.5000])
        """
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        # pos_embed_dim is not used for sep_pos_embed == True
        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches


        # for MViTV2, the positional embedding is not used here
        if self.sep_pos_embed:  # true for mvit 16x4 model and 32x3 model
            # this makes MViT model can only be used for same input size
            # patch_dim: 8x56x56
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.patch_dims[0], embed_dim)
            )
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, embed_dim)
                )
        elif self.rel_pos_embed:  # use relative embedding in the attention block instead
            pass
        else:
            # no separate positional embeddings, so
            # with or without cls_embeding, pos_embed_dim = patch_hwt + 1 or patch_hwt
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_dim, embed_dim)
            )

        # dropout rate = 0.5
        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        # depth = 16/24
        # dimention multiplier
        # 16x4 model: [[1, 2.0], [3, 2.0], [14, 2.0]]
        # 32x3 model: [[2, 2.0], [5, 2.0], [21, 2.0]]
        # increase the dimension of the next (i+1) block
        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        # depth=16/24
        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            # q pooling stride:
            # [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
            # so [1, 2, 2] kernel at layer 1, 3, 14
            # no q pooling at other layers
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][
                1:
            ]
            # q pooling kernel (Conv3D)
            if cfg.MVIT.POOL_KVQ_KERNEL is not None: # this is set by default
            # so q_pooling conv3D kernel always 3x3x3
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]

        if self.q_pool_all:
            for i in range(len(pool_q)):
                if not pool_q[i]:
                    pool_q[i] = cfg.MVIT.POOL_KVQ_KERNEL
                    stride_q[i] = [1, 1, 1]  # stride=1 conv3d

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[
                i
            ][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]

        # Default False
        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        self.blocks = nn.ModuleList()

        # head_mul [[1, 2.0], [3, 2.0], [14, 2.0]]
        # dim_mul [[1, 2.0], [3, 2.0], [14, 2.0]]
        #print(dim_mul, head_mul)  # all ones, 2 at the above index

        # need to keep track of the input size at each block
        input_size = self.patch_dims
        # 50% as in the paper
        max_mem_len = [self.max_mem_len if i//2==0 else 0 for i in range(depth)]
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])

            # junwei: MViT version 2, this helps reduce parameters and FLOPs
            if self.channel_expand_front:
                if i == 0:
                    embed_dim_mul = 1.0
                else:
                    embed_dim_mul = dim_mul[i-1]
                embed_dim = round_width(embed_dim, embed_dim_mul, divisor=num_heads)
                dim_out = round_width(dim_out, dim_mul[i], divisor=num_heads)
            else:
                embed_dim = round_width(embed_dim, dim_mul[i], divisor=num_heads)
                # compute the output dimension of each block
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1]),
                )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
                use_query_residual_pool=self.use_query_residual_pool,
                channel_expand_front=self.channel_expand_front,
                pool_skip_use_conv=self.pool_skip_use_conv,
                use_rel_pos=self.rel_pos_embed,
                use_sym_rel=self.use_sym_rel,
                rel_pos_zero_init=self.rel_pos_zero_init,
                input_size=input_size,
                no_einsum=self.no_einsum,
                use_skew=self.use_skew,
                use_mem=self.use_mem,
                max_mem_len=max_mem_len[i],
                mem_compress_factor=self.mem_compress_factor,
            )
            if cfg.MODEL.ACT_CHECKPOINT:
                attention_block = checkpoint_wrapper(attention_block)
            self.blocks.append(attention_block)

            if len(stride_q[i]) > 0:
                # [[]...[1, 2, 2]...[1, 2, 2],... [1, 2, 2]]
                input_size = [
                    size // stride for size, stride in zip(input_size, stride_q[i])
                ]

        # 768
        embed_dim = dim_out

        # MoCo v3 says ViT should not use LN+Pool if CLS token is not used
        # but on K400, without this is 3% worse on Top-1
        self.norm = norm_layer(embed_dim) if not cfg.MVIT.NO_NORM_BEFORE_AVG else None

        if self.sep_pos_embed:
            # from torch.nn.init
            trunc_normal_(self.pos_embed_spatial, std=0.02)
            trunc_normal_(self.pos_embed_temporal, std=0.02)
            if self.cls_embed_on:
                trunc_normal_(self.pos_embed_class, std=0.02)
        elif self.rel_pos_embed:
            pass
        else:
            trunc_normal_(self.pos_embed, std=0.02)

        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)

        self.add_spatial_max_pool_before_proj = cfg.DETECTION.USE_SPATIAL_MAXPOOL_BEFORE_PROJ

        self.use_roi_head = cfg.DETECTION.ENABLE and not cfg.DETECTION.USE_CUBE_PROP
        self.use_multi_head = cfg.MODEL.USE_MULTI_HEAD


        if not self.use_multi_head:
            if self.use_roi_head:
                temporal_pools = [o[1] for o in cfg.MVIT.POOL_Q_STRIDE]
                t_pool_kernel = cfg.DATA.NUM_FRAMES // cfg.MVIT.PATCH_STRIDE[0]
                for t_pool in temporal_pools:
                    t_pool_kernel = t_pool_kernel // t_pool

                self.head = head_helper.ResNetRoIHead(
                    dim_in=[
                        embed_dim,
                    ],
                    num_classes=num_classes,

                    pool_size=[
                        # we may downsample temporal dim in other places
                        #[cfg.DATA.NUM_FRAMES // cfg.MVIT.PATCH_STRIDE[0], 1, 1]
                        [t_pool_kernel, 1, 1]
                    ],
                    resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION]*2],
                    scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],

                    dropout_rate=cfg.MODEL.DROPOUT_RATE,
                    act_func=cfg.MODEL.HEAD_ACT,
                    aligned=cfg.DETECTION.ALIGNED,
                )

                # TODO(junwei): in the MeMViT paper, add a transformer layer
                # instead of linear layer
                # an MViT layer without pooling, before the linear classifier
                # so after ROI-pooling, we have [M, 7, 7, dim],
                # then a self-attention layer, got M, 7, 7, dim again
                # then linear classify
            else:
                self.head = head_helper.TransformerBasicHead(
                    embed_dim,
                    num_classes,
                    dropout_rate=cfg.MODEL.DROPOUT_RATE,
                    act_func=cfg.MODEL.HEAD_ACT,
                    use_act_in_train=cfg.MODEL.USE_HEAD_ACT_IN_TRAIN,
                )
        else:
            assert not self.use_roi_head, "not supported yet"

            self.head = head_helper.TransformerMultiHead(
                embed_dim,
                cfg.MODEL.MULTI_DATASETS,
                cfg.MODEL.MULTI_NUM_CLASSES,
                cfg.MODEL.MULTI_HEAD_ACT,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                use_MLP=cfg.MODEL.MULTI_USE_MLP,
                add_cross_proj=cfg.MODEL.MULTI_ADD_CROSS_PROJ,
                use_moco=cfg.MODEL.MULTI_USE_MOCO,
            )

            if self.cfg.MODEL.USE_INF_LOSS and self.cfg.MODEL.USE_INF_EXPANDER:
                self.expander = Mlp(
                    in_features=embed_dim,
                    hidden_features=4096,
                    out_features=embed_dim,
                    act_layer=nn.GELU,
                    drop_rate=0.0,
                )

        self.apply(self._init_weights)
        if cfg.MODEL.MULTI_USE_MOCO:
            self.init_head_moco()  # copy the weights to moco encoder

    def init_head_moco(self):
        self.head.init_moco()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        # False for MViT 16x4
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            """
            if self.sep_pos_embed:
                if self.cls_embed_on:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                        "cls_token",
                    }
                else:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                    }
            elif self.rel_pos_embed:

            else:
                if self.cls_embed_on:
                    return {"pos_embed", "cls_token"}
                else:
                    return {"pos_embed"}
        else:
            return {}
            """
            names = [
                "pos_embed_spatial",
                "pos_embed_temporal",
                "pos_embed_class",
                "cls_token",
                "rel_pos_h",
                "rel_pos_w",
                "rel_pos_t",
                "pos_embed"
            ]
        return names

    def forward(self, x, bboxes=None, dataset_name=None, run_cross_proj=False,
                use_moco=False, moco_momentum=0.9):
        # for MViT 16x4, 224 model
        #x: torch.Size([1, 3, 16, 224, 224])
        if not self.direct_input:
            # for slowfast inputs
            x = x[0]

        # Conv3D with kernel (3, 7, 7), stride (2, 4, 4), padding (1, 3, 3)
        # conv weights: [96, 3, 3, 7, 7]
        # so 16x224x224 -> 8x56x56
        x = self.patch_embed(x)

        # dimensions after patch convolutions
        T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        H = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        W = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
        B, _, _ = x.shape


        if self.cls_embed_on:  # default config on

            # nn.Parameter(torch.zeros(1, 1, embed_dim))
            # [B, 1, 96]
            # class embedding is the same for all samples
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks

            # class embedding is at the beginning of this tensor
            # random initialized class embedding concat to the input patch tokens
            # [B, 1 + T*H*W, 96]
            # so each sample has the beginning the same across samples
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:  # default true for MVIT B 32x3 and 16x4

            # self.patch_dims [8, 56, 56] for (16, 224, 224) inputs

            # so each token location has a embedding
            # same temporal location has the same position embedding

            # pos_embed_spatial: [1, 56*56, emb_dim]
            # pos_embed_temporal: [1, 8, emb_dim]
            # -> [1, 8*56*56, 96]
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.patch_dims[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.patch_dims[1] * self.patch_dims[2],
                dim=1,
            )

            if self.cls_embed_on:
                # positional embedding for the class [token] embedding
                # pos_embed_class: (1, 1, emb_dim)
                # -> [1, 1+T*H*W, 96]
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)

            # [B, 1 + T*H*W, 96]; broadcast to all sample in the mini-batch
            # all samples, has the same positional embedding vectors at the same location
            # and the class embedding is the same
            x = x + pos_embed

            # x[B, 0, emb_dim] all equal across different batch
        elif self.rel_pos_embed:
            pass  # for relative positional embedding, use them in the attention block
        else:
            x = x + self.pos_embed

        if self.drop_rate:
            # drop_out=0.5 here
            x = self.pos_drop(x)

        if self.norm_stem:  # default false
            # layernorm
            x = self.norm_stem(x)

        # 8, 56, 56 for 16x224x224 inputs
        thw = [T, H, W]
        # [B, 1 + T*H*W, 96] -> # [B, 1 + T*new_hw, new_channel]
        for blk in self.blocks:
            # x.size == T*H*W + 1, +1 is because CLS_EMBED_ON = True
            x, thw = blk(x, thw)
        # thw changes from 56x56 to 7x7, t is still 8 (16/2) for 16x4 model
        # so the spatial downsample is 32x

        # layer norm # true for original MViT
        if self.norm:
            x = self.norm(x)

        if self.feature_extraction:
            if self.cls_embed_on:
                # x is [B, 1+T*new_hw, channel]
                x = x[:, 0]  # THW+1, first dim is cls_emb
            else:
                # [B, T*new_hw, channel]
                x = x.mean(1)
            return x
        # for MViT model
        # POOL_Q_STRIDE: [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
        # torch.Size([1, 392, 768])
        # if POOL_Q_STRIDE: [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 1, 1]]
        # torch.Size([1, 1568, 768])

        # [B, (1+)T*H*W, c=768], 1+ for class embedding
        if not self.use_multi_head:
            if self.use_roi_head:
                T, H, W = thw
                channel = x.shape[-1]
                if self.cls_embed_on:  # 16x4 model has cls_embedding on
                    x = x[:, 1:, :]  # ignore cls embedding for now

                    x = x.reshape(B, T, H, W, channel)
                else:
                    x = x.reshape(B, T, H, W, channel)
                # [B, C, T, H, W]
                x = x.permute(0, 4, 1, 2, 3)
                #sys.exit()
                x = [x]  # slowfast format
                x = self.head(x, bboxes)
            else:
                if self.add_spatial_max_pool_before_proj:
                    # this will produce the same outputs as ROI Align
                    # if using the whole HW as box
                    # junwei: this is just a temporal function
                    if self.cls_embed_on:
                        x = x[:, 1:, :]  # ignore cls embedding for now
                    # [0,0,W,H] boxes
                    T, H, W = thw
                    channel = x.shape[-1]
                    x = x.reshape(B, T, H, W, channel)
                    x = x.mean(1)
                    # [B, C, H, W]
                    x = x.permute(0, 3, 1, 2)
                    # if the H and W != spatial resolution
                    feat_size = self.cfg.DATA.TEST_CROP_SIZE // self.cfg.DETECTION.SPATIAL_SCALE_FACTOR

                    # ONNX will include an if route if used H
                    #if H != self.cfg.DETECTION.ROI_XFORM_RESOLUTION:
                    if feat_size != self.cfg.DETECTION.ROI_XFORM_RESOLUTION:
                        roi_size = self.cfg.DETECTION.ROI_XFORM_RESOLUTION
                        x = torch.nn.functional.interpolate(
                            x,
                            size=(roi_size, roi_size),
                            mode="bilinear", align_corners=True)

                    # this is not supported by ONNX opset 12
                    #x = x.amax((-2, -1), keepdim=False)
                    #b = x.amax((-2, -1))
                    x, _ = x.max(dim=3)
                    x, _ = x.max(dim=2)
                    #assert torch.allclose(b, x)
                else:
                    if self.cls_embed_on:
                        # x is [B, 1+T*new_hw, channel]
                        x = x[:, 0]  # THW+1, first dim is cls_emb
                    else:
                        # [B, T*new_hw, channel]
                        x = x.mean(1)

                # [B, channel] -> [B, classes]  # or in the contrastive learning
                # case, [B, emb_dim]
                x = self.head(x)
        else:
            # for multi dataset forward,
            # only classification supported for now
            if self.cls_embed_on:
                # x is [B, 1+T*new_hw, channel]
                x = x[:, 0]  # THW+1, first dim is cls_emb
            else:
                x = x.mean(1)
            # we have x [B, channel], will output {dataset_name: [B, num_classes]}

            if self.cfg.MODEL.USE_INF_LOSS:
                # x is considered as the video representation
                # apply the robust loss on x


                # add an expander for the embedding, like the VICReg paper
                # they used 3-layer MLP of 8192
                if self.cfg.MODEL.USE_INF_EXPANDER:
                    inf_feature = self.expander(x)
                else:
                    inf_feature = x

                return self.head(
                    x, dataset_name, run_cross_proj=run_cross_proj,
                    use_moco=use_moco, moco_momentum=moco_momentum), inf_feature
            else:
                x = self.head(
                        x, dataset_name, run_cross_proj=run_cross_proj,
                        use_moco=use_moco, moco_momentum=moco_momentum)
        return x


    def reset_mem(self):
        # for long-term model only
        for block in self.blocks:
            block.attn.reset_mem()

class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    """

    def __init__(
        self,
        dim_in=3,
        dim_out=768,
        kernel=(1, 16, 16), # [3, 7, 7]
        stride=(1, 4, 4), # [2, 4, 4]
        padding=(1, 7, 7), # [1, 3, 3]
        conv_2d=False,
    ):
        super().__init__()
        if conv_2d:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d
        self.proj = conv(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.proj(x)
        # B C (T) H W -> B (T)HW C
        return x.flatten(2).transpose(1, 2)
