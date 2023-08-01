#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn
from slowfast.models.common import Mlp
#from detectron2.layers import ROIAlign

from functools import partial

from copy import deepcopy


class TransformerBasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        use_act_in_train=False,
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)
        self.use_act_in_train = use_act_in_train

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # remember to use bce_logits for loss for sigmoid head
        if self.use_act_in_train or not self.training:
            x = self.act(x)
        return x

class ContrastiveProjectionHead(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        use_MLP=False,
        dropout_rate=0.0,
    ):
        """
         Given [B, D], layernorm -> linear
        """
        super(ContrastiveProjectionHead, self).__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(dim_in)
        # WenLan paper and MoCo v2 paper uses 2-layer MLP, 2048-d, RELU
        if use_MLP:
            self.projection = Mlp(
                in_features=dim_in,
                hidden_features=2048,
                out_features=dim_out,
                act_layer=nn.GELU,
                drop_rate=dropout_rate,
            )
        else:
            self.projection = nn.Linear(dim_in, dim_out, bias=False)



    def forward(self, x):
        x = self.norm(x)
        x = self.projection(x)
        return x


def get_act_func(func_name):
    if func_name == "softmax":
        return nn.Softmax(dim=1)
    elif func_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(func_name)
        )

class TransformerMultiHead(nn.Module):
    """
    multiple classification head
    """

    def __init__(
        self,
        dim_in,
        dataset_names,
        dataset_num_classes,
        act_funcs,
        dropout_rate=0.0,
        use_MLP=False,
        add_cross_proj=False,  # add pair-wise dataset class projection layers
        use_moco=False, # use moco encoder for the cross dataset proj part
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerMultiHead, self).__init__()
        if dropout_rate > 0.0 and not use_MLP:
            # we will use dropout within MLP if used
            self.dropout = nn.Dropout(dropout_rate)

        # for cross_entropy loss, we dont use act during training , but bce need it
        self.heads = {}
        self.acts = {}

        self.cross_dataset_heads = {}  # dataset-to-dataset projection
        self.add_cross_proj = add_cross_proj
        self.use_moco = use_moco

        assert len(dataset_names) == len(dataset_num_classes) == len(act_funcs)
        for i, dataset_name in enumerate(dataset_names):
            num_classes = dataset_num_classes[i]
            if use_MLP:
                self.heads[dataset_name] = Mlp(
                    in_features=dim_in,
                    hidden_features=2048,
                    out_features=num_classes,
                    act_layer=nn.GELU,
                    drop_rate=dropout_rate,
                )
            else:
                self.heads[dataset_name] = nn.Linear(dim_in, num_classes, bias=True)
            self.acts[dataset_name] = get_act_func(act_funcs[i])

            if self.add_cross_proj:
                for j, other_dataset_name in enumerate(dataset_names):
                    if other_dataset_name == dataset_name:
                        continue
                    proj_name = "%s_%s" % (other_dataset_name, dataset_name)
                    other_dataset_num_classes = dataset_num_classes[j]

                    # mit_k700
                    # 305 x 700
                    self.cross_dataset_heads[proj_name] = nn.Linear(
                        other_dataset_num_classes, num_classes, bias=False)


        self.heads = nn.ModuleDict([[k, self.heads[k]] for k in self.heads])
        self.acts = nn.ModuleDict([[k, self.acts[k]] for k in self.acts])

        if self.add_cross_proj:
            self.cross_dataset_heads = nn.ModuleDict(
                [[k, self.cross_dataset_heads[k]]
                 for k in self.cross_dataset_heads])

            if self.use_moco:
                self.heads_moco = deepcopy(self.heads)

    def init_moco(self):
        for param_b, param_m in zip(self.heads.parameters(), \
                self.heads_moco.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

    @torch.no_grad()
    def _moco_update(self, momentum):
        for param_b, param_m in zip(self.heads.parameters(), \
                self.heads_moco.parameters()):
            param_m.data = param_m.data * momentum + param_b.data * (1 - momentum)

    def forward(self, x, dataset_name=None,
                run_cross_proj=False, use_moco=False, moco_momentum=0.9):
        """
        output {dataset_name: outputs}
        """
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        head_outputs = {}

        if use_moco:
            self._moco_update(moco_momentum)
            head_outputs_moco = {}

        if dataset_name is None:
            run_names = self.heads.keys()
        else:
            run_names = [dataset_name]
            assert dataset_name in self.heads.keys()

        for dataset_name in run_names:
            x_head = self.heads[dataset_name](x)

            # no activation func during training,
            # so for sigmoid head remember to use bce_logit for loss
            if not self.training:
                x_head = self.acts[dataset_name](x_head)

            head_outputs[dataset_name] = x_head
            if use_moco:
                head_outputs_moco[dataset_name] = self.heads_moco[dataset_name](x)

        # should only be used during training
        if self.add_cross_proj and run_cross_proj:
            assert self.training, "cross dataset projection is not supposed to be used during inf"


            for d1_d2 in self.cross_dataset_heads.keys():
                d1_name, d2_name = d1_d2.split("_")
                # so the output should be the same dim as d2_name
                if use_moco:
                    proj_inputs = head_outputs_moco[d1_name]
                else:
                    proj_inputs = head_outputs[d1_name]
                # junweil: should we add softmax here?
                #proj_inputs = self.acts[d1_name](proj_inputs)
                head_outputs[d1_d2] = self.cross_dataset_heads[d1_d2](proj_inputs)

        return head_outputs
