#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiHeadLoss(nn.Module):
    """
    Multi-head cross dataset training loss
    """

    def __init__(self, cfg):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(MultiHeadLoss, self).__init__()
        self.cfg = cfg

        self.dataset_loss = {}
        self.datasets = cfg.MODEL.MULTI_DATASETS
        assert len(cfg.MODEL.MULTI_DATASETS) == len(cfg.MODEL.MULTI_LOSS_FUNCS)
        for dataset_name, loss_name in zip(cfg.MODEL.MULTI_DATASETS, cfg.MODEL.MULTI_LOSS_FUNCS):
            if loss_name not in _SOFT_TARGET_LOSSES.keys():
                raise NotImplementedError("Loss {} is not supported for multi-dataset".format(loss_name))

            this_loss_func = _LOSSES[loss_name](reduction="none")

            self.dataset_loss[dataset_name] = this_loss_func

        self.add_cross_proj = cfg.MODEL.MULTI_ADD_CROSS_PROJ
        self.cross_proj_add_to_pred = cfg.MODEL.MULTI_CROSS_PROJ_ADD_TO_PRED

        self.proj_loss_func = cfg.MODEL.MULTI_PROJ_LOSS_FUNC
        if self.proj_loss_func is not None:
            self.proj_loss_func = _LOSSES[self.proj_loss_func](reduction="none")

        self.proj_loss_weight = cfg.MODEL.MULTI_PROJ_LOSS_WEIGHT

        self.dataset_loss_weight_dict = None
        self.use_mtl_weight = cfg.MODEL.MULTI_USE_MTL_WEIGHT
        if self.use_mtl_weight:
            # paper says log_{sigma^2} can be from -2 to 5.0 initialized value and get
            # same result
            # here we train sigma square
            self.dataset_sigma_sqs = nn.Parameter(torch.zeros((len(self.datasets))))
            nn.init.uniform_(self.dataset_sigma_sqs, 0.2, 1.0)
        else:
            # pre-defined loss weights for each dataset-head
            self.dataset_loss_weight_dict = {
                cfg.MODEL.MULTI_DATASETS[i]: cfg.MODEL.MULTI_LOSS_WEIGHTS[i]
                for i in range(len(cfg.MODEL.MULTI_DATASETS))}


    def forward(self, preds, labels, masks):

        losses = []

        batch_size = None
        for dataset_id, dataset_name in enumerate(self.datasets):

            loss_func = self.dataset_loss[dataset_name]

            pred = preds[dataset_name]

            if self.add_cross_proj and self.cross_proj_add_to_pred:
                for d1_d2 in preds.keys():
                    if d1_d2 in self.datasets:
                        continue
                    d1_name, d2_name = d1_d2.split("_")
                    if d2_name == dataset_name:
                        proj_pred = preds[d1_d2]  # [B, num_class]
                        pred = pred + proj_pred * self.proj_loss_weight
                        # don't use += , in-place change, pytorch will complain


            loss = loss_func(pred, labels[dataset_name])

            # TODO: change this to a gather function, watch out for mask all zero case
            # https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591

            # loss could be [B, num_class] for bce loss
            if len(loss.shape) == 2:
                loss = loss.mean(-1)  # along the class dimension

            batch_size = loss.shape[0]

            # add additional projection loss from other head
            if self.add_cross_proj and not self.cross_proj_add_to_pred:
                proj_losses = []
                # get all the projected outputs
                for d1_d2 in preds.keys():
                    if d1_d2 in self.datasets:
                        continue
                    d1_name, d2_name = d1_d2.split("_")
                    if d2_name == dataset_name:
                        proj_pred = preds[d1_d2]  # [B, num_class]
                        # [B]
                        proj_loss = self.proj_loss_func(proj_pred, labels[dataset_name])
                        proj_losses.append(proj_loss * self.proj_loss_weight)
                # [B]
                proj_loss = torch.stack(proj_losses, dim=1).mean(dim=1)
                loss = loss + proj_loss


            # mask out samples in this batch that is not this dataset's
            # loss is [B], mask is also [B]
            loss_masked = masks[dataset_name] * loss  # batch_size

            # one loss value for each dataset
            loss_summed = loss_masked.sum()

            if self.use_mtl_weight:
                # https://arxiv.org/pdf/1705.07115.pdf
                loss_summed = loss_summed / (2 * self.dataset_sigma_sqs[dataset_id]) + \
                    0.5 * torch.log(self.dataset_sigma_sqs[dataset_id])
            elif self.dataset_loss_weight_dict is not None:
                # bce loss is 0.6 and ce loss could be 5.0, so some balancing is needed
                loss_summed = self.dataset_loss_weight_dict[dataset_name] * loss_summed

            losses.append(loss_summed)  # num_dataset


        loss_per_minibatch = torch.stack(losses, dim=0).sum() / batch_size
        #print(loss_per_minibatch)
        #sys.exit()
        return loss_per_minibatch



class INFLoss(nn.Module):
    """
    informative loss and with uncertainty weighting
    """

    def __init__(self, cfg):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(INFLoss, self).__init__()
        self.cfg = cfg
        self.use_mtl_weight = cfg.MODEL.USE_INF_LEARN_WEIGHT
        self.std_weight = cfg.MODEL.INF_STD_WEIGHT
        self.covar_weight = cfg.MODEL.INF_COVAR_WEIGHT

        if self.use_mtl_weight:
            # paper says log_{sigma^2} can be from -2 to 5.0 initialized value and get
            # same result
            # here we train sigma square
            self.dataset_sigma_sqs = nn.Parameter(torch.zeros(2))
            nn.init.uniform_(self.dataset_sigma_sqs, 0.2, 1.0)



    def forward(self, emb):

        """
        enforce the representation/embedding to be more informative
        """
        # need to all_gather all emb of the whole batch to compute
        # [B, C]
        # TODO(junwei): check emb mean() and then var()?
        emb = emb - emb.mean(dim=0)
        std_emb = torch.sqrt(emb.var(dim=0) + 1e-4)
        # [B] -> scalar
        std_loss = torch.mean(F.relu(1 - std_emb))  # hinge loss

        batch_size = emb.shape[0]
        feature_size = emb.shape[1]
        cov_emb = (emb.T @ emb) / (batch_size - 1)  # 1/(n-1)

        # [B, B] -> scalar
        cov_loss = off_diagonal(cov_emb).pow_(2).sum().div(feature_size)

        std_weight = self.std_weight
        covar_weight = self.covar_weight
        if self.use_mtl_weight:
            std_weight = self.dataset_sigma_sqs[0]
            covar_weight = self.dataset_sigma_sqs[1]
            loss = std_loss / (2 * std_weight) + 0.5 * torch.log(std_weight) + \
                cov_loss / (2 * covar_weight) + 0.5 * torch.log(covar_weight)
            return loss
        else:
            return std_loss * self.std_weight + cov_loss * self.covar_weight


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class DualSoftmax(nn.Module):
    """
        https://github.com/starmemda/CAMoE
        softmax on the column, get the "prior prob map", meaning some
        logits might could be compared from column rather than from row
        # The temperature hyperparameter is used for smoothing the gradient
        # in training and should be set to 1 during inference.
    """
    def __init__(self,):
        super(DualSoftmax, self).__init__()

    def forward(self, sim_matrix, temp=1000):
        # sim_matrix : N, N
        # With an appropriate temperature parameter, the model achieves higher performance
        sim_matrix = sim_matrix * F.softmax(sim_matrix / temp, dim=0) * len(sim_matrix)
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        loss = -logpt
        return loss

class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target. For MixUp train
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        # x [B, num_classes]
        # y [B, C], 0 - 1.0
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError

class NormalizedSoftTargetCrossEntropy(nn.Module):
    """
    Normalized Cross entropy loss with soft target. For MixUp train
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(NormalizedSoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        # x [B, num_classes], logits
        # y [B, C], 0 - 1.0
        # softmax on logits and then log
        pred = F.log_softmax(x, dim=-1) # [B, num_classes]
        loss = - torch.sum(y * pred, dim=-1) / (- pred.sum(dim=-1))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError


class ReverseSoftTargetCrossEntropy(nn.Module):
    """
    Reverse Cross entropy loss with soft target. For MixUp train
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(ReverseSoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        # x [B, num_classes], logits
        # y [B, C], 0 - 1.0
        # softmax on logits and then log
        pred = F.softmax(x, dim=-1) # [B, num_classes]
        pred = torch.clamp(pred, min=1e-7, max=1.0)

        # so low confidence class will be ignore
        y = torch.clamp(y, min=1e-4, max=1.0)  # cannot be zeros
        y = torch.log(y)
        loss = - torch.sum(y * pred, dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError


class MeanAbsoluteError(nn.Module):
    """
    mean absolute error loss, it is said to be more robust to label noise compared
    to CE loss
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(MeanAbsoluteError, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        # x [B, num_classes], logits
        # y [B, C], 0 - 1.0
        # softmax on logits and then log
        pred = F.softmax(x, dim=-1) # [B, num_classes]
        loss = 1. - torch.sum(y*pred, dim=-1)
        # Note: Reduced MAE
        # Original: torch.abs(pred - label_one_hot).sum(dim=1)
        # $MAE = \sum_{k=1}^{K} |\bm{p}(k|\bm{x}) - \bm{q}(k|\bm{x})|$
        # $MAE = \sum_{k=1}^{K}\bm{p}(k|\bm{x}) - p(y|\bm{x}) + (1 - p(y|\bm{x}))$
        # $MAE = 2 - 2p(y|\bm{x})$
        #
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError

# combined loss with noramlized ce
# http://proceedings.mlr.press/v119/ma20c/ma20c.pdf
class NCEandRCE(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, reduction="mean"):
        super(NCEandRCE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nce = NormalizedSoftTargetCrossEntropy(reduction=reduction)
        self.rce = ReverseSoftTargetCrossEntropy(reduction=reduction)

    def forward(self, pred, labels):
        return self.alpha * self.nce(pred, labels) + self.beta * self.rce(pred, labels)


class LSEPLoss(nn.Module):
    """
    # http://openaccess.thecvf.com/content_cvpr_2017/html/Li_Improving_Pairwise_Ranking_CVPR_2017_paper.html
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(LSEPLoss, self).__init__()
        self.reduction = reduction

    def forward(self, scores, labels):
        # scores [B, num_classes]
        # labels [B, C], 0 -1 1.0

        mask = ((labels.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)) -
                     labels.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1))) > 0).float()
        loss = (scores.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1)) -
                     scores.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)))
        loss = loss.exp().mul(mask).sum().add(1).log()

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError

# AVA models use bce
# Kinetics models use MViT - soft_cross_entropy, slowfast-> cross_entropy
_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,  # this can be used with soft targets
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "mean_absolute_error": MeanAbsoluteError,
    "reverse_soft_cross_entropy": ReverseSoftTargetCrossEntropy,
    "normalized_soft_cross_entropy": NormalizedSoftTargetCrossEntropy,
    "nce_and_rce": NCEandRCE,
    "lsep": LSEPLoss,
}

# these losses takes labels of [B, C] as inputs
_SOFT_TARGET_LOSSES = {
    "bce": nn.BCELoss,  # this can be used with soft targets
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "nce_and_rce": NCEandRCE,
    "normalized_soft_cross_entropy": NormalizedSoftTargetCrossEntropy,
    "reverse_soft_cross_entropy": ReverseSoftTargetCrossEntropy,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]






