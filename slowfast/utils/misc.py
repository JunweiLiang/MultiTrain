#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import logging
import math
import numpy as np
import os
import cv2
from datetime import datetime
import psutil
import torch
from fvcore.nn.activation_count import activation_count
from fvcore.nn.flop_count import flop_count
from matplotlib import pyplot as plt
from torch import nn

import slowfast.utils.logging as logging
import slowfast.utils.multiprocessing as mpu
from slowfast.datasets.utils import pack_pathway_output
from slowfast.datasets.utils import pack_mem_clips
#from slowfast.models.batchnorm_helper import SubBatchNorm3d
from slowfast.utils.env import pathmgr

logger = logging.get_logger(__name__)


# helper function on video frame processing with numpy array by junwei
def pixel_norm(frames, mean, std, channel_first=True):
    """ Given np frames of [C, T, H, W] or [T, H, W, C], do pixel norm
        mean and std are shape=[3] array
    """
    if channel_first:
        C, T, H, W = frames.shape
        tiled_mean = np.tile(np.expand_dims(mean, axis=[1, 2, 3]), [1, T, H, W])
        tiled_std = np.tile(np.expand_dims(std, axis=[1, 2, 3]), [1, T, H, W])
    else:
        T, H, W, C = frames.shape
        tiled_mean = np.tile(np.expand_dims(mean, axis=[0, 1, 2]), [T, H, W, 1])
        tiled_std = np.tile(np.expand_dims(std, axis=[0, 1, 2]), [T, H, W, 1])

    return (frames - tiled_mean) / tiled_std

def crop_and_resize(np_frames, size_scale, crop_size, crop_tlbr=None, boxes=None,
                    keep_scale=True,
                    spatial_sample_index=1):
    """ Given the numpy frame, crop the tlbr out, and resize to size_scale and keeping ratio
        tlbr means the box to crop from np_frames before resizing, boxes are the
        ones within the image
        Args:
            size_scale: short_edge size
            crop_size: crop size
            crop_tlbr: used to crop np_frames at first
            boxes: the box associated with np_frames, need to change as frames change
            spatial_sample_index, 0: left (top), 1: center, 2: right (bottom) crop
    """
    if crop_tlbr is not None:
        # avoid negative numbers, possible bugs
        left, top, right, bottom = [max(int(o), 0) for o in crop_tlbr]
        # [T, H, W, C]
        # here the out-of-frame errors will be ignored and crop the largest possible
        # but we need to check for zero width and height

        cropped_frames = np_frames[:, top:bottom+1, left:right+1,  :]
        height, width = cropped_frames.shape[1:3]
        if height == 0 or width == 0:
            raise Exception("got zero size crop: %s, crop_tlbr: %s" % (
                cropped_frames.shape, crop_tlbr))
        if boxes is not None:
            # make the box local
            new_boxes = []
            for l, t, r, b in boxes:
                new_boxes.append([l - left, t - top, r - left, b - top])
            boxes = np.array(new_boxes)
    else:
        cropped_frames = np_frames

    #print(cropped_frames.shape)
    # (32, 631, 269, 3) -> 32, (600, 256, 3)  # (resizing)
    # (32, 304, 134, 3) -> 32, (580, 256, 3)

    cropped_resized_frames, _ = short_edge_resize(
        cropped_frames,
        size=size_scale,
        boxes=boxes,
        keep_scale=keep_scale)

    #print([c.shape for c in cropped_resized_frames], len(cropped_resized_frames))
    # 32, (600, 256, 3) -> 32, (256, 256, 3) # (center cropping)
    cropped_resized_frames, _ = spatial_shift_crop_list(
          crop_size, cropped_resized_frames, spatial_sample_index, boxes=boxes)
    #print([c.shape for c in cropped_resized_frames], len(cropped_resized_frames))
    #sys.exit()
    return cropped_resized_frames, boxes

def short_edge_resize(images, size, boxes=None, keep_scale=True):
    """
    Perform a spatial short scale jittering on the given images and
    corresponding boxes.
    Args:
            images (list): list of images to perform scale jitter. Dimension is
                    `height` x `width` x `channel`.
            size (int): short edge will be resized to this
            boxes (numpy array):
            keep_scale: if False, width and height will be resize to size, otherwise only short edge
    Returns:
            (list): the list of scaled images with dimension of
                    `new height` x `new width` x `channel`.
            (ndarray or None): the scaled boxes with dimension of
                    `num boxes` x 4.
    """
    # size=256
    # (32, 631, 269, 3) -> 32, (600, 256, 3),
    height = images[0].shape[0]
    width = images[0].shape[1]
    if (width <= height and width == size) or \
            (height <= width and height == size):
        return images, boxes
    new_width = size
    new_height = size

    if keep_scale:
        if width < height:
            new_height = int(math.floor((float(height) / width) * size))
        else:
            new_width = int(math.floor((float(width) / height) * size))

    if boxes is not None:
        boxes[:, [1, 3]] *= (float(new_height) / height)
        boxes[:, [0, 2]] *= (float(new_width) / width)
    return [
            cv2.resize(
                    image, (new_width, new_height),
                    interpolation=cv2.INTER_LINEAR).astype(np.float32)
            for image in images], boxes

def spatial_shift_crop_list(size, images, spatial_shift_pos, boxes=None):
    """
    Perform left, center, or right crop of the given list of images.
    Args:
            size (int): size to crop.
            image (list): ilist of images to perform short side scale. Dimension is
                    `height` x `width` x `channel` or `channel` x `height` x `width`.
            spatial_shift_pos (int): option includes 0 (left), 1 (middle), and
                    2 (right) crop.
            boxes (list): optional. Corresponding boxes to images.
                    Dimension is `num boxes` x 4.
    Returns:
            cropped (ndarray): the cropped list of images with dimension of
                    `height` x `width` x `channel`.
            boxes (list): optional. Corresponding boxes to images. Dimension is
                    `num boxes` x 4.
    """
    # size=256
    # 32, (600, 256, 3) -> 32, (256, 256, 3)
    assert spatial_shift_pos in [0, 1, 2]

    height = images[0].shape[0]
    width = images[0].shape[1]
    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_shift_pos == 0:
            y_offset = 0
        elif spatial_shift_pos == 2:
            y_offset = height - size
    else:
        if spatial_shift_pos == 0:
            x_offset = 0
        elif spatial_shift_pos == 2:
            x_offset = width - size

    cropped = [
            image[y_offset : y_offset + size, x_offset : x_offset + size, :]
            for image in images
    ]
    assert cropped[0].shape[0] == size, "Image height not cropped properly"
    assert cropped[0].shape[1] == size, "Image width not cropped properly"

    if boxes is not None:
        boxes[:, [0, 2]] -= x_offset
        boxes[:, [1, 3]] -= y_offset
    return cropped, boxes



def check_nan_losses(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    return math.isnan(loss)
    #raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))


def params_count(model, ignore_bn=False):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    if not ignore_bn:
        return np.sum([p.numel() for p in model.parameters()]).item()
    else:
        count = 0
        for m in model.modules():
            if not isinstance(m, nn.BatchNorm3d):
                for p in m.parameters(recurse=False):
                    count += p.numel()
    return count


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024 ** 3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total


def _get_model_analysis_input(cfg, use_train_input):
    """
    Return a dummy input for model analysis with batch size 1. The input is
        used for analyzing the model (counting flops and activations etc.).
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        use_train_input (bool): if True, return the input for training. Otherwise,
            return the input for testing.

    Returns:
        inputs: the input for model analysis.
    """
    rgb_dimension = 3
    if use_train_input:
        if cfg.TRAIN.DATASET in ["imagenet", "imagenetprefetch"]:
            input_tensors = torch.rand(
                rgb_dimension,
                cfg.DATA.TRAIN_CROP_SIZE,
                cfg.DATA.TRAIN_CROP_SIZE,
            )
        elif cfg.TEST.DATASET in ["kinetics_longterm"]:
            input_tensors = torch.rand(
                rgb_dimension,
                cfg.DATA.NUM_FRAMES * (cfg.MVIT.MAX_MEM_LEN + 1),
                cfg.DATA.TRAIN_CROP_SIZE,
                cfg.DATA.TRAIN_CROP_SIZE,
            )
        else:
            input_tensors = torch.rand(
                rgb_dimension,
                cfg.DATA.NUM_FRAMES,
                cfg.DATA.TRAIN_CROP_SIZE,
                cfg.DATA.TRAIN_CROP_SIZE,
            )
    else:
        if cfg.TEST.DATASET in ["imagenet", "imagenetprefetch"]:
            input_tensors = torch.rand(
                rgb_dimension,
                cfg.DATA.TEST_CROP_SIZE,
                cfg.DATA.TEST_CROP_SIZE,
            )
        elif cfg.TEST.DATASET in ["kinetics_longterm"]:
            input_tensors = torch.rand(
                rgb_dimension,
                cfg.DATA.NUM_FRAMES * (cfg.MVIT.MAX_MEM_LEN + 1),
                cfg.DATA.TEST_CROP_SIZE,
                cfg.DATA.TEST_CROP_SIZE,
            )
        else:
            input_tensors = torch.rand(
                rgb_dimension,
                cfg.DATA.NUM_FRAMES,
                cfg.DATA.TEST_CROP_SIZE,
                cfg.DATA.TEST_CROP_SIZE,
            )
    if cfg.TEST.DATASET in ["kinetics_longterm"]:
        model_inputs = pack_mem_clips(input_tensors, cfg.MVIT.MAX_MEM_LEN + 1)
    else:
        model_inputs = pack_pathway_output(cfg, input_tensors)
    for i in range(len(model_inputs)):
        model_inputs[i] = model_inputs[i].unsqueeze(0)
        if cfg.NUM_GPUS:
            model_inputs[i] = model_inputs[i].cuda(non_blocking=True)

    # If detection is enabled, count flops for one proposal.
    if cfg.DETECTION.ENABLE:
        bbox = torch.tensor([[0, 0, 1.0, 0, 1.0]])
        if cfg.NUM_GPUS:
            bbox = bbox.cuda()
        inputs = (model_inputs, bbox)
    else:
        inputs = (model_inputs,)
    return inputs


def get_model_stats(model, cfg, mode, use_train_input):
    """
    Compute statistics for the current model given the config.
    Args:
        model (model): model to perform analysis.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        mode (str): Options include `flop` or `activation`. Compute either flop
            (gflops) or activation count (mega).
        use_train_input (bool): if True, compute statistics for training. Otherwise,
            compute statistics for testing.

    Returns:
        float: the total number of count of the given model.
    """
    assert mode in [
        "flop",
        "activation",
    ], "'{}' not supported for model analysis".format(mode)
    if mode == "flop":
        model_stats_fun = flop_count
    elif mode == "activation":
        model_stats_fun = activation_count

    # Set model to evaluation mode for analysis.
    # Evaluation mode can avoid getting stuck with sync batchnorm.
    model_mode = model.training
    model.eval()
    # a random tensor
    inputs = _get_model_analysis_input(cfg, use_train_input)
    #print(inputs[0][0].shape)  #torch.Size([1, 3, 16, 224, 224])
    # for multi-dataset/head model, this fails
    assert not cfg.MVIT.USE_MEM, "FLOP count for long-term model not available yet"
    count_dict, *_ = model_stats_fun(model, inputs)
    #print(count_dict)
    #sys.exit()
    count = sum(count_dict.values())
    model.train(model_mode)
    return count


def log_model_info(model, cfg, use_train_input=True):
    """
    Log info, includes number of parameters, gpu usage, gflops and activation count.
        The model info is computed when the model is in validation mode.
    Args:
        model (model): model to log the info.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        use_train_input (bool): if True, log info for training. Otherwise,
            log info for testing.
    """
    params = params_count(model)
    gpu_mem_use = gpu_mem_usage()
    flops = get_model_stats(model, cfg, "flop", use_train_input)
    # the following may cause OOM
    #activation = get_model_stats(model, cfg, "activation", use_train_input)
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(params))
    logger.info("Mem: {:,} MB".format(gpu_mem_use))
    logger.info(
        "Flops: {:,} G".format(
            flops
        )
    )
    #logger.info(
    #    "Activations: {:,} M".format(
    #        activation
    #   )
    #)
    #logger.info("nvidia-smi")
    #os.system("nvidia-smi")


def is_eval_epoch(cfg, cur_epoch, multigrid_schedule):
    """
    Determine if the model should be evaluated at the current epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (int): current epoch.
        multigrid_schedule (List): schedule for multigrid training.
    """
    if cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH:
        return True
    if multigrid_schedule is not None:
        prev_epoch = 0
        for s in multigrid_schedule:
            if cur_epoch < s[-1]:
                period = max(
                    (s[-1] - prev_epoch) // cfg.MULTIGRID.EVAL_FREQ + 1, 1
                )
                return (s[-1] - 1 - cur_epoch) % period == 0
            prev_epoch = s[-1]

    return (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0


def plot_input(tensor, bboxes=(), texts=(), path="./tmp_vis.png"):
    """
    Plot the input tensor with the optional bounding box and save it to disk.
    Args:
        tensor (tensor): a tensor with shape of `NxCxHxW`.
        bboxes (tuple): bounding boxes with format of [[x, y, h, w]].
        texts (tuple): a tuple of string to plot.
        path (str): path to the image to save to.
    """
    tensor = tensor.float()
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    f, ax = plt.subplots(nrows=1, ncols=tensor.shape[0], figsize=(50, 20))
    for i in range(tensor.shape[0]):
        ax[i].axis("off")
        ax[i].imshow(tensor[i].permute(1, 2, 0))
        # ax[1][0].axis('off')
        if bboxes is not None and len(bboxes) > i:
            for box in bboxes[i]:
                x1, y1, x2, y2 = box
                ax[i].vlines(x1, y1, y2, colors="g", linestyles="solid")
                ax[i].vlines(x2, y1, y2, colors="g", linestyles="solid")
                ax[i].hlines(y1, x1, x2, colors="g", linestyles="solid")
                ax[i].hlines(y2, x1, x2, colors="g", linestyles="solid")

        if texts is not None and len(texts) > i:
            ax[i].text(0, 0, texts[i])
    f.savefig(path)


def frozen_bn_stats(model):
    """
    Set all the bn layers to eval mode.
    Args:
        model (model): model to set bn layers to eval mode.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()




def launch_job(cfg, init_method, func, daemon=False):
    """
    Run 'func' on one or more GPUs, specified in cfg
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        init_method (str): initialization method to launch the job with multiple
            devices.
        func (function): job to run on GPU(s)
        daemon (bool): The spawned processes’ daemon flag. If set to True,
            daemonic processes will be created
    """
    if cfg.NUM_GPUS > 1:
        # spawn needed process within this machine,
        # multi-machine support is through init_process_group in mpu.run()
        torch.multiprocessing.spawn(
            mpu.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                func,  # train_net -> train()
                init_method,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                cfg.DIST_BACKEND,
                cfg,
            ),
            daemon=daemon,
        )
    else:
        func(cfg=cfg)


def get_class_names(path, parent_path=None, subset_path=None):
    """
    Read json file with entries {classname: index} and return
    an array of class names in order.
    If parent_path is provided, load and map all children to their ids.
    Args:
        path (str): path to class ids json file.
            File must be in the format {"class1": id1, "class2": id2, ...}
        parent_path (Optional[str]): path to parent-child json file.
            File must be in the format {"parent1": ["child1", "child2", ...], ...}
        subset_path (Optional[str]): path to text file containing a subset
            of class names, separated by newline characters.
    Returns:
        class_names (list of strs): list of class names.
        class_parents (dict): a dictionary where key is the name of the parent class
            and value is a list of ids of the children classes.
        subset_ids (list of ints): list of ids of the classes provided in the
            subset file.
    """
    try:
        with pathmgr.open(path, "r") as f:
            class2idx = json.load(f)
    except Exception as err:
        print("Fail to load file from {} with error {}".format(path, err))
        return

    max_key = max(class2idx.values())
    class_names = [None] * (max_key + 1)

    for k, i in class2idx.items():
        class_names[i] = k

    class_parent = None
    if parent_path is not None and parent_path != "":
        try:
            with pathmgr.open(parent_path, "r") as f:
                d_parent = json.load(f)
        except EnvironmentError as err:
            print(
                "Fail to load file from {} with error {}".format(
                    parent_path, err
                )
            )
            return
        class_parent = {}
        for parent, children in d_parent.items():
            indices = [
                class2idx[c] for c in children if class2idx.get(c) is not None
            ]
            class_parent[parent] = indices

    subset_ids = None
    if subset_path is not None and subset_path != "":
        try:
            with pathmgr.open(subset_path, "r") as f:
                subset = f.read().split("\n")
                subset_ids = [
                    class2idx[name]
                    for name in subset
                    if class2idx.get(name) is not None
                ]
        except EnvironmentError as err:
            print(
                "Fail to load file from {} with error {}".format(
                    subset_path, err
                )
            )
            return

    return class_names, class_parent, subset_ids
