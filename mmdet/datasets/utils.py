import copy
from collections import Sequence

import mmcv
from mmcv.runner import obj_from_dict
import torch

import matplotlib.pyplot as plt
import numpy as np
from .concat_dataset import ConcatDataset
from .repeat_dataset import RepeatDataset
from .. import datasets


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


def random_scale(img_scales, mode='range'):
    """Randomly select a scale from a list of scales or scale ranges.

    Args:
        img_scales (list[tuple]): Image scale or scale range.
        mode (str): "range" or "value".

    Returns:
        tuple: Sampled image scale.
    """
    num_scales = len(img_scales)
    if num_scales == 1:  # fixed scale is specified
        img_scale = img_scales[0]
    elif num_scales == 2:  # randomly sample a scale
        if mode == 'range':
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            long_edge = np.random.randint(
                min(img_scale_long),
                max(img_scale_long) + 1)
            short_edge = np.random.randint(
                min(img_scale_short),
                max(img_scale_short) + 1)
            img_scale = (long_edge, short_edge)
        elif mode == 'value':
            img_scale = img_scales[np.random.randint(num_scales)]
    else:
        if mode != 'value':
            raise ValueError(
                'Only "value" mode supports more than 2 image scales')
        img_scale = img_scales[np.random.randint(num_scales)]
    return img_scale


def show_ann(coco, img, ann_info):
    plt.imshow(mmcv.bgr2rgb(img))
    plt.axis('off')
    coco.showAnns(ann_info)
    plt.show()

def get_dataset(data_cfg):
    if data_cfg['type'] == 'RepeatDataset':
        return RepeatDataset(
            get_dataset(data_cfg['dataset']), data_cfg['times'])

    # rgb 
    if isinstance(data_cfg['ann_file_r'], (list, tuple)):
        ann_files_r = data_cfg['ann_file_r']
        num_dset_r = len(ann_files_r)
    else:
        ann_files_r = [data_cfg['ann_file_r']]
        num_dset_r = 1
    # infrared
    if isinstance(data_cfg['ann_file_i'], (list, tuple)):
        ann_files_i = data_cfg['ann_file_i']
        num_dset_i = len(ann_files_i)
    else:
        ann_files_i = [data_cfg['ann_file_i']]
        num_dset_i = 1

    if 'proposal_file' in data_cfg.keys():
        if isinstance(data_cfg['proposal_file'], (list, tuple)):
            proposal_files = data_cfg['proposal_file']
        else:
            proposal_files = [data_cfg['proposal_file']]
    else:
        proposal_files = [None] * num_dset_r
    assert len(proposal_files) == num_dset_r

    # rgb 
    if isinstance(data_cfg['img_prefix_r'], (list, tuple)):
        img_prefixes_r = data_cfg['img_prefix_r']
    else:
        img_prefixes_r = [data_cfg['img_prefix_r']] * num_dset_r
    assert len(img_prefixes_r) == num_dset_r
    # infrared
    if isinstance(data_cfg['img_prefix_i'], (list, tuple)):
        img_prefixes_i = data_cfg['img_prefix_i']
    else:
        img_prefixes_i = [data_cfg['img_prefix_i']] * num_dset_i
    assert len(img_prefixes_i) == num_dset_i

    dsets = []
    # compare the data num of the rgb and infrared,choose the min num : min(num_dset_r, num_dset_r)
    for i in range(min(num_dset_r, num_dset_i)):
        data_info = copy.deepcopy(data_cfg)
        data_info['ann_file_r'] = ann_files_r[i]
        data_info['img_prefix_r'] = img_prefixes_r[i]
        data_info['ann_file_i'] = ann_files_i[i]
        data_info['img_prefix_i'] = img_prefixes_i[i]
        data_info['proposal_file'] = proposal_files[i]
        dset = obj_from_dict(data_info, datasets)
        dsets.append(dset)
    if len(dsets) > 1:
        dset = ConcatDataset(dsets)
    else:
        dset = dsets[0]
    return dset
