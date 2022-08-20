from __future__ import division

import argparse
import os.path as osp
import shutil
import tempfile

import mmcv
from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval, \
    HBBSeg2Comp4, OBBDet2Comp4, OBBDetComp4, \
    HBBOBB2Comp4, HBBDet2Comp4, OBBDetComp4_rgb, OBBDetComp4_infrared     # rgb & infrared

import argparse

from mmdet import __version__
from mmdet.datasets import get_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector
import torch
import json
from mmcv import Config
import sys

from DOTA_devkit.ResultMerge_multi_process import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default='configs/DOTA/faster_rcnn_r101_fpn_1x_dota2_v3_RoITrans_v5.py')
    parser.add_argument('--type', default=r'HBB',
                        help='parse type of detector')
    args = parser.parse_args()

    return args

def OBB2HBB(srcpath, dstpath):
    filenames = util.GetFileFromThisRootDir(srcpath)
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)
    for file in filenames:
        with open(file, 'r') as f_in:
            with open(os.path.join(dstpath, os.path.basename(os.path.splitext(file)[0]) + '.txt'), 'w') as f_out:
                lines = f_in.readlines()
                splitlines = [x.strip().split() for x in lines]
                for index, splitline in enumerate(splitlines):
                    imgname = splitline[0]
                    score = splitline[1]
                    poly = splitline[2:]
                    poly = list(map(float, poly))
                    xmin, xmax, ymin, ymax = min(poly[0::2]), max(poly[0::2]), min(poly[1::2]), max(poly[1::2])
                    rec_poly = [xmin, ymin, xmax, ymax]
                    outline = imgname + ' ' + score + ' ' + ' '.join(map(str, rec_poly))
                    if index != (len(splitlines) - 1):
                        outline = outline + '\n'
                    f_out.write(outline)

def parse_results(config_file, resultfile, dstpath, type):
    cfg = Config.fromfile(config_file)

    data_test = cfg.data['test']
    dataset = get_dataset(data_test)
    outputs = mmcv.load(resultfile)
    if type == 'OBB':
        #  dota1 has tested
        obb_results_dict = OBBDetComp4(dataset, outputs)
        current_thresh = 0.1
    elif type == 'HBB':
        # dota1 has tested
        hbb_results_dict = HBBDet2Comp4(dataset, outputs)
    elif type == 'HBBOBB':
        # dota1 has tested
        # dota2
        hbb_results_dict, obb_results_dict = HBBOBB2Comp4(dataset, outputs)
        current_thresh = 0.3
    elif type == 'Mask':
        # TODO: dota1 did not pass
        # dota2, hbb has passed, obb has passed
        hbb_results_dict, obb_results_dict = HBBSeg2Comp4(dataset, outputs)
        current_thresh = 0.3

    dataset_type = cfg.dataset_type

    if 'obb_results_dict' in vars():
        if not os.path.exists(os.path.join(dstpath, 'Task1_results')):
            os.makedirs(os.path.join(dstpath, 'Task1_results'))

        for cls in obb_results_dict:
            with open(os.path.join(dstpath, 'Task1_results', cls + '.txt'), 'w') as obb_f_out:
                for index, outline in enumerate(obb_results_dict[cls]):
                    if index != (len(obb_results_dict[cls]) - 1):
                        obb_f_out.write(outline + '\n')
                    else:
                        obb_f_out.write(outline)

    if 'hbb_results_dict' in vars():
        if not os.path.exists(os.path.join(dstpath, 'Task2_results')):
            os.makedirs(os.path.join(dstpath, 'Task2_results'))
        for cls in hbb_results_dict:
            with open(os.path.join(dstpath, 'Task2_results', cls + '.txt'), 'w') as f_out:
                for index, outline in enumerate(hbb_results_dict[cls]):
                    if index != (len(hbb_results_dict[cls]) - 1):
                        f_out.write(outline + '\n')
                    else:
                        f_out.write(outline)

def parse_results_TS(config_file, resultfile_r, resultfile_i, dstpath, type):
    cfg = Config.fromfile(config_file)

    data_test = cfg.data['test']
    dataset = get_dataset(data_test)
    outputs_r = mmcv.load(resultfile_r)
    outputs_i = mmcv.load(resultfile_i)

    if type == 'OBB':
        obb_results_dict_r = OBBDetComp4_rgb(dataset, outputs_r)
        obb_results_dict_i = OBBDetComp4_infrared(dataset, outputs_i)
        current_thresh = 0.1

    dataset_type = cfg.dataset_type

    if 'obb_results_dict_r' in vars():
        if not os.path.exists(os.path.join(dstpath, 'Task1_results_r')):
            os.makedirs(os.path.join(dstpath, 'Task1_results_r'))

        for cls in obb_results_dict_r:
            with open(os.path.join(dstpath, 'Task1_results_r', cls + '.txt'), 'w') as obb_f_out_r:
                for index, outline in enumerate(obb_results_dict_r[cls]):
                    if index != (len(obb_results_dict_r[cls]) - 1):
                        obb_f_out_r.write(outline + '\n')
                    else:
                        obb_f_out_r.write(outline)

    if 'obb_results_dict_i' in vars():
        if not os.path.exists(os.path.join(dstpath, 'Task1_results_i')):
            os.makedirs(os.path.join(dstpath, 'Task1_results_i'))

        for cls in obb_results_dict_i:
            with open(os.path.join(dstpath, 'Task1_results_i', cls + '.txt'), 'w') as obb_f_out_i:
                for index, outline in enumerate(obb_results_dict_i[cls]):
                    if index != (len(obb_results_dict_i[cls]) - 1):
                        obb_f_out_i.write(outline + '\n')
                    else:
                        obb_f_out_i.write(outline)

if __name__ == '__main__':
    args = parse_args()
    config_file = args.config
    config_name = os.path.splitext(os.path.basename(config_file))[0]
    pkl_file_r = os.path.join('work_dirs', config_name, 'results_r.pkl')
    pkl_file_i = os.path.join('work_dirs', config_name, 'results_i.pkl')
    output_path = os.path.join('work_dirs', config_name)
    type = args.type
    parse_results_TS(config_file, pkl_file_r, pkl_file_i, output_path, type)

