# --------------------------------------------------------
# DroneVehicle_evaluation_obb
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yiming Sun, based on code from Bharath Hariharan
# --------------------------------------------------------

"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 5 result files
    search for PATH_TO_BE_CONFIGURED to config the paths
"""
import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt
import polyiou
from functools import partial
import pdb

def object_classes():
    return ['car', 'freight_car', 'truck', 'bus', 'van']

def get_segmentation(points):
    return [points[0], points[1], points[2] + points[0], points[1],
             points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]

def parse_gt(xml_path):
    """
    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation = []

    for obj in root.findall('object'):
        object_struct = {}

        name = obj.find('name').text
        if name == '*':
            continue
        if name in ['feright car', 'feright_car', 'feright', 'freight car', 'freight_car']:
            object_struct['name'] = 'freight_car'
        elif name == 'truvk':
            object_struct['name'] = 'truck'
        else:
            object_struct['name'] = name

        object_struct['difficult'] = 0

        if obj.find('polygon'):
            polygon = obj.find('polygon')
            x1 = float(polygon.find('x1').text)-100
            y1 = float(polygon.find('y1').text)-100
            x2 = float(polygon.find('x2').text)-100
            y2 = float(polygon.find('y2').text)-100
            x3 = float(polygon.find('x3').text)-100
            y3 = float(polygon.find('y3').text)-100
            x4 = float(polygon.find('x4').text)-100
            y4 = float(polygon.find('y4').text)-100
            object_struct['bbox'] = [x1, y1, x2, y2, x3, y3, x4, y4]

            annotation.append(object_struct)

        if obj.find('bndbox'):
            bnd_box = obj.find('bndbox')
            xmin = float(bnd_box.find('xmin').text)-100
            ymin = float(bnd_box.find('ymin').text)-100
            xmax = float(bnd_box.find('xmax').text)-100
            ymax = float(bnd_box.find('ymax').text)-100
            w = xmax - xmin
            h = ymax - ymin
            object_struct['bbox'] = get_segmentation([xmin, ymin, w, h])

            annotation.append(object_struct)
    return annotation

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
            # cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """

    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    recs = {}
    for i, imagename in enumerate(imagenames):
        #print('parse_files name: ', annopath.format(imagename))
        recs[imagename] = parse_gt(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d].split(".")[0]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]
            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):

                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]
        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def main():
    detpath = r'/root/UA-CMDet/work_dirs/UACMDet/Task1_results/{:s}.txt'
    # annopath = r'/root/UA-CMDet/data/DroneVehicle/val/vallabelr/{:s}.xml'    # for eval val set
    annopath = r'/root/UA-CMDet/data/DroneVehicle/test/testlabelr/{:s}.xml'     # for eval test set

    # imagesetfile = r'root/UA-CMDet/filename_val.txt'      # for eval val set
    imagesetfile = r'root/UA-CMDet/filename_test.txt'       # for eval test set

    label_ids = {name: i + 1 for i, name in enumerate(object_classes())}
    print(label_ids)

    # For DroneVehicle
    classnames = ['car', 'freight_car', 'truck', 'bus', 'van']
    classaps = []
    map = 0
    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=True)
        map = map + ap
        print('ap: ', ap)
        classaps.append(ap)

    map = map/len(classnames)
    print('map:', map*100)
    classaps = 100*np.array(classaps)
    print('classaps: ', classaps)


if __name__ == '__main__':
    main()
