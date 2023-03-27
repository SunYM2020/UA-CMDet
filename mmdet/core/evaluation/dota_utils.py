import mmcv
import numpy as np
import pycocotools.mask as mask_utils
# from mmdet.datasets import get_dataset

import cv2

# get dataset

import os
# from xx import *
def TuplePoly2Poly(poly):
    outpoly = [poly[0][0], poly[0][1],
                       poly[1][0], poly[1][1],
                       poly[2][0], poly[2][1],
                       poly[3][0], poly[3][1]
                       ]
    return outpoly

def seg2poly_old(rle):
    # TODO: debug for this function
    """
    This function transform a single encoded RLE to a single poly
    :param seg: RlE
    :return: poly
    """
    try:
        binary_mask = mask_utils.decode(rle)
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_areas = np.array(list(map(cv2.contourArea, contours)))
        max_id = contour_areas.argmax()
        max_contour = contours[max_id]
        rect = cv2.minAreaRect(max_contour)
        poly = cv2.boxPoints(rect)
        poly = TuplePoly2Poly(poly)
        return poly
    except:
        return []

def seg2poly(rle):
    # TODO: debug for this function
    """
    This function transform a single encoded RLE to a single poly
    :param seg: RlE
    :return: poly
    """
    try:
        binary_mask = mask_utils.decode(rle)
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)
        poly = cv2.boxPoints(rect)
        poly = TuplePoly2Poly(poly)
        return poly
    except:
        return []

def OBBDet2Comp4(dataset, results):
    results_dict = {}
    for idx in range(len(dataset)):
        filename = dataset.img_infos[idx]['filename']
        result = results[idx]
        for label in range(len(result)):
            rbboxes = result[label]
            cls_name = dataset.CLASSES[label]
            polys = RotBox2Polys(rbboxes[:, :-1])
            if cls_name not in results_dict:
                results_dict[cls_name] = []
            for i in range(rbboxes.shape[0]):
                poly = polys[i]
                score = float(rbboxes[i][-1])
                outline = filename + ' ' + str(score) + ' ' + ' '.join(map(str, poly))
                results_dict[cls_name].append(outline)
    return results_dict

def OBBDetComp4(dataset, results):
    results_dict = {}
    for idx in range(len(dataset)):
        # filename = dataset.img_infos[idx]['filename']
        filename = dataset.img_infos_i[idx]['filename']
        result = results[idx]
        for label in range(len(result)):
            rbboxes = result[label]
            # import pdb
            # pdb.set_trace()
            cls_name = dataset.CLASSES[label]
            if cls_name not in results_dict:
                results_dict[cls_name] = []
            for i in range(rbboxes.shape[0]):
                poly = rbboxes[i][:-1]
                score = float(rbboxes[i][-1])
                outline = filename + ' ' + str(score) + ' ' + ' '.join(map(str, poly))
                results_dict[cls_name].append(outline)
    return results_dict

def OBBDetComp4_rgb(dataset, results):
    results_dict = {}
    for idx in range(len(dataset)):
        filename = dataset.img_infos_r[idx]['filename']
        result = results[idx]
        for label in range(len(result)):
            rbboxes = result[label]
            # import pdb
            # pdb.set_trace()
            cls_name = dataset.CLASSES[label]
            if cls_name not in results_dict:
                results_dict[cls_name] = []
            for i in range(rbboxes.shape[0]):
                poly = rbboxes[i][:-1]
                score = float(rbboxes[i][-1])
                outline = filename + ' ' + str(score) + ' ' + ' '.join(map(str, poly))
                results_dict[cls_name].append(outline)
    return results_dict

def OBBDetComp4_infrared(dataset, results):
    results_dict = {}
    for idx in range(len(dataset)):
        filename = dataset.img_infos_i[idx]['filename']
        result = results[idx]
        for label in range(len(result)):
            rbboxes = result[label]
            # import pdb
            # pdb.set_trace()
            cls_name = dataset.CLASSES[label]
            if cls_name not in results_dict:
                results_dict[cls_name] = []
            for i in range(rbboxes.shape[0]):
                poly = rbboxes[i][:-1]
                score = float(rbboxes[i][-1])
                outline = filename + ' ' + str(score) + ' ' + ' '.join(map(str, poly))
                results_dict[cls_name].append(outline)
    return results_dict

def HBBDet2Comp4(dataset, results):
    results_dict = {}
    for idx in range(len(dataset)):
        # print('idx: ', idx, 'total: ', len(dataset))
        filename = dataset.img_infos[idx]['filename']
        result = results[idx]
        for label in range(len(result)):
            bboxes = result[label]

            # try:
            cls_name = dataset.CLASSES[label]
            # except:
            #     import pdb
            #     pdb.set_trace()
            if cls_name not in results_dict:
                results_dict[cls_name] = []
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                score = float(bboxes[i][-1])
                outline = filename + ' ' + str(score) + ' ' + ' '.join(map(str, bbox))
                results_dict[cls_name].append(outline)
    return results_dict


def HBBSeg2Comp4(dataset, results):
    hbb_results_dict = {}
    obb_results_dict = {}
    prog_bar = mmcv.ProgressBar(len(dataset))
    for idx in range(len(dataset)):
        # import pdb
        # pdb.set_trace()
        filename = dataset.img_infos[idx]['filename']
        # print('filename: ', filename)
        det, seg = results[idx]
        for label in range(len(det)):
            bboxes = det[label]
            segms = seg[label]
            cls_name = dataset.CLASSES[label]
            if cls_name not in hbb_results_dict:
                hbb_results_dict[cls_name] = []
            if cls_name not in obb_results_dict:
                obb_results_dict[cls_name] = []
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i].tolist()
                score = float(bboxes[i][-1])
                hbb_outline = filename + ' ' + str(score) + ' ' + ' '.join(map(str, bbox))
                hbb_results_dict[cls_name].append(hbb_outline)

                poly = seg2poly(segms[i])
                if poly != []:
                    score = float(bboxes[i][-1])
                    obb_outline = filename + ' ' + str(score) + ' ' + ' '.join(map(str, poly))
                    obb_results_dict[cls_name].append(obb_outline)
        prog_bar.update()
    return hbb_results_dict, obb_results_dict

def HBBOBB2Comp4(dataset, results):
    hbb_results_dict = {}
    obb_results_dict = {}
    for idx in range(len(dataset)):
        filename = dataset.img_infos[idx]['filename']

        hbb_det, obb_det = results[idx]

        for label in range(len(hbb_det)):
            bboxes = hbb_det[label]
            rbboxes = obb_det[label]
            cls_name = dataset.CLASSES[label]
            if cls_name not in hbb_results_dict:
                hbb_results_dict[cls_name] = []
            if cls_name not in obb_results_dict:
                obb_results_dict[cls_name] = []
            # parse hbb results
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                score = float(bboxes[i][-1])
                outline = filename + ' ' + str(score) + ' ' + ' '.join(map(str, bbox))
                hbb_results_dict[cls_name].append(outline)
            # parse obb results
            for i in range(rbboxes.shape[0]):
                poly = rbboxes[i]
                score = float(rbboxes[i][-1])
                outline = filename + ' ' + str(score) + ' ' + ' '.join(map(str, poly))
                obb_results_dict[cls_name].append(outline)
    return hbb_results_dict, obb_results_dict












