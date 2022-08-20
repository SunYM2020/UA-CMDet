import logging
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import torch.nn as nn
import pycocotools.mask as maskUtils

from mmdet.core import tensor2imgs, get_classes

import torch
import torch.nn.functional as F

class BaseDetectorNew(nn.Module):
    """Base class for detectors"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetectorNew, self).__init__()

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_shared_head_rbbox(self):
        return hasattr(self, 'shared_head_rbbox') and self.shared_head_rbbox is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_rbbox(self):
        return hasattr(self, 'rbbox_head') and self.rbbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @property
    def with_neck_r(self):
        return hasattr(self, 'neck_r') and self.neck_r is not None

    @property
    def with_shared_head_r(self):
        return hasattr(self, 'shared_head_r') and self.shared_head_r is not None

    @property
    def with_shared_head_rbbox_r(self):
        return hasattr(self, 'shared_head_rbbox_r') and self.shared_head_rbbox_r is not None

    @property
    def with_bbox_r(self):
        return hasattr(self, 'bbox_head_r') and self.bbox_head_r is not None

    @property
    def with_rbbox_r(self):
        return hasattr(self, 'rbbox_head_r') and self.rbbox_head_r is not None

    @property
    def with_mask_r(self):
        return hasattr(self, 'mask_head_r') and self.mask_head_r is not None

    @property
    def with_neck_i(self):
        return hasattr(self, 'neck_i') and self.neck_i is not None

    @property
    def with_shared_head_i(self):
        return hasattr(self, 'shared_head_i') and self.shared_head_i is not None

    @property
    def with_shared_head_rbbox_i(self):
        return hasattr(self, 'shared_head_rbbox_i') and self.shared_head_rbbox_i is not None

    @property
    def with_bbox_i(self):
        return hasattr(self, 'bbox_head_i') and self.bbox_head_i is not None

    @property
    def with_rbbox_i(self):
        return hasattr(self, 'rbbox_head_i') and self.rbbox_head_i is not None

    @property
    def with_mask_i(self):
        return hasattr(self, 'mask_head_i') and self.mask_head_i is not None

    @property
    def with_shared_head_f(self):
        return hasattr(self, 'shared_head_f') and self.shared_head_f is not None

    @property
    def with_shared_head_rbbox_f(self):
        return hasattr(self, 'shared_head_rbbox_f') and self.shared_head_rbbox_f is not None

    @property
    def with_bbox_f(self):
        return hasattr(self, 'bbox_head_f') and self.bbox_head_f is not None

    @property
    def with_rbbox_f(self):
        return hasattr(self, 'rbbox_head_f') and self.rbbox_head_f is not None

    @property
    def with_mask_f(self):
        return hasattr(self, 'mask_head_f') and self.mask_head_f is not None

    @abstractmethod
    def extract_feat_rgb(self, imgs_r):
        pass

    @abstractmethod
    def extract_feat_infrared(self, imgs_i):
        pass

    @abstractmethod
    def _init_layers(self):
        pass

    def extract_feats_rgb(self, imgs_r):
        assert isinstance(imgs_r, list)
        for img_r in imgs_r:
            yield self.extract_feat_rgb(img_r)

    def extract_feats_infrared(self, imgs_i):
        assert isinstance(imgs_i, list)
        for img_i in imgs_i:
            yield self.extract_feat_infrared(img_i)

    def extract_feat_fusion(self, imgs_r, imgs_i):
        assert isinstance(imgs_r, list)
        assert isinstance(imgs_i, list)
        for i in range(len(imgs_i)):
            x_r = self.extract_feat_rgb(imgs_r[i])
            x_i = self.extract_feat_infrared(img_i[i])
            x_f = []
            for j in range(len(x_r)):
                concat = torch.cat((x_r[j], x_i[j]), dim=1)
                concat = self.fusion_conv(concat)
                #concat = F.relu(concat, inplace=True)
                x_f.append(concat)
            yield tuple(x_f)

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, imgs_r, imgs_i, img_metas_r, img_metas_i, rgb_dark, **kwargs):
        for var_r, name_r in [(imgs_r, 'imgs_r'), (img_metas_r, 'img_metas_r')]:
            if not isinstance(var_r, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name_r, type(var_r)))
        for var_i, name_i in [(imgs_i, 'imgs_i'), (img_metas_i, 'img_metas_i')]:
            if not isinstance(var_i, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name_i, type(var_i)))

        num_augs_r = len(imgs_r)
        if num_augs_r != len(img_metas_r):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs_r), len(img_metas_r)))
        num_augs_i = len(imgs_i)
        if num_augs_i != len(img_metas_i):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs_i), len(img_metas_i)))

        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs_r[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs_r == 1 and num_augs_i == 1:
            return self.simple_test(imgs_r[0], imgs_i[0], img_metas_r[0], img_metas_i[0], rgb_dark[0], **kwargs)
        else:
            return self.aug_test(imgs_r, imgs_i, img_metas_r, img_metas_i, rgb_dark, **kwargs)

    def forward(self, img_r, img_i, img_meta_r, img_meta_i, rgb_dark, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img_r, img_i, img_meta_r, img_meta_i, rgb_dark, **kwargs)
        else:
            return self.forward_test(img_r, img_i, img_meta_r, img_meta_i, rgb_dark, **kwargs)

    def show_result(self,
                    data,
                    result,
                    img_norm_cfg,
                    dataset=None,
                    score_thr=0.3):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None

        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        assert len(imgs) == len(img_metas)

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            bboxes = np.vstack(bbox_result)
            # draw segmentation masks
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    color_mask = np.random.randint(
                        0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            mmcv.imshow_det_bboxes(
                img_show,
                bboxes,
                labels,
                class_names=class_names,
                score_thr=score_thr)
