import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset
import cv2

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         SegMapTransform, Numpy2Tensor)
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation
from .rotate_aug import RotateAugmentation
from .rotate_aug import RotateTestAugmentation

class TSCustomDataset(Dataset):
    """Two Stream Custom dataset for detection.
    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file_r,        # rgb
                 ann_file_i,        # infrared
                 img_prefix_r,      # rgb
                 img_prefix_i,      # infrared
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_semantic_seg=False,
                 seg_prefix=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 rotate_aug=None,
                 rotate_test_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False):
        # prefix of images path
        self.img_prefix_r = img_prefix_r     # rgb
        self.img_prefix_i = img_prefix_i     # infrared

        # load annotations (and proposals)
        self.img_infos_r = self.load_annotations_r(ann_file_r)       # rgb
        self.img_infos_i = self.load_annotations_i(ann_file_i)       # infrared

        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None

        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        # with semantic segmentation (stuff) annotation or not
        self.with_seg = with_semantic_seg
        # prefix of semantic segmentation map path
        self.seg_prefix = seg_prefix
        # rescale factor for segmentation maps
        self.seg_scale_factor = seg_scale_factor
        # in test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.seg_transform = SegMapTransform(self.size_divisor)
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # if use rotation augmentation
        if rotate_aug is not None:
            self.rotate_aug = RotateAugmentation(self.CLASSES, **rotate_aug)
        else:
            self.rotate_aug = None

        if rotate_test_aug is not None:
            #  dot not support argument settings currently
            self.rotate_test_aug = RotateTestAugmentation()
        else:
            self.rotate_test_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

    def __len__(self):
        return len(self.img_infos_i)             #infrared

    def load_annotations_r(self, ann_file):         # rgb
        return mmcv.load(ann_file)

    def load_annotations_i(self, ann_file):       # infrared
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info_r(self, idx):                    # rgb
        return self.img_infos_r[idx]['ann']

    def get_ann_info_i(self, idx):                    # infrared
        return self.img_infos_i[idx]['ann']

    def _filter_imgs_r(self, min_size=32):                    # rgb
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos_r):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _filter_imgs_i(self, min_size=32):                    # infrared
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos_i):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos_i[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def img_to_GRAY(self, img):
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        r,c = gray_img.shape[:2]   # row and col of a gray image
        piexs_sum=r*c              #sum of pixels in a gray image
        dark_points = (gray_img < 60)
        target_array = gray_img[dark_points]
        dark_sum = target_array.size
        dark_prop=dark_sum/(piexs_sum)
        if dark_prop >=0.45:
            w_dark = 1-dark_prop
        else:
            w_dark = 1
        return w_dark

    def prepare_train_img(self, idx):
        img_info_r = self.img_infos_r[idx]
        img_info_i = self.img_infos_i[idx]
        # load image
        img_r = mmcv.imread(osp.join(self.img_prefix_r, img_info_r['filename']))
        img_i = mmcv.imread(osp.join(self.img_prefix_i, img_info_i['filename']))
        crop_region = np.array([100, 100, 739, 611])
        img_r = mmcv.imcrop(img_r, crop_region)
        img_i = mmcv.imcrop(img_i, crop_region)
        
        w_dark = self.img_to_GRAY(img_r)

        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann_r = self.get_ann_info_r(idx)
        ann_i = self.get_ann_info_i(idx)
        gt_bboxes_r = ann_r['bboxes']
        gt_labels_r = ann_r['labels']
        gt_bboxes_i = ann_i['bboxes']
        gt_labels_i = ann_i['labels']

        if self.with_mask:
            gt_masks_r = ann_r['masks']
            gt_masks_i = ann_i['masks']
        if self.with_crowd:
            gt_bboxes_ignore_r = ann_r['bboxes_ignore']
            gt_bboxes_ignore_i = ann_i['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes_r) == 0 or len(gt_bboxes_i) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img_r, gt_bboxes_r, gt_labels_r = self.extra_aug(img_r, gt_bboxes_r,
                                                       gt_labels_r)
            img_i, gt_bboxes_i, gt_labels_i = self.extra_aug(img_i, gt_bboxes_i,
                                                       gt_labels_i)

        # rotate augmentation
        if self.rotate_aug is not None:
            # only support mask now, TODO: support none mask version
            img_r, gt_bboxes_r, gt_masks_r, gt_labels_r = self.rotate_aug(img_r, gt_bboxes_r,
                                                                      gt_masks_r, gt_labels_r, img_info_r['filename'])
            img_i, gt_bboxes_i, gt_masks_i, gt_labels_i = self.rotate_aug(img_i, gt_bboxes_i,
                                                                      gt_masks_i, gt_labels_i, img_info_i['filename'])

            gt_bboxes_r = np.array(gt_bboxes_r).astype(np.float32)
            gt_bboxes_i = np.array(gt_bboxes_i).astype(np.float32)
            # skip the image if there is no valid gt bbox
            if len(gt_bboxes_r) == 0 or len(gt_bboxes_i) == 0:
                return None

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale 
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        # RGB
        img_r, img_shape_r, pad_shape_r, scale_factor_r = self.img_transform(
            img_r, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img_r = img_r.copy()
        # Infrared
        img_i, img_shape_i, pad_shape_i, scale_factor_i = self.img_transform(
            img_i, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img_i = img_i.copy()

        if self.with_seg:
            gt_seg = mmcv.imread(
                osp.join(self.seg_prefix, img_info_r['file_name'].replace(
                    'jpg', 'png')),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]

        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape_i, scale_factor_i,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals

        gt_bboxes_r = self.bbox_transform(gt_bboxes_r, img_shape_r, scale_factor_r,
                                        flip)
        gt_bboxes_i = self.bbox_transform(gt_bboxes_i, img_shape_i, scale_factor_i,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore_r = self.bbox_transform(gt_bboxes_ignore_r, img_shape_r,
                                                   scale_factor_r, flip)
            gt_bboxes_ignore_i = self.bbox_transform(gt_bboxes_ignore_i, img_shape_i,
                                                   scale_factor_i, flip)
        if self.with_mask:
            # gt_masks = self.mask_transform(ann['masks'], pad_shape,
            #                                scale_factor, flip)
            gt_masks_r = self.mask_transform(gt_masks_r, pad_shape_r,
                                           scale_factor_r, flip)
            gt_masks_i = self.mask_transform(gt_masks_i, pad_shape_i,
                                           scale_factor_i, flip)

        ori_shape = (img_info_i['height'], img_info_i['width'], 3)
        img_meta_r = dict(
            ori_shape=ori_shape,
            img_shape=img_shape_r,
            pad_shape=pad_shape_r,
            scale_factor=scale_factor_r,
            flip=flip)
        img_meta_i = dict(
            ori_shape=ori_shape,
            img_shape=img_shape_i,
            pad_shape=pad_shape_i,
            scale_factor=scale_factor_i,
            flip=flip)

        data = dict(
            img_r=DC(to_tensor(img_r), stack=True),
            img_i=DC(to_tensor(img_i), stack=True),
            img_meta_r=DC(img_meta_r, cpu_only=True),
            img_meta_i=DC(img_meta_i, cpu_only=True),
            gt_bboxes_r=DC(to_tensor(gt_bboxes_r)),
            gt_bboxes_i=DC(to_tensor(gt_bboxes_i)),
            rgb_dark=DC(to_tensor(w_dark)))

        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels_r'] = DC(to_tensor(gt_labels_r))
            data['gt_labels_i'] = DC(to_tensor(gt_labels_i))
        if self.with_crowd:
            data['gt_bboxes_ignore_r'] = DC(to_tensor(gt_bboxes_ignore_r))
            data['gt_bboxes_ignore_i'] = DC(to_tensor(gt_bboxes_ignore_i))
        if self.with_mask:
            data['gt_masks_r'] = DC(gt_masks_r, cpu_only=True)
            data['gt_masks_i'] = DC(gt_masks_i, cpu_only=True)
        if self.with_seg:
            data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)
        return data


    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info_r = self.img_infos_r[idx]
        img_info_i = self.img_infos_i[idx]

        img_r = mmcv.imread(osp.join(self.img_prefix_r, img_info_r['filename']))
        img_i = mmcv.imread(osp.join(self.img_prefix_i, img_info_i['filename']))
        crop_region = np.array([100, 100, 739, 611])
        img_r = mmcv.imcrop(img_r, crop_region)
        img_i = mmcv.imcrop(img_i, crop_region)

        w_dark = self.img_to_GRAY(img_r)

        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None
        # TODO: make the flip and rotate at the same time
        # TODO: when implement the img rotation, we do not consider the proposals, add it in future
        def prepare_single(img_r, img_i, scale, flip, proposal=None):
            # rgb 
            _img_r, img_shape_r, pad_shape_r, scale_factor_r = self.img_transform(
                img_r, scale, flip, keep_ratio=self.resize_keep_ratio)
            # infrared 
            _img_i, img_shape_i, pad_shape_i, scale_factor_i = self.img_transform(
                img_i, scale, flip, keep_ratio=self.resize_keep_ratio)

            _img_r = to_tensor(_img_r)
            _img_i = to_tensor(_img_i)

            _img_meta_r = dict(
                ori_shape=(img_info_r['height'], img_info_r['width'], 3),
                img_shape=img_shape_r,
                pad_shape=pad_shape_r,
                scale_factor=scale_factor_r,
                flip=flip,
                angle=0)
            _img_meta_i = dict(
                ori_shape=(img_info_i['height'], img_info_i['width'], 3),
                img_shape=img_shape_i,
                pad_shape=pad_shape_i,
                scale_factor=scale_factor_i,
                flip=flip,
                angle=0)

            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img_r, _img_i, _img_meta_r, _img_meta_i, _proposal

        def prepare_rotation_single(img_r, img_i, scale, flip, angle):
            # rgb
            _img_r, img_shape_r, pad_shape_r, scale_factor_r = self.rotate_test_aug(
                img_r, angle=angle)
            _img_r, img_shape_r, pad_shape_r, scale_factor_r = self.img_transform(
                _img_r, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img_r = to_tensor(_img_r)
            # infrared
            _img_i, img_shape_i, pad_shape_i, scale_factor_i = self.rotate_test_aug(
                img_i, angle=angle)
            _img_i, img_shape_i, pad_shape_i, scale_factor_i = self.img_transform(
                _img_i, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img_i = to_tensor(_img_i)
            # if self.rotate_test_aug is not None:
            _img_meta_r = dict(
                ori_shape=(img_info_r['height'], img_info_r['width'], 3),
                img_shape=img_shape_r,
                pad_shape=pad_shape_r,
                scale_factor=scale_factor_r,
                flip=flip,
                angle=angle
            )
            _img_meta_i = dict(
                ori_shape=(img_info_i['height'], img_info_i['width'], 3),
                img_shape=img_shape_i,
                pad_shape=pad_shape_i,
                scale_factor=scale_factor_i,
                flip=flip,
                angle=angle
            )
            return _img_r, _img_i, _img_meta_r, _img_meta_i

        imgs_r = []
        img_metas_r = []
        imgs_i = []
        img_metas_i = []
        proposals = []

        for scale in self.img_scales:
            _img_r, _img_i, _img_meta_r, _img_meta_i, _proposal = prepare_single(
                img_r, img_i, scale, False, proposal)

            imgs_r.append(_img_r)
            img_metas_r.append(DC(_img_meta_r, cpu_only=True))
            imgs_i.append(_img_i)
            img_metas_i.append(DC(_img_meta_i, cpu_only=True))
            proposals.append(_proposal)

            if self.flip_ratio > 0:
                _img_r, _img_i, _img_meta_r, _img_meta_i, _proposal = prepare_single(
                    img_r, img_i, scale, True, proposal)
                imgs_r.append(_img_r)
                img_metas_r.append(DC(_img_meta_r, cpu_only=True))
                imgs_i.append(_img_i)
                img_metas_i.append(DC(_img_meta_i, cpu_only=True))
                proposals.append(_proposal)

        if self.rotate_test_aug is not None :
            # rotation augmentation
            # do not support proposals currently
            # img_show = img.copy()
            # mmcv.imshow(img_show, win_name='original')
            for angle in [90, 180, 270]:

                for scale in self.img_scales:
                    _img_r, _img_i, _img_meta_r, _img_meta_i = prepare_rotation_single(
                        img_r, img_i, scale, False, angle)
                    imgs_r.append(_img_r)
                    img_metas_r.append(DC(_img_meta_r, cpu_only=True))
                    imgs_i.append(_img_i)
                    img_metas_i.append(DC(_img_meta_i, cpu_only=True))
                    # proposals.append(_proposal)
                    if self.flip_ratio > 0:
                        _img_r, _img_i, _img_meta_r, _img_meta_i = prepare_rotation_single(
                            img_r, img_i, scale, True, proposal, angle)
                        imgs_r.append(_img_r)
                        img_metas_r.append(DC(_img_meta_r, cpu_only=True))
                        imgs_i.append(_img_i)
                        img_metas_i.append(DC(_img_meta_i, cpu_only=True))
                    # # # # TODO: rm if after debug
                    # if angle == 180:
                    #     img_show = _img.cpu().numpy().copy()
                    #     mmcv.imshow(img_show, win_name=str(angle))
                    # import pdb;pdb.set_trace()

        data = dict(img_r=imgs_r, img_i=imgs_i, img_meta_r=img_metas_r, img_meta_i=img_metas_i, rgb_dark=w_dark)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data
