from __future__ import division

import torch
import torch.nn as nn

import numpy as np
import cv2
import torch.nn.functional as F
from mmcv.cnn import normal_init

from .base_new import BaseDetectorNew
from .test_mixins import RPNTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import (build_assigner, bbox2roi, dbbox2roi, bbox2result, build_sampler,
                        dbbox2result, merge_aug_masks, roi2droi, mask2poly,
                        get_best_begin_point, polygonToRotRectangle_batch,
                        gt_mask_bp_obbs_list, choose_best_match_batch,
                        choose_best_Rroi_batch, dbbox_rotate_mapping, bbox_rotate_mapping)
from mmdet.core import (bbox_mapping, merge_aug_proposals, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms, merge_rotate_aug_proposals,
                        merge_rotate_aug_bboxes, multiclass_nms_rbbox)
import copy
from mmdet.core import RotBox2Polys, polygonToRotRectangle_batch
from mmdet.core.bbox.geometry import bbox_overlaps_cy, rbbox_overlaps_cy

@DETECTORS.register_module
class LightThreeStreamUncertainty(BaseDetectorNew, RPNTestMixin):

    def __init__(self,
                 backbone_r,
                 backbone_i,
                 neck_r=None,
                 neck_i=None,
                 rpn_head_r=None,
                 rpn_head_i=None,
                 rpn_head_f=None,
                 bbox_roi_extractor_r=None,
                 bbox_roi_extractor_i=None,
                 bbox_roi_extractor_f=None,
                 bbox_head_r=None,
                 bbox_head_i=None,
                 bbox_head_f=None,
                 rbbox_roi_extractor_r=None,
                 rbbox_roi_extractor_i=None,
                 rbbox_roi_extractor_f=None,
                 rbbox_head_r=None,
                 rbbox_head_i=None,
                 rbbox_head_f=None,
                 shared_head_r=None,
                 shared_head_i=None,
                 shared_head_f=None,
                 shared_head_rbbox_r=None,
                 shared_head_rbbox_i=None,
                 shared_head_rbbox_f=None,
                 mask_roi_extractor_r=None,
                 mask_roi_extractor_i=None,
                 mask_roi_extractor_f=None,
                 mask_head_r=None,
                 mask_head_i=None,
                 mask_head_f=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        assert bbox_roi_extractor_r is not None     # rgb
        assert bbox_head_r is not None
        assert bbox_roi_extractor_i is not None     # infrared
        assert bbox_head_i is not None
        assert bbox_roi_extractor_f is not None     # fusion
        assert bbox_head_f is not None

        assert rbbox_roi_extractor_r is not None    # rgb
        assert rbbox_head_r is not None
        assert rbbox_roi_extractor_i is not None    # infrared
        assert rbbox_head_i is not None
        assert rbbox_roi_extractor_f is not None    # Fusion
        assert rbbox_head_f is not None

        super(LightThreeStreamUncertainty, self).__init__()

        self.backbone_r = builder.build_backbone(backbone_r)    # rgb
        self.backbone_i = builder.build_backbone(backbone_i)    # infrared

        if neck_r is not None:                                  # rgb
            self.neck_r = builder.build_neck(neck_r)
        if neck_i is not None:                                  # infrared
            self.neck_i = builder.build_neck(neck_i)

        if rpn_head_r is not None:                                # rgb
            self.rpn_head_r = builder.build_head(rpn_head_r)
        if rpn_head_i is not None:                                # infrared
            self.rpn_head_i = builder.build_head(rpn_head_i)
        if rpn_head_f is not None:                                # Fusion
            self.rpn_head_f = builder.build_head(rpn_head_f)

        if shared_head_r is not None:                                       # rgb
            self.shared_head_r = builder.build_shared_head(shared_head_r)
        if shared_head_i is not None:                                       # infrared
            self.shared_head_i = builder.build_shared_head(shared_head_i)
        if shared_head_f is not None:                                       # Fusion
            self.shared_head_f = builder.build_shared_head(shared_head_f)

        if shared_head_rbbox_r is not None:                                           # rgb
            self.shared_head_rbbox_r = builder.build_shared_head(shared_head_rbbox_r)
        if shared_head_rbbox_i is not None:                                           # infrared
            self.shared_head_rbbox_i = builder.build_shared_head(shared_head_rbbox_i)
        if shared_head_rbbox_f is not None:                                           # fusion
            self.shared_head_rbbox_f = builder.build_shared_head(shared_head_rbbox_f)

        if bbox_head_r is not None:                                      # rgb
            self.bbox_roi_extractor_r = builder.build_roi_extractor(
                bbox_roi_extractor_r)
            self.bbox_head_r = builder.build_head(bbox_head_r)
        if bbox_head_i is not None:                                      # infrared
            self.bbox_roi_extractor_i = builder.build_roi_extractor(
                bbox_roi_extractor_i)
            self.bbox_head_i = builder.build_head(bbox_head_i)
        if bbox_head_f is not None:                                      # infrared
            self.bbox_roi_extractor_f = builder.build_roi_extractor(
                bbox_roi_extractor_f)
            self.bbox_head_f = builder.build_head(bbox_head_f)

        # import pdb
        # pdb.set_trace()
        if rbbox_head_r is not None:                                      # rgb
            self.rbbox_roi_extractor_r = builder.build_roi_extractor(
                rbbox_roi_extractor_r)
            self.rbbox_head_r = builder.build_head(rbbox_head_r)
        if rbbox_head_i is not None:                                      # infrared
            self.rbbox_roi_extractor_i = builder.build_roi_extractor(
                rbbox_roi_extractor_i)
            self.rbbox_head_i = builder.build_head(rbbox_head_i)
        if rbbox_head_f is not None:                                      # fusion
            self.rbbox_roi_extractor_f = builder.build_roi_extractor(
                rbbox_roi_extractor_f)
            self.rbbox_head_f = builder.build_head(rbbox_head_f)

        if mask_head_r is not None:                                      # rgb
            if mask_roi_extractor_r is not None:
                self.mask_roi_extractor_r = builder.build_roi_extractor(
                    mask_roi_extractor_r)
                self.share_roi_extractor_r = False
            else:
                self.share_roi_extractor_r = True
                self.mask_roi_extractor_r = self.rbbox_roi_extractor_r
            self.mask_head_r = builder.build_head(mask_head_r)

        if mask_head_i is not None:                                      # infrared
            if mask_roi_extractor_i is not None:
                self.mask_roi_extractor_i = builder.build_roi_extractor(
                    mask_roi_extractor_i)
                self.share_roi_extractor_i = False
            else:
                self.share_roi_extractor_i = True
                self.mask_roi_extractor_i = self.rbbox_roi_extractor_i
            self.mask_head_i = builder.build_head(mask_head_i)

        if mask_head_f is not None:                                      # Fusion
            if mask_roi_extractor_f is not None:
                self.mask_roi_extractor_f = builder.build_roi_extractor(
                    mask_roi_extractor_f)
                self.share_roi_extractor_f = False
            else:
                self.share_roi_extractor_f = True
                self.mask_roi_extractor_f = self.rbbox_roi_extractor_f
            self.mask_head_f = builder.build_head(mask_head_f)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers()
        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn_r(self):                                                 # rgb
        return hasattr(self, 'rpn_head_r') and self.rpn_head_r is not None

    @property
    def with_rpn_i(self):                                                 # infrared
        return hasattr(self, 'rpn_head_i') and self.rpn_head_i is not None

    @property
    def with_rpn_f(self):                                                 # Fusion
        return hasattr(self, 'rpn_head_f') and self.rpn_head_f is not None

    def _init_layers(self):
        self.fusion_conv = nn.Conv2d(512, 256, 1)

    def init_weights(self, pretrained=None):
        super(LightThreeStreamUncertainty, self).init_weights(pretrained)

        self.backbone_r.init_weights(pretrained=pretrained)
        self.backbone_i.init_weights(pretrained=pretrained)

        normal_init(self.fusion_conv, std=0.01)

        if self.with_neck_r:
            if isinstance(self.neck_r, nn.Sequential):
                for m in self.neck_r:
                    m.init_weights()
            else:
                self.neck_r.init_weights()
        if self.with_rpn_r:
            self.rpn_head_r.init_weights()
        if self.with_shared_head_r:
            self.shared_head_r.init_weights(pretrained=pretrained)
        if self.with_shared_head_rbbox_r:
            self.shared_head_rbbox_r.init_weights(pretrained=pretrained)
        if self.with_bbox_r:
            self.bbox_roi_extractor_r.init_weights()
            self.bbox_head_r.init_weights()
        if self.with_rbbox_r:
            self.rbbox_roi_extractor_r.init_weights()
            self.rbbox_head_r.init_weights()
        if self.with_mask_r:
            self.mask_head_r.init_weights()
            if not self.share_roi_extractor_r:
                self.mask_roi_extractor_r.init_weights()

        if self.with_neck_i:
            if isinstance(self.neck_i, nn.Sequential):
                for m in self.neck_i:
                    m.init_weights()
            else:
                self.neck_i.init_weights()
        if self.with_rpn_i:
            self.rpn_head_i.init_weights()
        if self.with_shared_head_i:
            self.shared_head_i.init_weights(pretrained=pretrained)
        if self.with_shared_head_rbbox_i:
            self.shared_head_rbbox_i.init_weights(pretrained=pretrained)
        if self.with_bbox_i:
            self.bbox_roi_extractor_i.init_weights()
            self.bbox_head_i.init_weights()
        if self.with_rbbox_i:
            self.rbbox_roi_extractor_i.init_weights()
            self.rbbox_head_i.init_weights()
        if self.with_mask_i:
            self.mask_head_i.init_weights()
            if not self.share_roi_extractor_i:
                self.mask_roi_extractor_i.init_weights()

        if self.with_rpn_f:
            self.rpn_head_f.init_weights()
        if self.with_shared_head_f:
            self.shared_head_f.init_weights(pretrained=pretrained)
        if self.with_shared_head_rbbox_f:
            self.shared_head_rbbox_f.init_weights(pretrained=pretrained)
        if self.with_bbox_f:
            self.bbox_roi_extractor_f.init_weights()
            self.bbox_head_f.init_weights()
        if self.with_rbbox_f:
            self.rbbox_roi_extractor_f.init_weights()
            self.rbbox_head_f.init_weights()
        if self.with_mask_f:
            self.mask_head_f.init_weights()
            if not self.share_roi_extractor_f:
                self.mask_roi_extractor_f.init_weights()

    def extract_feat_rgb(self, img):
        x = self.backbone_r(img)
        if self.with_neck_r:
            x = self.neck_r(x)
        return x

    def extract_feat_infrared(self, img):
        x = self.backbone_i(img)
        if self.with_neck_i:
            x = self.neck_i(x)
        return x

    def forward_train(self,
                      img_r,
                      img_i,
                      img_meta_r,
                      img_meta_i,
                      rgb_dark,
                      gt_bboxes_r,
                      gt_bboxes_i,
                      gt_labels_r,
                      gt_labels_i,
                      gt_bboxes_ignore_r=None,
                      gt_bboxes_ignore_i=None,
                      gt_masks_r=None,
                      gt_masks_i=None,
                      proposals=None):

        x_r = self.extract_feat_rgb(img_r)
        x_i = self.extract_feat_infrared(img_i)

        x_f = []
        for i in range(len(x_r)):
            concat = torch.cat((x_r[i].detach(), x_i[i].detach()), dim=1)          
            concat = self.fusion_conv(concat)
            x_f.append(concat)
        x_f = tuple(x_f)

        losses = dict()

        gt_obbs_r = gt_mask_bp_obbs_list(gt_masks_r)
        gt_obbs_i = gt_mask_bp_obbs_list(gt_masks_i)

        obb_u_r = []
        obb_u_i = []
        for i in range(min(len(gt_obbs_r),len(gt_obbs_i))):    
            gt_obbs_best_roi_r = choose_best_Rroi_batch(gt_obbs_r[i])
            gt_obbs_best_roi_i = choose_best_Rroi_batch(gt_obbs_i[i])

            overlaps = rbbox_overlaps_cy(gt_obbs_best_roi_r, gt_obbs_best_roi_i)            
            num_gts_r, num_gts_i = len(overlaps), len(overlaps[0])

            W_dark = rgb_dark[i].cpu().numpy()
            
            obb_uncertains_r = [W_dark[0] for _ in range(num_gts_r)] 
            obb_uncertains_i = [1 for _ in range(num_gts_i)]

            list_gt_obbs_r = gt_obbs_r[i].tolist()
            list_gt_label_r = gt_labels_r[i].cpu().numpy().tolist()
            list_gtboxes_r = gt_bboxes_r[i].cpu().numpy().tolist()
            gt_device = gt_labels_r[i].device

            list_gt_obbs_i = gt_obbs_i[i].tolist()
            list_gt_label_i = gt_labels_i[i].cpu().numpy().tolist()
            list_gtboxes_i = gt_bboxes_i[i].cpu().numpy().tolist()

            for k in range(num_gts_i):
                if np.mean(overlaps[:,k]) < 0.01:
                    obb_uncertains_r.append(0.1)
                    list_gt_obbs_r.append(gt_obbs_i[i][k].tolist())
                    list_gtboxes_r.append(gt_bboxes_i[i][k].cpu().numpy().tolist())
                    list_gt_label_r.append(gt_labels_i[i][k].cpu().numpy().tolist())

            gt_obbs_r[i] = np.array(list_gt_obbs_r)
            gt_bboxes_r[i] = torch.Tensor(list_gtboxes_r).to(gt_device)
            gt_labels_r[i] = torch.Tensor(list_gt_label_r).long().to(gt_device)

            inds = np.where((overlaps < 0.8)*(overlaps > 0.1))

            for index in range(len(inds[0])):
                rgb_index = inds[0][index]
                infrared_index = inds[1][index]
                if max(overlaps[rgb_index]) < 0.8:
                    obb_uncertain_r = overlaps[rgb_index][infrared_index]
                    obb_uncertains_r[rgb_index] = obb_uncertain_r * obb_uncertains_r[rgb_index]

            obb_u_r.append(obb_uncertains_r)

            for j in range(num_gts_r):
                if np.mean(overlaps[j]) < 0.01:
                    obb_uncertains_i.append(1)
                    list_gt_obbs_i.append(gt_obbs_r[i][j].tolist())
                    list_gtboxes_i.append(gt_bboxes_r[i][j].cpu().numpy().tolist())
                    list_gt_label_i.append(gt_labels_r[i][j].cpu().numpy().tolist())

            gt_obbs_i[i] = np.array(list_gt_obbs_i)
            gt_bboxes_i[i] = torch.Tensor(list_gtboxes_i).to(gt_device)
            gt_labels_i[i] = torch.Tensor(list_gt_label_i).long().to(gt_device)

            obb_u_i.append(obb_uncertains_i)

        if self.with_rpn_r:
            rpn_outs_r = self.rpn_head_r(x_r)
            rpn_loss_inputs_r = rpn_outs_r + (gt_bboxes_r, img_meta_r, obb_u_r,      
                                          self.train_cfg.rpn)
            rpn_losses_r = self.rpn_head_r.loss_r(
                *rpn_loss_inputs_r, gt_bboxes_ignore=gt_bboxes_ignore_r)
            losses.update(rpn_losses_r)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            proposal_inputs_r = rpn_outs_r + (img_meta_r, proposal_cfg)

            proposal_list_r = self.rpn_head_r.get_bboxes(*proposal_inputs_r)
        else:
            proposal_list_r = proposals

        if self.with_rpn_i:
            rpn_outs_i = self.rpn_head_i(x_i)
            rpn_loss_inputs_i = rpn_outs_i + (gt_bboxes_i, img_meta_i, obb_u_i,
                                          self.train_cfg.rpn)
            rpn_losses_i = self.rpn_head_i.loss_i(
                *rpn_loss_inputs_i, gt_bboxes_ignore=gt_bboxes_ignore_i)
            losses.update(rpn_losses_i)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            proposal_inputs_i = rpn_outs_i + (img_meta_i, proposal_cfg)

            proposal_list_i = self.rpn_head_i.get_bboxes(*proposal_inputs_i)
        else:
            proposal_list_i = proposals

        if self.with_rpn_f:
            rpn_outs_f = self.rpn_head_f(x_f)
            rpn_loss_inputs_f = rpn_outs_f + (gt_bboxes_i, img_meta_i, obb_u_i,
                                          self.train_cfg.rpn)
            rpn_losses_f = self.rpn_head_f.loss_f(
                *rpn_loss_inputs_f, gt_bboxes_ignore=gt_bboxes_ignore_i)
            losses.update(rpn_losses_f)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            proposal_inputs_f = rpn_outs_f + (img_meta_i, proposal_cfg)

            proposal_list_f = self.rpn_head_f.get_bboxes(*proposal_inputs_f)
        else:
            proposal_list_f = proposals

        if self.with_bbox_r or self.with_mask_r:
            bbox_assigner_r = build_assigner(self.train_cfg.rcnn[0].assigner)
            bbox_sampler_r = build_sampler(
                self.train_cfg.rcnn[0].sampler, context=self)
            num_imgs_r = img_r.size(0)
            if gt_bboxes_ignore_r is None:
                gt_bboxes_ignore_r = [None for _ in range(num_imgs_r)]
            sampling_results_r = []
            for i in range(num_imgs_r):
                assign_result_r = bbox_assigner_r.assign(
                    proposal_list_r[i], gt_bboxes_r[i], gt_bboxes_ignore_r[i],
                    gt_labels_r[i])
                sampling_result_r = bbox_sampler_r.sample(
                    assign_result_r,
                    proposal_list_r[i],
                    gt_bboxes_r[i],                                                
                    gt_labels_r[i],                                                
                    feats_r=[lvl_feat_r[i][None] for lvl_feat_r in x_r])
                sampling_results_r.append(sampling_result_r)

        if self.with_bbox_i or self.with_mask_i:
            bbox_assigner_i = build_assigner(self.train_cfg.rcnn[0].assigner)
            bbox_sampler_i = build_sampler(
                self.train_cfg.rcnn[0].sampler, context=self)
            num_imgs_i = img_i.size(0)
            if gt_bboxes_ignore_i is None:
                gt_bboxes_ignore_i = [None for _ in range(num_imgs_i)]
            sampling_results_i = []
            for i in range(num_imgs_i):
                assign_result_i = bbox_assigner_i.assign(
                    proposal_list_i[i], gt_bboxes_i[i], gt_bboxes_ignore_i[i],
                    gt_labels_i[i])
                sampling_result_i = bbox_sampler_i.sample(
                    assign_result_i,
                    proposal_list_i[i],
                    gt_bboxes_i[i],
                    gt_labels_i[i],
                    feats_i=[lvl_feat_i[i][None] for lvl_feat_i in x_i])
                sampling_results_i.append(sampling_result_i)

        if self.with_bbox_f or self.with_mask_f:
            bbox_assigner_f = build_assigner(self.train_cfg.rcnn[0].assigner)
            bbox_sampler_f = build_sampler(
                self.train_cfg.rcnn[0].sampler, context=self)
            num_imgs_f = img_i.size(0)
            if gt_bboxes_ignore_i is None:
                gt_bboxes_ignore_i = [None for _ in range(num_imgs_f)]
            sampling_results_f = []
            for i in range(num_imgs_f):
                assign_result_f = bbox_assigner_f.assign(
                    proposal_list_f[i], gt_bboxes_i[i], gt_bboxes_ignore_i[i],
                    gt_labels_i[i])
                sampling_result_f = bbox_sampler_f.sample(
                    assign_result_f,
                    proposal_list_f[i],
                    gt_bboxes_i[i],
                    gt_labels_i[i],
                    feats_f=[lvl_feat_f[i][None] for lvl_feat_f in x_f])
                sampling_results_f.append(sampling_result_f)

        if self.with_bbox_r:
            rois_r = bbox2roi([res_r.bboxes for res_r in sampling_results_r])
            bbox_feats_r = self.bbox_roi_extractor_r(
                x_r[:self.bbox_roi_extractor_r.num_inputs], rois_r)

            if self.with_shared_head_r:
                bbox_feats_r = self.shared_head_r(bbox_feats_r)
            cls_score_r, bbox_pred_r = self.bbox_head_r(bbox_feats_r)

            rbbox_targets_r = self.bbox_head_r.get_target(
                sampling_results_r, gt_obbs_r, obb_u_r, gt_labels_r, self.train_cfg.rcnn[0])

            loss_bbox_r = self.bbox_head_r.loss_r(cls_score_r, bbox_pred_r,
                                            *rbbox_targets_r)
            for name_r, value_r in loss_bbox_r.items():
                losses['s{}.{}'.format(0, name_r)] = (value_r)

        pos_is_gts_r = [res_r.pos_is_gt for res_r in sampling_results_r]
        roi_labels_r = rbbox_targets_r[0]
        with torch.no_grad():
            rotated_proposal_list_r = self.bbox_head_r.refine_rbboxes(
                roi2droi(rois_r), roi_labels_r, bbox_pred_r, pos_is_gts_r, img_meta_r
            )

        if self.with_bbox_i:
            rois_i = bbox2roi([res_i.bboxes for res_i in sampling_results_i])
            bbox_feats_i = self.bbox_roi_extractor_i(
                x_i[:self.bbox_roi_extractor_i.num_inputs], rois_i)

            if self.with_shared_head_i:
                bbox_feats_i = self.shared_head_i(bbox_feats_i)
            cls_score_i, bbox_pred_i = self.bbox_head_i(bbox_feats_i)
            rbbox_targets_i = self.bbox_head_i.get_target(
                sampling_results_i, gt_obbs_i, obb_u_i, gt_labels_i, self.train_cfg.rcnn[0])

            loss_bbox_i = self.bbox_head_i.loss_i(cls_score_i, bbox_pred_i,
                                            *rbbox_targets_i)
            for name_i, value_i in loss_bbox_i.items():
                losses['s{}.{}'.format(0, name_i)] = (value_i)

        pos_is_gts_i = [res_i.pos_is_gt for res_i in sampling_results_i]
        roi_labels_i = rbbox_targets_i[0]
        with torch.no_grad():
            rotated_proposal_list_i = self.bbox_head_i.refine_rbboxes(
                roi2droi(rois_i), roi_labels_i, bbox_pred_i, pos_is_gts_i, img_meta_i
            )

        if self.with_bbox_f:
            rois_f = bbox2roi([res_f.bboxes for res_f in sampling_results_f])
            bbox_feats_f = self.bbox_roi_extractor_f(
                x_f[:self.bbox_roi_extractor_f.num_inputs], rois_f)

            if self.with_shared_head_f:
                bbox_feats_f = self.shared_head_f(bbox_feats_f)
            cls_score_f, bbox_pred_f = self.bbox_head_f(bbox_feats_f)
            rbbox_targets_f = self.bbox_head_f.get_target(
                sampling_results_f, gt_obbs_i, obb_u_i, gt_labels_i, self.train_cfg.rcnn[0])

            loss_bbox_f = self.bbox_head_f.loss_f(cls_score_f, bbox_pred_f,
                                            *rbbox_targets_f)
            for name_f, value_f in loss_bbox_f.items():
                losses['s{}.{}'.format(0, name_f)] = (value_f)

        pos_is_gts_f = [res_f.pos_is_gt for res_f in sampling_results_f]
        roi_labels_f = rbbox_targets_f[0]
        with torch.no_grad():
            rotated_proposal_list_f = self.bbox_head_f.refine_rbboxes(
                roi2droi(rois_f), roi_labels_f, bbox_pred_f, pos_is_gts_f, img_meta_i
            )

        if self.with_rbbox_r:
            bbox_assigner_r = build_assigner(self.train_cfg.rcnn[1].assigner)
            bbox_sampler_r = build_sampler(
                self.train_cfg.rcnn[1].sampler, context=self)
            num_imgs_r = img_r.size(0)
            if gt_bboxes_ignore_r is None:
                gt_bboxes_ignore_r = [None for _ in range(num_imgs_r)]
            sampling_results_r = []
            for i in range(num_imgs_r):
                gt_obbs_best_roi_r = choose_best_Rroi_batch(gt_obbs_r[i])
                assign_result_r = bbox_assigner_r.assign(
                    rotated_proposal_list_r[i], gt_obbs_best_roi_r, gt_bboxes_ignore_r[i],
                    gt_labels_r[i]) 
                sampling_result_r = bbox_sampler_r.sample(
                    assign_result_r,
                    rotated_proposal_list_r[i],                                               
                    torch.from_numpy(gt_obbs_best_roi_r).float().to(rotated_proposal_list_r[i].device),
                    gt_labels_r[i], 
                    feats_r=[lvl_feat_r[i][None] for lvl_feat_r in x_r])
                sampling_results_r.append(sampling_result_r)

        if self.with_rbbox_i:
            bbox_assigner_i = build_assigner(self.train_cfg.rcnn[1].assigner)
            bbox_sampler_i = build_sampler(
                self.train_cfg.rcnn[1].sampler, context=self)
            num_imgs_i = img_i.size(0)
            if gt_bboxes_ignore_i is None:
                gt_bboxes_ignore_i = [None for _ in range(num_imgs_i)]
            sampling_results_i = []
            for i in range(num_imgs_i):
                gt_obbs_best_roi_i = choose_best_Rroi_batch(gt_obbs_i[i])
                assign_result_i = bbox_assigner_i.assign(
                    rotated_proposal_list_i[i], gt_obbs_best_roi_i, gt_bboxes_ignore_i[i],
                    gt_labels_i[i])
                sampling_result_i = bbox_sampler_i.sample(
                    assign_result_i,
                    rotated_proposal_list_i[i],                                                      
                    torch.from_numpy(gt_obbs_best_roi_i).float().to(rotated_proposal_list_i[i].device),
                    gt_labels_i[i],
                    feats_i=[lvl_feat_i[i][None] for lvl_feat_i in x_i])
                sampling_results_i.append(sampling_result_i)

        if self.with_rbbox_f:
            bbox_assigner_f = build_assigner(self.train_cfg.rcnn[1].assigner)
            bbox_sampler_f = build_sampler(
                self.train_cfg.rcnn[1].sampler, context=self)
            num_imgs_f = img_i.size(0)
            if gt_bboxes_ignore_i is None:
                gt_bboxes_ignore_i = [None for _ in range(num_imgs_f)]
            sampling_results_f = []
            for i in range(num_imgs_f):
                gt_obbs_best_roi_f = choose_best_Rroi_batch(gt_obbs_i[i])
                assign_result_f = bbox_assigner_f.assign(
                    rotated_proposal_list_f[i], gt_obbs_best_roi_f, gt_bboxes_ignore_i[i],
                    gt_labels_i[i])
                sampling_result_f = bbox_sampler_f.sample(
                    assign_result_f,
                    rotated_proposal_list_f[i],                                                         
                    torch.from_numpy(gt_obbs_best_roi_f).float().to(rotated_proposal_list_f[i].device),
                    gt_labels_i[i],
                    feats_f=[lvl_feat_f[i][None] for lvl_feat_f in x_f])
                sampling_results_f.append(sampling_result_f)

        if self.with_rbbox_r:
            rrois_r = dbbox2roi([res_r.bboxes for res_r in sampling_results_r])
            rrois_r[:, 3] = rrois_r[:, 3] * self.rbbox_roi_extractor_r.w_enlarge
            rrois_r[:, 4] = rrois_r[:, 4] * self.rbbox_roi_extractor_r.h_enlarge
            rbbox_feats_r = self.rbbox_roi_extractor_r(x_r[:self.rbbox_roi_extractor_r.num_inputs],
                                                   rrois_r)
            if self.with_shared_head_rbbox_r:
                rbbox_feats_r = self.shared_head_rbbox_r(rbbox_feats_r)
            cls_score_r, rbbox_pred_r = self.rbbox_head_r(rbbox_feats_r)
            rbbox_targets_r = self.rbbox_head_r.get_target_rbbox(sampling_results_r, gt_obbs_r,
                                                        gt_labels_r, obb_u_r, self.train_cfg.rcnn[1])
            loss_rbbox_r = self.rbbox_head_r.loss_r(cls_score_r, rbbox_pred_r, *rbbox_targets_r)
            for name_r, value_r in loss_rbbox_r.items():
                losses['s{}.{}'.format(1, name_r)] = (value_r)

        if self.with_rbbox_i:
            rrois_i = dbbox2roi([res_i.bboxes for res_i in sampling_results_i])
            rrois_i[:, 3] = rrois_i[:, 3] * self.rbbox_roi_extractor_i.w_enlarge
            rrois_i[:, 4] = rrois_i[:, 4] * self.rbbox_roi_extractor_i.h_enlarge
            rbbox_feats_i = self.rbbox_roi_extractor_i(x_i[:self.rbbox_roi_extractor_i.num_inputs],
                                                   rrois_i)
            if self.with_shared_head_rbbox_i:
                rbbox_feats_i = self.shared_head_rbbox_i(rbbox_feats_i)
            cls_score_i, rbbox_pred_i = self.rbbox_head_i(rbbox_feats_i)
            rbbox_targets_i = self.rbbox_head_i.get_target_rbbox(sampling_results_i, gt_obbs_i,
                                                        gt_labels_i, obb_u_i, self.train_cfg.rcnn[1])
            loss_rbbox_i = self.rbbox_head_i.loss_i(cls_score_i, rbbox_pred_i, *rbbox_targets_i)
            for name_i, value_i in loss_rbbox_i.items():
                losses['s{}.{}'.format(1, name_i)] = (value_i)

        if self.with_rbbox_f:
            rrois_f = dbbox2roi([res_f.bboxes for res_f in sampling_results_f])
            rrois_f[:, 3] = rrois_f[:, 3] * self.rbbox_roi_extractor_f.w_enlarge
            rrois_f[:, 4] = rrois_f[:, 4] * self.rbbox_roi_extractor_f.h_enlarge
            rbbox_feats_f = self.rbbox_roi_extractor_f(x_f[:self.rbbox_roi_extractor_f.num_inputs],
                                                   rrois_f)
            if self.with_shared_head_rbbox_f:
                rbbox_feats_f = self.shared_head_rbbox_f(rbbox_feats_f)
            cls_score_f, rbbox_pred_f = self.rbbox_head_f(rbbox_feats_f)
            rbbox_targets_f = self.rbbox_head_f.get_target_rbbox(sampling_results_f, gt_obbs_i,
                                                        gt_labels_i, obb_u_i, self.train_cfg.rcnn[1])
            loss_rbbox_f = self.rbbox_head_f.loss_f(cls_score_f, rbbox_pred_f, *rbbox_targets_f)
            for name_f, value_f in loss_rbbox_f.items():
                losses['s{}.{}'.format(1, name_f)] = (value_f)
        return losses

    def simple_test(self, img_r, img_i, img_meta_r, img_meta_i, rgb_dark, proposals=None, rescale=False):
        x_r = self.extract_feat_rgb(img_r)
        x_i = self.extract_feat_infrared(img_i)    
        x_f = []
        for i in range(len(x_r)):
            concat = torch.cat((x_r[i], x_i[i]), dim=1)
            concat = self.fusion_conv(concat)
            x_f.append(concat)
        x_f = tuple(x_f)

        proposal_list_r = self.simple_test_rpn_r(
            x_r, img_meta_r, self.test_cfg.rpn) if proposals is None else proposals
        proposal_list_i = self.simple_test_rpn_i(
            x_i, img_meta_i, self.test_cfg.rpn) if proposals is None else proposals
        proposal_list_f = self.simple_test_rpn_f(
            x_f, img_meta_r, self.test_cfg.rpn) if proposals is None else proposals 

        img_shape_r = img_meta_r[0]['img_shape']
        scale_factor_r = img_meta_r[0]['scale_factor']
        img_shape_i = img_meta_i[0]['img_shape']
        scale_factor_i = img_meta_i[0]['scale_factor']

        rcnn_test_cfg = self.test_cfg.rcnn

        rois_r = bbox2roi(proposal_list_r)          # rgb
        bbox_feats_r = self.bbox_roi_extractor_r(
            x_r[:len(self.bbox_roi_extractor_r.featmap_strides)], rois_r)
        if self.with_shared_head_r:
            bbox_feats_r = self.shared_head_r(bbox_feats_r)
        rois_i = bbox2roi(proposal_list_i)              # infrared
        bbox_feats_i = self.bbox_roi_extractor_i(
            x_i[:len(self.bbox_roi_extractor_i.featmap_strides)], rois_i)
        if self.with_shared_head_i:
            bbox_feats_i = self.shared_head_i(bbox_feats_i)
        rois_f = bbox2roi(proposal_list_f)             # fusion
        bbox_feats_f = self.bbox_roi_extractor_f(
            x_f[:len(self.bbox_roi_extractor_f.featmap_strides)], rois_f)
        if self.with_shared_head_f:
            bbox_feats_f = self.shared_head_f(bbox_feats_f)

        cls_score_r, bbox_pred_r = self.bbox_head_r(bbox_feats_r)
        cls_score_i, bbox_pred_i = self.bbox_head_i(bbox_feats_i)
        cls_score_f, bbox_pred_f = self.bbox_head_f(bbox_feats_f)

        bbox_label_r = cls_score_r.argmax(dim=1)
        bbox_label_i = cls_score_i.argmax(dim=1)
        bbox_label_f = cls_score_f.argmax(dim=1)

        rrois_r = self.bbox_head_r.regress_by_class_rbbox(roi2droi(rois_r), bbox_label_r, bbox_pred_r,
                                                      img_meta_r[0])
        rrois_i = self.bbox_head_i.regress_by_class_rbbox(roi2droi(rois_i), bbox_label_i, bbox_pred_i,
                                                      img_meta_i[0])
        rrois_f = self.bbox_head_f.regress_by_class_rbbox(roi2droi(rois_f), bbox_label_f, bbox_pred_f,
                                                      img_meta_r[0])
        # rgb
        rrois_enlarge_r = copy.deepcopy(rrois_r)
        rrois_enlarge_r[:, 3] = rrois_enlarge_r[:, 3] * self.rbbox_roi_extractor_r.w_enlarge
        rrois_enlarge_r[:, 4] = rrois_enlarge_r[:, 4] * self.rbbox_roi_extractor_r.h_enlarge
        rbbox_feats_r = self.rbbox_roi_extractor_r(
            x_r[:len(self.rbbox_roi_extractor_r.featmap_strides)], rrois_enlarge_r)
        if self.with_shared_head_rbbox_r:
            rbbox_feats_r = self.shared_head_rbbox_r(rbbox_feats_r)

        # infrared
        rrois_enlarge_i = copy.deepcopy(rrois_i)
        rrois_enlarge_i[:, 3] = rrois_enlarge_i[:, 3] * self.rbbox_roi_extractor_i.w_enlarge
        rrois_enlarge_i[:, 4] = rrois_enlarge_i[:, 4] * self.rbbox_roi_extractor_i.h_enlarge
        rbbox_feats_i = self.rbbox_roi_extractor_i(
            x_i[:len(self.rbbox_roi_extractor_i.featmap_strides)], rrois_enlarge_i)
        if self.with_shared_head_rbbox_i:
            rbbox_feats_i = self.shared_head_rbbox_i(rbbox_feats_i)

        # fusion
        rrois_enlarge_f = copy.deepcopy(rrois_f)
        rrois_enlarge_f[:, 3] = rrois_enlarge_f[:, 3] * self.rbbox_roi_extractor_f.w_enlarge
        rrois_enlarge_f[:, 4] = rrois_enlarge_f[:, 4] * self.rbbox_roi_extractor_f.h_enlarge
        rbbox_feats_f = self.rbbox_roi_extractor_f(
            x_f[:len(self.rbbox_roi_extractor_f.featmap_strides)], rrois_enlarge_f)
        if self.with_shared_head_rbbox_f:
            rbbox_feats_f = self.shared_head_rbbox_f(rbbox_feats_f)

        rcls_score_r, rbbox_pred_r = self.rbbox_head_r(rbbox_feats_r)
        rcls_score_i, rbbox_pred_i = self.rbbox_head_i(rbbox_feats_i)
        rcls_score_f, rbbox_pred_f = self.rbbox_head_f(rbbox_feats_f)

        det_rbboxes_fusion, det_labels_fusion = self.rbbox_head_f.get_fusion_det_rbboxes(
            rrois_r,
            rcls_score_r,
            rbbox_pred_r,
            img_shape_r,
            scale_factor_r,
            rrois_i,
            rcls_score_i,
            rbbox_pred_i,
            img_shape_i,
            scale_factor_i,
            rrois_f,
            rcls_score_f,
            rbbox_pred_f,
            rgb_dark,
            rescale=rescale,
            cfg=rcnn_test_cfg)

        rbbox_results_fusion = dbbox2result(det_rbboxes_fusion, det_labels_fusion,
                                     self.rbbox_head_f.num_classes)

        return rbbox_results_fusion
