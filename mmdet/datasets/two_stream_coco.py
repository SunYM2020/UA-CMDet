import numpy as np
from pycocotools.coco import COCO

from .two_stream_custom import TSCustomDataset 


class TSCocoDataset(TSCustomDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def load_annotations_r(self, ann_file):
        self.coco_r = COCO(ann_file)
        self.cat_ids_r = self.coco_r.getCatIds()
        self.cat2label_r = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids_r)
        }
        self.img_ids_r = self.coco_r.getImgIds()
        img_infos_r = []
        for i in self.img_ids_r:
            info_r = self.coco_r.loadImgs([i])[0]
            info_r['filename'] = info_r['file_name']
            img_infos_r.append(info_r)
        #print(ann_file)
        return img_infos_r

    def load_annotations_i(self, ann_file):
        self.coco_i = COCO(ann_file)
        self.cat_ids_i = self.coco_i.getCatIds()
        self.cat2label_i = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids_i)
        }
        self.img_ids_i = self.coco_i.getImgIds()
        img_infos_i = []
        for i in self.img_ids_i:
            info_i = self.coco_i.loadImgs([i])[0]
            info_i['filename'] = info_i['file_name']
            img_infos_i.append(info_i)
        #print(ann_file)
        return img_infos_i

    def get_ann_info_r(self, idx):
        img_id_r = self.img_infos_r[idx]['id']           # rgb
        ann_ids_r = self.coco_r.getAnnIds(imgIds=[img_id_r])
        ann_info_r = self.coco_r.loadAnns(ann_ids_r)
        #print(ann_info_r)
        return self._parse_ann_info_r(ann_info_r, self.with_mask)

    def get_ann_info_i(self, idx):
        img_id_i = self.img_infos_i[idx]['id']           # infrared
        ann_ids_i = self.coco_i.getAnnIds(imgIds=[img_id_i])
        ann_info_i = self.coco_i.loadAnns(ann_ids_i)
        #print(ann_info_i)
        return self._parse_ann_info_i(ann_info_i, self.with_mask)

    def _filter_imgs_r(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds_r = []
        ids_with_ann_r = set(_['image_id'] for _ in self.coco_r.anns.values())
        for i, img_info in enumerate(self.img_infos_r):            # rgb
            if self.img_ids_r[i] not in ids_with_ann_r:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds_r.append(i)
        print('RGB:',len(valid_inds_r))
        return valid_inds_r

    def _filter_imgs_i(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds_i = []
        ids_with_ann_i = set(_['image_id'] for _ in self.coco_i.anns.values())
        for i, img_info in enumerate(self.img_infos_i):            # infrared
            if self.img_ids_i[i] not in ids_with_ann_i:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds_i.append(i)
        print('tir:',len(valid_inds_i))
        return valid_inds_i

    def _parse_ann_info_r(self, ann_info, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label_r[ann['category_id']])
            if with_mask:
                gt_masks.append(self.coco_r.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann

    def _parse_ann_info_i(self, ann_info, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label_i[ann['category_id']])
            if with_mask:
                gt_masks.append(self.coco_i.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann
