# Reference: https://github.com/RocketFlash/CAP_augmentation

import cv2
import numpy as np
import random
from glob import glob
import random

# how to use: cap_aug(p=0.5, n_objects_range=[1, 3], glob_split='_', retry_iters=30, min_inter_area=10, glob_suffix='*.png')()
# the pasted objects(images) glob path will be PATH_ROOT + CLS + glob_split + OBJ + glob_suffix

PATH_ROOT = r'/path/to/pasted objects/'
CLS = ['car', 'person', 'bus', 'motorbike', 'bicycle']
PROB_CLS = [0.0, 0.1, 0.3, 0.3, 0.3]
OBJ = ['small', 'medium', 'large']
PROB_OBJ = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1.0, 0.0, 0.0]]
# so, the file directorys of pasted objects are: /path/to/pasted objects/(car|person|bus|motorbike|bicycle)_(small|medium|large)*.png (using as glob path)


def resize_keep_ar(image, height=500, scale=None):
    "image:HWC"
    if scale is not None:
        image = cv2.resize(image, None, fx=float(scale), fy=float(scale))
    else:
        r = height / float(image.shape[0])
        width = r * image.shape[1]
        width = 1 if int(width) == 0 else int(width)
        image = cv2.resize(image, (int(width), int(height)))
    return image


class CAP_AUG_Multiclass(object):
    '''
    cap_augs - list of cap augmentations for each class
    probabilities - list of probabilities for each augmentation
    class_idxs - class indexes
    '''

    def __init__(self, cap_augs, probabilities, p=0.5):
        self.cap_augs = cap_augs
        self.probabilities = probabilities
        self.p = p
        assert len(self.cap_augs) == len(self.probabilities)

    def __call__(self, image, annots=None):
        if random.uniform(0, 1) > self.p:
            return image, annots
        return self.generate_objects(image, annots)

    def generate_objects(self, image, annots=None):
        result_image = image.copy()
        total_result_coords = []
        total_semantic_masks = []
        total_instance_masks = []

        for cap_aug, p in zip(self.cap_augs, self.probabilities):
            # return image_dst, {'coords_all': coords_all, 'semantic_mask': semantic_mask, 'instance_mask': instance_mask}
            if p >= np.random.uniform(0, 1, size=1):
                result_image, result_dict = cap_aug(result_image, annots)
                # result_image, result_coords, semantic_mask, instance_mask =
                if annots is None:
                    total_result_coords.append(result_dict['coords_all'])
                    total_semantic_masks.append(result_dict['semantic_mask'])
                    total_instance_masks.append(result_dict['instance_mask'])
                else:
                    # print(result_dict)
                    annots['bbox'] = np.vstack((result_dict['bbox'], annots['bbox']))
                    annots['cls'] = np.hstack((np.reshape(result_dict['cls'], (-1)), annots['cls']))
                    # print(annots)
        if annots is None:
            if len(total_result_coords) > 0:
                total_result_coords = np.vstack(total_result_coords)

            return result_image, {'total_result_coords': total_result_coords, 'total_semantic_masks': total_semantic_masks, 'total_instance_masks': total_instance_masks}
        else:
            return result_image, annots


class CAP_AUG(object):
    '''
    source_images - list of images paths
    bev_transform - bird's eye view transformation
    probability_map - mask with probability values
    mean_h_norm - mean normilized height
    n_objects_range - [min, max] number of objects
    s_range - range of scales of original image size
    h_range - range of objects heights
              if bev_transform is not None range in meters, else in pixels
    x_range - if bev_transform is None -> range in the image coordinate system (in pixels) [int, int]
              else                     -> range in camera coordinate system (in meters) [float, float]
    y_range - if bev_transform is None -> range in the image coordinate system (in pixels) [int, int]
              else                     -> range in camera coordinate system (in meters) [float, float]
    z_range - if bev_transform is None -> range in the image coordinate system (in pixels) [int, int]
              else                     -> range in camera coordinate system (in meters) [float, float]
    objects_idxs - objects indexes from dataset to paste [idx1, idx2, ...]
    random_h_flip - source image random horizontal flip
    random_v_flip - source image random vertical flip
    histogram_matching - apply histogram matching
    hm_offset - histogram matching offset
    blending_coeff - coefficient of image blending
    image_format - color image format : {bgr, rgb}
    coords_format - output coordinates format: {xyxy, xywh, yolo}
    normilized_range - range in normilized image coordinates (all values are in range [0, 1])
    class_idx - class id to result bounding boxes, output bboxes will be in [x1, y1, x2, y2, class_idx] format
    albu_transforms - albumentations transformations applied to pasted objects 
    '''

    def __init__(self, source_images,
                 retry_iters=50,
                 min_inter_area=10,
                 n_objects_range=[1, 3],
                 h_range=None,
                 z_range=None,
                 y_range=None,
                 s_range=None,
                 x_range=None,
                 #  s_range=[0.5, 1.5],
                 #  x_range=[200, 500],
                 #  y_range=[100, 300],
                 #  z_range=[0, 0],
                 objects_idxs=None,
                 random_h_flip=True,
                 random_v_flip=False,
                 image_format='bgr',
                 coords_format='xyxy',
                 class_idx=None,
                 albu_transforms=None):

        if isinstance(source_images, str):
            self.source_images = glob(source_images)
        else:  # list
            self.source_images = source_images
        self.retry_iters = retry_iters
        self.min_inter_area = min_inter_area
        self.bboxes = []
        self.n_objects_range = n_objects_range
        self.s_range = s_range
        self.h_range = h_range
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.objects_idxs = objects_idxs
        self.random_h_flip = random_h_flip
        self.random_v_flip = random_v_flip
        self.image_format = image_format
        self.coords_format = coords_format
        self.class_idx = class_idx
        self.albu_transforms = albu_transforms

    def __call__(self, image, annots=None):
        # img HWC
        # ann['bbox'] N*4 xyxy
        # ann['cls'] 1*N
        if annots is not None:
            # self.bboxes = np.hstack((annots['bbox'][:, [1, 0, 3, 2]], np.reshape(annots['cls'], (-1, 1))))
            self.bboxes = list(annots['bbox'][:, [1, 0, 3, 2]])
        h, w, _ = image.shape
        self.h_range = [min(16, h), min(64, h)]
        # self.s_range = [0.5, 1.5]
        self.x_range = [0, w]
        self.y_range = [0, h]
        image_dst, coords_all, semantic_mask, instance_mask = self.generate_objects(image)
        if annots is None:
            return image_dst, {'coords_all': coords_all, 'semantic_mask': semantic_mask, 'instance_mask': instance_mask}
        else:
            # print('corrds_all:'+str(coords_all))
            bboxtmp, clstmp = np.hsplit(np.array(coords_all), [4])
            # annots['bbox'] = np.vstack((bboxtmp[:, [1, 0, 3, 2]], annots['bbox']))
            # annots['cls'] = np.vstack((np.reshape(clstmp, (-1)), annots['cls']))
            ann = dict()
            ann['bbox'] = bboxtmp[:, [1, 0, 3, 2]]
            ann['cls'] = np.reshape(clstmp, (-1))
            # print('ann:'+str(ann))
            return image_dst, ann

    def select_image(self, object_idx):
        source_image_path = self.source_images[object_idx]
        image_src = cv2.imread(str(source_image_path), cv2.IMREAD_UNCHANGED)
        if image_src.shape[2] == 4:
            if self.image_format == 'rgb':
                image_src = cv2.cvtColor(image_src, cv2.COLOR_BGRA2RGBA)
            return image_src
        if self.image_format == 'rgb':
            image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGBA)
        else:
            image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2BGRA)
        return image_src

    def check_bbox_no_overlap(self, x1, y1, x2, y2, coords_all):
        for bbox in (self.bboxes+coords_all):
            x3, y3, x4, y4 = bbox
            left_max = max(x1, x3)
            top_max = max(y1, y3)
            right_min = min(x2, x4)
            bottom_min = min(y2, y4)
            inter = max(0, (right_min-left_max)) * max(0, (bottom_min-top_max))
            if inter > self.min_inter_area:
                return False
        return True

    def generate_objects(self, image):

        n_objects = random.randint(*self.n_objects_range)
        if self.objects_idxs is None:
            objects_idxs = [random.randint(0, len(self.source_images)-1) for _ in range(n_objects)]
        else:
            objects_idxs = self.objects_idxs

        image_dst = image.copy()
        dst_h, dst_w, _ = image_dst.shape

        coords_all = []
        semantic_mask = np.zeros((dst_h, dst_w), dtype=np.uint8)
        instance_mask = np.zeros((dst_h, dst_w), dtype=np.uint8)

        for i in range(n_objects):
            src_img = self.select_image(objects_idxs[i])
            h, w, _ = src_img.shape
            for _ in range(self.retry_iters):
                point = np.random.randint(low=[self.x_range[0], self.y_range[0]],
                                          high=[self.x_range[1], self.y_range[1]],
                                          size=(2))
                height = scale = None

                if self.s_range is not None:
                    scale = random.uniform(*self.s_range)
                    # w = round(w * scale)
                    # h = round(h * scale)
                elif self.h_range is not None:
                    height = random.randint(*self.h_range)
                    # r = height / float(h)
                    # h = height
                    # w = r * w
                else:
                    print("s_range and h_range is both None.")
                src_img = resize_keep_ar(src_img, height=height, scale=scale)
                h, w, _ = src_img.shape
                if w <= 0 or h <= 0 or w >= dst_w or h >= dst_h:
                    continue
                x1, x2 = point[0]-w, point[0]
                y1, y2 = point[1]-h, point[1]
                # print(f'{h} {w} {x1} {y1} {x2} {y2}')

                if x1 < 0 or y1 < 0:
                    continue
                if not self.check_bbox_no_overlap(x1, y1, x2, y2, coords_all):
                    continue
                # print(f'{h} {w} {x1} {y1} {x2} {y2}')
                image_dst, mask = self.paste_object(image_dst, src_img, x1, y1, x2, y2)
                curr_mask = mask/255
                curr_mask = curr_mask.astype(np.uint8)
                curr_mask_ins = curr_mask*(i+1)

                roi_mask_sem = semantic_mask[y1:y2, x1:x2]
                roi_mask_ins = instance_mask[y1:y2, x1:x2]

                mask_inv = cv2.bitwise_not(curr_mask*255)

                img_sem_bg = cv2.bitwise_and(roi_mask_sem, roi_mask_sem, mask=mask_inv)
                img_ins_bg = cv2.bitwise_and(roi_mask_ins, roi_mask_ins, mask=mask_inv)

                dst_sem = cv2.add(img_sem_bg, curr_mask)
                dst_ins = cv2.add(img_ins_bg, curr_mask_ins)

                semantic_mask[y1:y2, x1:x2] = dst_sem
                instance_mask[y1:y2, x1:x2] = dst_ins
                coords_all.append([x1, y1, x2, y2])
                break
        coords_all = np.array(coords_all)

        if self.coords_format == 'yolo':
            x = coords_all.copy()
            x = x.astype(float)
            dw = 1./dst_w
            dh = 1./dst_h
            ws = (coords_all[:, 2] - coords_all[:, 0])
            hs = (coords_all[:, 3] - coords_all[:, 1])
            x[:, 0] = dw * ((coords_all[:, 0] + ws/2.0)-1)
            x[:, 1] = dh * ((coords_all[:, 1] + hs/2.0)-1)
            x[:, 2] = dw * ws
            x[:, 3] = dh * hs
            coords_all = x
        elif self.coords_format == 'xywh':
            x = coords_all.copy()
            x[:, 2] = coords_all[:, 2] - coords_all[:, 0]
            x[:, 3] = coords_all[:, 3] - coords_all[:, 1]
            coords_all = x

        if self.class_idx is not None:
            coords_all = np.c_[coords_all, self.class_idx*np.ones(len(coords_all))]
        # print(coords_all)
        return image_dst, coords_all, semantic_mask, instance_mask

    def paste_object(self, image_dst, image_src, x1, y1, x2, y2):
        src_h, src_w, _ = image_src.shape
        y1_m, y2_m = 0, src_h
        x1_m, x2_m = 0, src_w
        if self.random_h_flip:
            if random.uniform(0, 1) > 0.5:
                image_src = cv2.flip(image_src, 1)

        if self.random_v_flip:
            if random.uniform(0, 1) > 0.5:
                image_src = cv2.flip(image_src, 0)

        # Simple cut and paste without preprocessing
        mask_src = image_src[:, :, 3]
        rgb_img = image_src[:, :, :3]

        # can't resize. make sure the all image is still the object to be pasted
        if self.albu_transforms is not None:
            transformed = self.albu_transforms(image=rgb_img, mask=mask_src)
            rgb_img = transformed['image']
            mask_src = transformed['mask']

        mask_inv = cv2.bitwise_not(mask_src)
        # print(f'{src_h} {src_w} {x1} {y1} {x2} {y2}')
        # print(image_dst[y1:y2, x1:x2].shape)
        # print(image_dst.shape)
        # print(mask_inv[y1_m:y2_m, x1_m:x2_m].shape)
        # print(type(mask_inv[y1_m:y2_m, x1_m:x2_m][0, 0]))
        img1_bg = cv2.bitwise_and(image_dst[y1:y2, x1:x2], image_dst[y1:y2,
                                                                     x1:x2], mask=mask_inv[y1_m:y2_m, x1_m:x2_m])
        img2_fg = cv2.bitwise_and(rgb_img[y1_m:y2_m, x1_m:x2_m], rgb_img[y1_m:y2_m,
                                                                         x1_m:x2_m], mask=mask_src[y1_m:y2_m, x1_m:x2_m])
        out_img = cv2.add(img1_bg, img2_fg)
        mask_visible = mask_src[y1_m:y2_m, x1_m:x2_m]
        image_dst[y1:y2, x1:x2] = out_img
        return image_dst, mask_visible


def cap_aug(p=0.5, n_objects_range=[1, 3], glob_split='_', retry_iters=30, min_inter_area=10, glob_suffix='*.png'):
    '''
    return a instance of class:CAP_AUG_Multiclass
    '''
    # get presum of prob
    prob_cls_presum_value = 0
    prob_cls_presum = []
    for prob in PROB_CLS:
        prob_cls_presum_value += prob
        prob_cls_presum.append(prob_cls_presum_value)
    prob_obj_presum = []
    for cls in PROB_OBJ:
        prob_obj_presum_value = 0
        prob_obj_presum_list = []
        for prob in cls:
            prob_obj_presum_value += prob
            prob_obj_presum_list.append(prob_obj_presum_value)
        prob_obj_presum.append(prob_obj_presum_list)

    n_objects = random.randint(*n_objects_range)
    # get prefix of source images path.
    # a list of 2nd-level-prefix(cls+obj). len:n_objects
    path_prefix = {}
    cap_augs = []
    for _ in range(n_objects):
        prefix = PATH_ROOT
        prob_cls_idx_value = random.uniform(0, 1)
        for i in range(len(CLS)):
            if prob_cls_idx_value < prob_cls_presum[i]:
                # cls_idx.append(i)
                prefix += CLS[i]
                prob_obj_idx_value = random.uniform(0, 1)
                for j in range(len(OBJ)):
                    if prob_obj_idx_value < prob_obj_presum[i][j]:
                        prefix += glob_split + OBJ[j]
                        if prefix in path_prefix.keys():
                            path_prefix[prefix][0] += 1
                        else:
                            path_prefix[prefix] = [1, i+1]
                        break
                break

    for glob_prefix, [num, idx] in path_prefix.items():
        cap_augs.append(CAP_AUG(source_images=glob_prefix+glob_suffix,
                                retry_iters=retry_iters,
                                min_inter_area=min_inter_area,
                                s_range=[0.5, 1.5],
                                n_objects_range=[num, num],
                                random_h_flip=True,
                                image_format='bgr',
                                coords_format='xyxy',
                                class_idx=idx,
                                albu_transforms=None))
    return CAP_AUG_Multiclass(cap_augs=cap_augs,
                              probabilities=[1]*len(cap_augs),
                              p=p)


# cap_aug()
