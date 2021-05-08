from effdet.data.transforms import *
import numpy as np
from pycocotools.coco import COCO
import cv2
import matplotlib.pyplot as plt

img_size = 512
interpolation = 'random'
fill_color = 'mean'
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
mean = IMAGENET_DEFAULT_MEAN
fill_color = resolve_fill_color(fill_color, mean)

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White

# img_path = 'path/to/images/'
# ann_files = "path/to/annotations"
# coco = COCO(ann_files+'val.json')
# category_id_to_name = getCategoryIdToName(coco)


def getCategoryIdToName(coco):
    "return category_id_to_name:dict"
    catIds = coco.getCatIds()
    catInfos = coco.loadCats(catIds)
    category_id_to_name = dict()
    for catinfo in catInfos:
        category_id_to_name[catinfo['id']] = catinfo['name']
    return category_id_to_name


def visualize_bbox(img, bbox, class_name, score=None, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    text = class_name+' '+'%.2f' % (float(score)) if score is not None else class_name
    cv2.putText(
        img,
        text=text,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name, scores=None):
    "image:cv2.imread() bboxes:xywh"
    img = image.copy()
    fig = plt.figure(figsize=(16, 12))
    if scores is None:
        for bbox, category_id in zip(bboxes, category_ids):
            class_name = category_id_to_name[category_id]
            img = visualize_bbox(img, bbox, class_name)
    else:
        for bbox, category_id, score in zip(bboxes, category_ids, scores):
            class_name = category_id_to_name[category_id]
            img = visualize_bbox(img, bbox, class_name, score)
    # plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)


def visualizeFromAnnots(img, annots, category_id_to_name, toXywh=False, withScore=False):
    "image:cv2.imread() bboxes:xywh"
    bboxes, category_ids = annots['bbox'], annots['cls']
    if withScore:
        scores = annots['score']
    else:
        scores = None
    if toXywh:
        bboxes = bboxes[:, [1, 0, 3, 2]]
        bboxes[:, 2] -= bboxes[:, 0]
        bboxes[:, 3] -= bboxes[:, 1]
    visualize(img, bboxes, category_ids, category_id_to_name, scores)


def getAnnotFromImgIds(coco, imgIds, toYxyx=False):
    '''
    input : imgIds:[imgid...]
    return: annots:[annot...]
            annot:dict {'bbox':ndarray N*4,'cls',ndarray, N}
    '''
    annots = []
    for imgId in imgIds:
        annids = coco.getAnnIds(imgIds=[imgId])
        anninfos = coco.loadAnns(annids)
        bboxes = [anninfo['bbox'] for anninfo in anninfos]
        category_ids = [anninfo['category_id'] for anninfo in anninfos]
        annotations = {
            'bbox': np.array(bboxes, dtype='float64'),
            'cls': np.array(category_ids, dtype='int64')
        }
        if toYxyx:
            annotations['bbox'][:, 2] += annotations['bbox'][:, 0]
            annotations['bbox'][:, 3] += annotations['bbox'][:, 1]
            annotations['bbox'] = annotations['bbox'][:, [1, 0, 3, 2]]
        annots.append(annotations)
    return annots


def convertPilToCv(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
