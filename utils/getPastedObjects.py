# get pasted objects for copy-and-paste augmentation method
# objects type are among categoryNames defined below

from pycocotools.coco import COCO
import cv2
import math

ann_file_path = r"path/to/annotation.json"
img_path = r'path/to/images'
coco = COCO(ann_file_path)
save_path = r'path/to/cap_all_objects/'

# get category infos
categoryNames = ['car', 'person', 'bus', 'motorbike', 'bicycle']
categoryIds = coco.getCatIds(catNms=categoryNames)
categoryInfos = coco.loadCats(categoryIds)
categoryIdToName = {cat['id']: cat['name'] for cat in categoryInfos}
# {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 6: 'bus'}
categoryCount = {cat['id']: 1 for cat in categoryInfos}

# get all anninfos
annids = coco.getAnnIds(catIds=categoryIds)
anninfos = coco.loadAnns(annids)

# for every anninfo
for anninfo in anninfos:
    # get imginfo according to anninfo
    imgid = anninfo['image_id']
    imginfo = coco.loadImgs([imgid])[0]
    # download image if not exists
    imgname = imginfo['file_name']
    img = cv2.imread(img_path+imgname, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    # get mask according to segmentation annotation
    mask = coco.annToMask(anninfo)  # 2d array, size same as img
    assert img.shape[:2] == mask.shape
    img[:, :, 3] = mask*255
    x, y, w, h = anninfo['bbox']
    if w*h <= 32*32:
        sizetype = 'small'
    elif w*h <= 96*96:
        sizetype = 'medium'
    else:
        sizetype = 'large'
    x2, y2 = x+w, y+h
    x, y = list(map(math.floor, [x, y]))
    x2, y2 = list(map(math.ceil, [x2, y2]))
    catid = anninfo['category_id']
    # save new image
    save_name = save_path+categoryIdToName[catid]+'_'+sizetype+'_'+str(categoryCount[catid])+'.png'
    cv2.imwrite(save_name, img[y:y2, x:x2])
    categoryCount[catid] += 1
    if categoryCount[catid] % 1000 == 0:
        print(f'processed {categoryIdToName[catid]} num: {categoryCount[catid]}')
