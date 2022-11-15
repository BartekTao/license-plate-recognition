import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

# 要生成的文件夹
sets = [('2022', 'train'), ('2022', 'val')]

# 类别
classes = ["plate"]

annotationsPath = './coco128/annotations'
labelsPath = './coco128/lables'
imagesPath = './images'
annotationsPath = './coco128/annotations'

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(year, image_id,image_set):
    in_file = open('./coco128/annotations/%s.xml' % (year, image_id))
    out_file = open('./coco128/labels/%s/%s.txt' % (image_set,image_id ), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text),
             float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')



# wd = getcwd()

# 遍历sets = [('2012', 'train'), ('2012', 'val')]
for year, image_set in sets:
    # 在当前路径创建文件
    if not os.path.exists('./coco128/labels/'
                          '%s' % (image_set)):
        os.makedirs('./coco128/labels/%s' % (image_set))
    image_ids = open('F:/data/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'
                     % (year, image_set)).read().strip().split()
    if not os.path.exists('VOC2012/images/%s' % (image_set)):
        os.makedirs('VOC2012/images/%s' % (image_set))
    
    list_file = open('VOC2012/images/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        # 在images中创建一个txt写入每一个图片数据的绝对路径
        list_file.write('F:/data/VOCdevkit/VOC2012/JPEGImages/%s.jpg\n' % (image_id))
        convert_annotation(year, image_id,image_set)
    list_file.close()