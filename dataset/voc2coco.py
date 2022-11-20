import shutil
import random
import json
import cv2
import os
import xml.etree.ElementTree as ET
from os.path import exists
def get(root, name):
    return root.findall(name)

def get_and_check(root, name, length):
    vars = root.findall(name)
    
    if len(vars) == 0:
        raise NotImplementedError('Can not find {} in {}.'.format(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of {} is supposed to be {},but is {}.'.format(name, length, len(vars)))
    if length == 1:
        vars = vars[0]

    return vars

def transfer_xml_to_annos(xmlPath, saveDir, imageDir, classes):
    n = 1
    for xml in xmlPath:
        tree = ET.parse(xml)
        root = tree.getroot()
        # 圖片名稱
        filename = get_and_check(root, 'filename', 1).text
        #print('filename=',filename)
        # 圖片長寬
        im = cv2.imread(imageDir+filename)
        #print(im.shape) #((height),(width),(3))
        width = im.shape[1]
        height = im.shape[0]
        #print(filename+" width=",width)
        #print(filename+" height=",height)
        
        # 處理每個標註的檢測框
        with open(saveDir, "a") as bbox:
            #從<xml> object 開始搜尋
            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text
                #print(category) # <name>licence</name> 
                #label_index = str(classes.index(category) + 1)
                label_index = str(classes.index(category))
            
                bndbox = get_and_check(obj, 'bndbox', 1)
                #xmin = (int(get_and_check(bndbox, 'xmin', 1).text) - 1) / width
                #ymin = (int(get_and_check(bndbox, 'ymin', 1).text) - 1) / height
                xmin = (int(get_and_check(bndbox, 'xmin', 1).text)) / width
                ymin = (int(get_and_check(bndbox, 'ymin', 1).text)) / height
                xmax = (int(get_and_check(bndbox, 'xmax', 1).text)) / width
                ymax = (int(get_and_check(bndbox, 'ymax', 1).text)) / height
                
                #print(filename+" width=",width)
                #print(filename+" height=",height)
                #print(filename+" xmin=",xmin)
                #print(filename+" ymin=",ymin)
                #print(filename+" xmax=",xmax)
                #print(filename+" ymax=",ymax)
                #exit(0)

                bbox.write(filename +' {} {} {} {} {}\n'.format(label_index, xmin, ymin, xmax, ymax))
                      
        #print('第{:3d}個xml檔案完成'.format(n))
        #print('剩{:3d}個需轉換'.format(len(xmlPath)-n))
        #print("-" * 35)
        n += 1
    #再依序讀出另存一個檔案    
    with open("./labels/annos.txt",'r') as data_file:
        for line in data_file:
            data = line.split()
            #print(data)
            label=data[0].split(".")
            labelsname = label[0]
            index = data[1]
            xmin =  data[2]
            ymin =  data[3]
            xmax =  data[4]
            ymax =  data[5]
            f = open("./labels/"+labelsname+".txt", "w")
            f.write('{} {} {} {} {}\n'.format(index,xmin,ymin,xmax,ymax))
            f.close()
            #print(labelsname,index,xmin,ymin,xmax,ymax)
    #print(n)        
        

# 將圖片依照比例分配train與val
def train_val_split(source, ratio):
    # 讀取images資料夾內圖片檔名
    indexes = os.listdir(os.path.join(source, 'images'))
    # 檔案順序隨機
    random.shuffle(indexes)
    # 創建訓練或驗證集(待優化，自動比例split)
    pic_num = len(indexes)
    train_num = int(pic_num * ratio)
    train_list = indexes[:train_num]
    val_list = indexes[train_num:]

    return train_list, val_list        

def transfer_and_save_coco(source, split_list, dataset, phase):
    # 紀錄處理的圖片數量
    count = 0
    # 讀取bbox信息
    with open(os.path.join(source, './labels/annos.txt')) as tr:
        annos = tr.readlines()
        # 轉換為coco格式
        for k, index in enumerate(split_list):
            count += 1
            # opencv讀取圖片，得到圖片寬、高
            im = cv2.imread(os.path.join(source, 'images/') + index)
            height, width, _ = im.shape

            # 將圖片檔名、index、寬高信息存入dataset
            dataset['images'].append({'file_name': index,
                                      'id': k,
                                      'width': width,
                                      'height': height})

            for i, anno in enumerate(annos):
                parts = anno.strip().split()

                # 如果圖片檔名與標籤名稱相同，則添加標籤
                if parts[0] == index:
                    # 類別
                    cls_id = parts[1]
                    # x_min
                    x1 = float(parts[2])
                    # y_min
                    y1 = float(parts[3])
                    # x_max
                    x2 = float(parts[4])
                    # y_max
                    y2 = float(parts[5])
                    width = max(0, x2 - x1)
                    height = max(0, y2 - y1)
                    dataset['annotations'].append({
                        'area': width * height,
                        'bbox': [x1, y1, width, height],
                        'category_id': int(cls_id),
                        'id': i,
                        'image_id': k,
                        'iscrowd': 0,
                        # 影像分割時使用，矩形是從左上角順時針畫4點(mask)
                        # 影像分割時'ignore':0與
                        # 'segmentation':[[x1,y1,x2,y1,x2,y2,x1,y2]]
                        'segmentation': []
                    })

            #print('   {} images handled'.format(count))

    # 儲存json檔
    folder = os.path.join(source, 'labels')
    if not os.path.exists(folder):
        os.makedirs(folder)
    json_name = os.path.join(source, 'labels/{}.json'.format(phase))
    with open(json_name, 'w') as f:
        json.dump(dataset, f)

# 生成train與val之coco格式json檔
def txt_to_coco_json(source, classes, split_list, phase):
    # dataset存放圖片信息和標籤(instances目標檢測、segementation影像分割)
    dataset = {'info': {'description': '', 'url': '', 'version': '1.0',
                        'year': 2022, 'contributor': 'Team5', 
                        'date_created': ''}, 
               'categories': [], 
               'annotations': [], 
               'images': [], 
               'type': 'instances'}

    # 建立標籤與id的對應關係
    for i, cls in enumerate(classes, 1):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'Team5'})

    # train, val資料轉換成coco格式，以json儲存
    print('開始轉換{}'.format(phase))
    transfer_and_save_coco(source, split_list, dataset, phase)
    print('{}.json Done'.format(phase))

# 移動圖片到train與val資料夾
def split_images_to_train_and_val(source, train_list, val_list):
    # 創建圖片train與val資料夾
    folder1 = os.path.join(source, 'train2022')
    if not os.path.exists(folder1):
        os.makedirs(folder1)
    folder2 = os.path.join(source, 'val2022')
    if not os.path.exists(folder2):
        os.makedirs(folder2)
    # 移動圖片到資料夾
    for move_it in train_list:
        shutil.move(source + '/images/' + move_it, 
                    os.path.join(source, 'train2022', ''))
    for move_it in val_list:
        shutil.move(source + '/images/' + move_it, 
                    os.path.join(source, 'val2022', ''))
    print('移動圖片到train與val資料夾 Done')

def main():
    source = '../train/'
    # 讀取標籤類別
    with open(os.path.join(source, 'classes.txt')) as f:
        classes = f.read().strip().split()
        
    # [Step1] : xml轉換為annos.txt：其中每行為imageName、classId、xMin、 yMim、xMax、yMax，一個bbox對應一行(coco格式的id編號從1起算)
    print('[Step1] xml轉annos.txt')
    
    # annos.txt存檔路徑
    saveDir = os.path.join(source, './labels/annos.txt')
    # image資料夾路徑
    imageDir = os.path.join(source, 'images/')
    # image檔案路徑
    imagePath = os.listdir(imageDir)
    imagePath = [imageDir + i for i in imagePath]
    # xml資料夾路徑
    xmlDir = os.path.join(source, 'annotations/')
    # xml檔案路徑
    xmlPath = os.listdir(xmlDir)
    xmlPath = [xmlDir + i for i in xmlPath]
    # 將xml轉換為annos
    transfer_xml_to_annos(xmlPath, saveDir, imageDir, classes)
    print('=' * 60)
    
    #[Step2]將標籤轉換成coco格式，並以json格式存檔。資料夾包含images(圖片資料夾)、annos.txt(bbox標記)、classes.txt(類別清單)及annotations(儲存json的資料夾)。
    #print('[Step2] annos.txt轉coco，並以json格式儲存')

    print("[Step2] 將圖片依照比例分配train與val")
    train_list, val_list = train_val_split(source, 0.8)

    # 生成train與val之coco格式json檔
    #txt_to_coco_json(source, classes, train_list, 'instances_train2022')
    #print('-' * 35)
    #txt_to_coco_json(source, classes, val_list, 'instances_val2022')
    #print('-' * 35)

    # 移動圖片到train與val資料夾
    split_images_to_train_and_val(source, train_list, val_list)
    print('程式執行結束')
    

if __name__ == '__main__':
    main()