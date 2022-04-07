import cv2
import numpy as np
import random
import os 

rectWidth = 224  # size
patchNumbers = 20  # number

def load_images_from_dir(dir_name, shuffle = False) :
    valid_image_extension = [".jpg", ".jpeg", ".png", ".tif", ".JPG",'.bmp']

    file_list = []
    nb_image = 0
    for filename in os.listdir(dir_name):
        extension = os.path.splitext(filename)[1]
        if extension.lower() in valid_image_extension:
            file_list.append(filename)
            nb_image += 1

    print('    ', nb_image, 'images loaded')

    if shuffle:
        random.shuffle(file_list)
    return file_list, nb_image

def cut_one_image_random(input_path, cut_savePath,full_savePath, file_name):
    img = cv2.imread(input_path + '/' + file_name)
    imgw = img.shape[1]
    imgh = img.shape[0]
    if (imgw < rectWidth or imgh < rectWidth):
        print(file_name, "file too small")
        return

    cv2.imwrite(full_savePath+ file_name,img)
    blockcnt = 0
    while (blockcnt<patchNumbers):
        px = random.randint(0, imgw-rectWidth-1)
        py = random.randint(0, imgh-rectWidth-1)

        cropImg = img[py:py+rectWidth, px:px+rectWidth]
        tName = "%s/%s#%s.bmp" %(cut_savePath, file_name.split('.')[0], str(blockcnt).zfill(4))
        if not os.path.exists(cut_savePath):
            os.makedirs(cut_savePath)
        cv2.imwrite(tName, cropImg)
        blockcnt += 1

def cut_image_patches(source_path, patch_dir,cut_savePath,full_savePath): 
    source_path += patch_dir
    image_real, number_real = load_images_from_dir(source_path, shuffle = True)
    for i in range(number_real):
        k = random.randint(0,9)
        if k==0:
            cut_path = cut_savePath+'/test/'+patch_dir
            full_path = full_savePath + '/test/'+patch_dir
            cut_one_image_random(source_path, cut_path, full_path, image_real[i])
                    
        elif k==1:
            cut_path = cut_savePath+'/valid/'+patch_dir
            full_path = full_savePath + '/valid/'+patch_dir
            cut_one_image_random(source_path,cut_path, full_path, image_real[i])
            
        else :
            cut_path = cut_savePath+'/train/'+patch_dir
            full_path = full_savePath + '/train/'+patch_dir
            cut_one_image_random(source_path, cut_path, full_path, image_real[i])
           

source_path ='./1800/full_img/'
pg_dir = 'pg/'
cg_dir = 'cg/'
cut_savePath = './1800/cut_224/'
full_savePath = './1800/full_img_divide/'
cut_image_patches(source_path,pg_dir,cut_savePath,full_savePath)
cut_image_patches(source_path,cg_dir,cut_savePath,full_savePath)
print("Done.")
