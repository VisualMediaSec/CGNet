import cv2 
import numpy as np 
import random 
import os 

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
    return file_list

def cut_one_image_random(input_path, output_path,file_name,model,patchNumbers = 20,rectWidth=224):
    img = cv2.imread(input_path + '/' + file_name)
    imgw = img.shape[1]
    imgh = img.shape[0]
    if (imgw < rectWidth or imgh < rectWidth):
        print(file_name, "file too small")
        return
    blockcnt = 0
    while (blockcnt<patchNumbers):
        px = random.randint(0, imgw-rectWidth)
        py = random.randint(0, imgh-rectWidth)
        cropImg = img[py:py+rectWidth, px:px+rectWidth]
        k = random.randint(0,10)
        if k <3:
            cropImg = attack(cropImg,model)
        tName = "%s/%s#%s.bmp" %(output_path, file_name.split('.')[0], str(blockcnt).zfill(4))
        cv2.imwrite(tName, cropImg)
        blockcnt += 1


def translation(img,dis):
    # ping yi 
    k1 = random.randint(-dis,dis)
    k2 = random.randint(-dis,dis)
    M = np.float32([[1,0,k1],[0,1,k2]])
    img_ = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
    return img_

def noise(img,snr,model):
    if(model == 'sp'):
        return sp_noise(img,snr)
    else:
        return gauss_noise(img,snr)

def sp_noise(img,snr):
    # 
    snr = random.uniform(snr,1)
    img_ = img.copy()
    h,w,c = img_.shape
    mask = np.random.choice((0,1,2),size=(h,w,1),p=[snr,(1-snr)/2.,(1-snr)/2.])
    mask = np.repeat(mask,c,axis=-1)
    img_[mask==1] =255
    img_[mask==2] =0
    return img_

def gauss_noise(src,snr):
    percetage = 1-snr
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        NoiseImg[randX, randY]=NoiseImg[randX,randY]+random.gauss(0,1)
        channel = src.shape[2] - 1
        for c in  range(channel):
            if  NoiseImg[randX, randY, c] < 0:
                NoiseImg[randX, randY]=0
            elif NoiseImg[randX, randY, c] > 255:
                NoiseImg[randX, randY]=255
    return NoiseImg

def zoom(img,k1,k2):
    k = random.uniform(k1,k2)
    mask = np.ndarray(img.shape,dtype=np.uint8)
    img = cv2.resize(img,(int(img.shape[1]*k),int(img.shape[0]*k)))
    if k > 1:
        mask[:,:,:] = img[:mask.shape[0],:mask.shape[1],:]
    else:
        mask[:img.shape[0],:img.shape[1],:] = img[:,:,:]
    return mask 

def block(img,bs):
    img_ = img.copy()
    h = random.randint(0,img.shape[0]-bs)
    w = random.randint(0,img.shape[1]-bs)
    img_[h:h+bs,w:w+bs,:] = 0 
    return img_ 

def color_cvt(img,model):
    if model=='gray':
        res = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    elif model=='hsv':
        res = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    return res 

def affine_transf(img):
    row,col,ch = img.shape
    pts1 = np.float32([[0,0],[col-1,0],[0,row-1]])
    pts2 = np.float32([[col*0.2,row*0.1],[col*0.9,row*0.2],[col*0.1,row*0.9]])
    M = cv2.getAffineTransform(pts1,pts2)
    res = cv2.warpAffine(img,M,(col,row))
    return res 

def im_filter(img,model):
    if(model =='gauss'):
        img = cv2.GaussianBlur(img,(3,3),0)
    elif(model == 'median'):
        img = cv2.medianBlur(img,3)
    return img 

def attack(img,model):
    a = random.randint(0,7)
    if(model =='easy'):
        if(a==0):    
            return noise(img,0.9,'sp')
        elif a==1:
            return translation(img,50)
        elif a== 2:
            return zoom(img,0.75,1.25)
        elif a==3:
            return block(img,5)
        elif a==4:
            return color_cvt(img,'gray')
        elif a== 5:
            return affine_transf(img)
        else:
            return im_filter(img,'gauss')
    else:
        if(a==0):
            k = random.randint(0,2)
            if(k==1):
                return noise(img,0.8,'sp')
            else:
                return noise(img,0.8,'gau')
        elif a==1:
            return translation(img,100)
        elif a== 2:
            return zoom(img,0.5,1.5)
        elif a==3:
            return block(img,10)
        elif a==4:
            k = random.randint(0,2)
            if(k==1):
                return color_cvt(img,'gray')
            else :
                return color_cvt(img,'hsv')
        elif a== 5:
            return affine_transf(img)
        else:
            k = random.randint(0,2)
            if(k==1):
                return im_filter(img,'gauss')
            else :
                return im_filter(img,'median')

if __name__ =='__main__':
    pg_path = './M0vote/pg/'
    cg_path = './M0vote/cg/'
    pg_out_path = './M1train/pg/'
    cg_out_path = './M1train/cg/'
    pg_list = load_images_from_dir(pg_path)
    cg_list = load_images_from_dir(cg_path)
    for i in pg_list:
        cut_one_image_random(pg_path,pg_out_path,i,'easy')
    for i in cg_list:
        cut_one_image_random(cg_path,cg_out_path,i,'easy')

    pg_path = './4850/full_img/pg/'
    cg_path = './4850/full_img/cg/'
    pg_list = load_images_from_dir(pg_path)
    cg_list = load_images_from_dir(cg_path)
    for i in pg_list:
        cut_one_image_random(pg_path,pg_out_path,i,'easy')
    for i in cg_list:
        cut_one_image_random(cg_path,cg_out_path,i,'easy')
