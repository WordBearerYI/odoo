from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

import warnings
import argparse
import numpy as np
import yaml
import cv2
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

warnings.filterwarnings("ignore")
cur_dir = sys.path[0]+'/'
pack_dir = cur_dir+'ssd.pytorch/'
if pack_dir not in sys.path:
    sys.path.append(pack_dir)
    
from ssd import build_ssd
from postprocess import box_filter_hough, box_filter_none, box_filter_center

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

f = open(cur_dir+'config.yaml')
params = yaml.load(f)

save_path = cur_dir + params['save_path']
post_process = params['post_process']
post_param_1 = params['post_param1'] 
post_param_2 = params['post_param2']
base_model = cur_dir + params['base_model']
ssd_model = cur_dir + params['ssd_model']
cuda = params['cuda']
voc_root = cur_dir + params['voc_root']

'''
parser = argparse.ArgumentParser(description='FISH_signal seperation')
parser.add_argument('--post_process',default='hough',
                    help='postprocess type') 
parser.add_argument('--post_param_1',default= 12.9,
                    help='postprocess parameter 1') 
parser.add_argument('--post_param_2',default= 100,
                    help='postprocess parameter 2') 
parser.add_argument('--base_model', default=cur_dir+'/weight/vgg16_reducedfc.pth', 
                    help='pretrained base model')
parser.add_argument('--ssd_model',default=cur_dir+'/weight/ssd_cell_20190625_1_870.pth',
                    help = 'trained ssd model') 
parser.add_argument('--save_path', default=cur_dir+'/eval/', type=str,
                    help='File path to save results')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=cur_dir+"/ssd.pytorch/ydx_tmp/", 
                    help='Location of VOC root directory')
args = parser.parse_args()

save_path = cur_dir + args.save_path
post_process = args.post_process
post_param_1 = args.post_param_1 
post_param_2 = args.post_param_2
base_model = cur_dir + args.base_model
ssd_model =  cur_dir + args.ssd_model
cuda = args.cuda
voc_root =  cur_dir + args.voc_root
'''

if not os.path.exists(save_path):
    os.mkdir(save_path)

if cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(voc_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
YEAR = '2007'
dataset_mean = (100, 100, 100)
set_type = 'test'

def cvt(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def SobelinCPU(img):
    if img.size == 0:
        raise ValueError('empty image.')
    
    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    sobelimg = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return sobelimg

def cal_mean_grayvalue(image, mask):
    num_pixel = len(np.nonzero(mask)[0])
    sum_pixel_value = np.sum(image*mask)
    mean_value = sum_pixel_value / float(num_pixel)
    return mean_value

def cal_signal_num(image, tumor_cell_mask, large_signal_thresh=30, save_signal_image=False, save_path=None):
    
    CHANNEL_THRESH = 100
    MASK_THRESH = 30
    
    def find_signal(img, mask):       
        img = cv2.threshold(img, CHANNEL_THRESH, 255, cv2.THRESH_BINARY)[1]
        img_blur = cv2.medianBlur(img, 3) 
        #img_f = SobelinCPU(img_blur)
        kernel = np.ones((3,3), np.uint8)
        img_f = cv2.morphologyEx(img_blur, cv2.MORPH_CLOSE, kernel)
        #img_f = cv2.threshold(img_f, IMG_THRESH, 255, cv2.THRESH_BINARY)[1]
        #mask_f = cv2.threshold(mask, MASK_THRESH, 255, cv2.THRESH_OTSU)[1]
        #img_f = img_f*mask_f
        signal_area = []
        
        _, contours, hierarchy = cv2.findContours(img_f, 1, 2)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # remove empty contour
            if area == 0:
                continue
            signal_area.append(area)
            
        return img_f, signal_area

    channels = cv2.split(image)
    b_channel = channels[0]
    g_channel = channels[1]
    r_channel = channels[2]
    
    if save_signal_image == True:
        cv2.imwrite(save_path+'/origin_image.jpg', image)
    else:
        pass

    red_signal, red_signal_area = find_signal(r_channel, b_channel)
    green_signal, green_signal_area = find_signal(g_channel, b_channel)
    
    r_mask = red_signal == 0
    g_mask = green_signal == 0
    
    if save_signal_image == True:
        cv2.imwrite(save_path+'/red_signal_image.jpg', ((red_signal*g_mask)[..., np.newaxis]>0)*image)
        cv2.imwrite(save_path+'/green_signal_image.jpg', (green_signal[..., np.newaxis]>0)*image)
    else:
        pass
    
    ### her2/cep17 > 20
    # thr = 15 because basically cep17 signal larger than her2 signal
    #if np.sum(green_signal)!=0 and np.sum(red_signal) / np.sum(green_signal) > 15 :
    #    return -1, -1, -1, -1
    
    num_red_signal = len(red_signal_area)
    # get approximate signal area use median
    if num_red_signal != 0:
        ave_area = (sorted(red_signal_area))[num_red_signal//4]
        # ave_area = sum(sorted(red_signal_area)[:5]) / 5
        # print(ave_area)
        # calculate large signal
        for i in red_signal_area:
            if i > large_signal_thresh:
                num_red_signal += max((round(i / ave_area) - 1), 0)
    
    num_green_signal = len(green_signal_area)
    # get approximate signal area use median
    
    if num_green_signal != 0:
        ave_area = (sorted(green_signal_area))[num_green_signal//4]
        # ave_area = sum(sorted(green_signal_area)[:5]) / 5
        # print(ave_area)
        # calculate large signal
        for j in green_signal_area:
            if j > large_signal_thresh:
                num_green_signal += max((round(j / ave_area) - 1), 0)
    
    return red_signal, green_signal, num_red_signal, num_green_signal


def count_writer():
    
    data_dir = voc_root+'VOC2007/JPEGImages/'
    img_lst = filter(lambda x: (x.split('.')[-1] == 'jpg') | (x.split('.')[-1] == 'png'), os.listdir(data_dir)) 
    num_classes = 2
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(ssd_model))
    net.eval()
    print('Finished loading model!')
    
    for name in tqdm(img_lst):
        cur_save_path = save_path + ''.join(name.split('.')[:-1])
       
        if not os.path.exists(cur_save_path):
            os.makedirs(cur_save_path)
      
        image = cv2.imread(data_dir+name)
        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (100.0, 100.0, 100.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()                   # bgr to rgb
        x = torch.from_numpy(x).permute(2, 0, 1)   # channel first
        xx = Variable(x.unsqueeze(0))              # wrap tensor in Variable
        if cuda and torch.cuda.is_available():
            xx = xx.cuda()
            net = net.cuda()
        y = net(xx)
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([image.shape[1::-1], image.shape[1::-1]]).view(1,4)
        detections = detections.cuda()
        image_draw = image.copy()
        mask = np.zeros(image.shape)
        for i in range(detections.size(1)):
            j = 0
            while detections[0,i,j,0] >= 0.3:
                score = detections[0,i,j,0]    
                pt = (detections[0,i,j,1:]*scale)[0].cpu().numpy()
                box = (round(pt[0]),round(pt[1]),round(pt[2]),round(pt[3]))
                if post_process=='hough':
                    filter_res = box_filter_hough(box,image,post_param_1,post_param_2)
                elif post_process=='center':
                    filter_res = box_filter_center(box,image,post_param_1,post_param_2)
                else:
                    filter_res = box_filter_none(box,image,post_param_1,post_param_2)
                if filter_res:
                    cv2.rectangle(image_draw, (pt[0], pt[1]), (pt[2], pt[3]), (255, 255, 255), 2)
                    cv2.rectangle(mask, (pt[0], pt[1]), (pt[2], pt[3]), (255,255,255), -1)
                j += 1
                

        cv2.imwrite(cur_save_path +'/tumor_cell.jpg', image_draw)
        red_signal, green_signal, num_red_signal, num_green_signal = cal_signal_num(image, mask, save_signal_image=True, save_path=cur_save_path)
        with open(cur_save_path + '/result.txt', 'w') as f:
            f.write('number of her2 signal in tumor cells: '+str(int(num_red_signal))+'\n')
            f.write('number of cep17 signal in tumor cells: '+str(int(num_green_signal)))
        
        

if __name__ == '__main__':
    count_writer()
    
    
    
    
    



