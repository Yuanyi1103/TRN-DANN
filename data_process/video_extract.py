import torch
import torchvision
import torch.nn as nn
import os
import imageio
import numpy as np

import threading
from multiprocessing.dummy import Pool as ThreadPool
import time
import argparse


path_input = '/home/zhangyuanyi31/new/data/hmdb51/RGB/'  #path for all videos
class_file = '/home/zhangyuanyi31/new/data/data_split/hmdb51_split/class_list.txt' #classes for nedded list

start_class = 1
end_class = -1

FRAME_ROOT = '/home/zhangyuanyi31/new/data/hmdb51/hmdb_ucf_full'
num_thread = 8

list_class = os.listdir(path_input)  #all classes in the video
list_class.sort()
#print(list_class)

pool = ThreadPool(num_thread)

if class_file == 'none':
        class_name_proc = ['unlabeled']
else:
        class_name_proc = [line.strip().split(' ', 1)[1] for line in open(class_file)]  #class name in split list
#print(class_name_proc)

def extract(video, video_file, tmpl='%05d.jpg'):
    os.system(f'ffmpeg -i {video} -vf scale=256:256 '
              f'{FRAME_ROOT}/{video_file[:-4]}/{tmpl}')

def extract_images(video_file):
#    print(video_file)
    path_video =  path_input + class_name + '/' + video_file #all path for the videos
    #print(path_video)
    os.makedirs(os.path.join(FRAME_ROOT, video_file[:-4]))
    
    extract(path_video, video_file)
    
    #reader = imageio.get_reader(path_input + class_name + '/' + video_file)
    #print(reader)

id_class_start = start_class-1
id_class_end = len(list_class) if end_class <= 0 else end_class
start = time.time()

for i in range(id_class_start, id_class_end):
    start_class = time.time()
    class_name = list_class[i]
    if class_name in class_name_proc:
        list_video = os.listdir(path_input + class_name + '/')
        list_video.sort()
        array_video = np.array(list_video)
        pool.map(extract_images, list_video, chunksize=1)
