import os
import numpy as np
import random
import cv2

method_read = 'frame'
dataset = 'ucf101_train'
data_path = '/home/zhangyuanyi31/new/data/'
video_in = 'RGB'
feature_in = 'feature'
max_num = -1
DA_setting = 'hmdb_ucf_small'  #hmdb_ucf|ucf_olympic
path_video_dataset = data_path + dataset + '/' + video_in + '/'  #path of videos
path_frame_dataset = data_path + dataset + '/' + DA_setting + '_frame/'
list_class = os.listdir(path_video_dataset)  #all class for the video type
suffix = '_' + DA_setting + '-frame'
list_class.sort()


if dataset == 'ucf101' or dataset == 'hmdb51':
    file_suffix = '_' + DA_setting
    class_file = data_path + 'data_split/' + dataset + '_split/class_list' + file_suffix + '.txt'  #split file
    #all class id in split file
    class_id = [int(line.strip().split(' ', 1)[0]) for line in open(class_file)]  # number shown in th text file
    #all class name in split file
    class_names = [line.strip().split(' ', 1)[1] for line in open(class_file)]
    #print(class_id)
    #print(class_names)
elif 'unlabeled' in dataset:
	class_id = [-1]  # number shown in th text file
	class_names = ['unlabeled']

num_class = len(set(class_id))  #sum of the class in split
#print(num_class)
list_class_video = [[] for i in range(num_class)]  ##create a list to store video paths in terms of new categories [[], [], [], [], []]
num_class_video = np.zeros(num_class, dtype=int)  #[0, 0, 0, 0, 0]
for i in range(len(class_names)):
    #print(i, class_names[i])
    list_video = os.listdir(path_video_dataset + class_names[i])
    list_video.sort()
    #print(list_video)
    list_video_name = [v.split('.')[0] for v in list_video]
    #print(list_video_name)
    id_category = class_id[i]   
    #print(id_category)
    if method_read == 'frame':
        for t in range(len(list_video)):
            lines_path = list_video_name[t] + ' ' + str(len(os.listdir(path_frame_dataset + list_video_name[t]))) + ' ' + str(id_category) + '\n' 
            print(lines_path)
    list_class_video[id_category] = list_class_video[id_category] + lines_path
    num_class_video[id_category] += len(list_video)

file = open(data_path + dataset + '/' + 'list_' + dataset + suffix + '.txt','w')
print(video_in, ': ')
    
for i in range(len(list_class_video)):
    list_video_clips = list_class_video[i]
    num_videos = len(list_video_clips)
    full_list = range(num_videos)
    select_list = random.sample(full_list, max_num) if max_num>0 and max_num<num_videos else full_list
    for j in select_list:
        file.write(list_class_video[i][j])
    print(i, len(full_list), '-->', len(select_list)) # print the number of videos in the category

file.close()
    
