import os
import torch
import torchvision
import torchvision.datasets as datasets


ROOT_DATASET = '/home/zhangyuanyi31/new'


def return_hmdb_ucf_small(modelity):
    filename_categories = 'data/data_pre/classInd_hmdb_ucf_small.txt'
    if modelity == 'RGB':
        root_data_source = '/home/zhangyuanyi31/new/data/ucf101/frame'
        root_data_target = '/home/zhangyuanyi31/new/data/hmdb51/frame'
        filename_sourcelist_train = 'data/ucf101/list_ucf101_train_hmdb_ucf_small-frame.txt'
        filename_targetlist_train = 'data/hmdb51/list_hmdb51_train_hmdb_ucf_small-frame.txt'
        filename_imglist_val = 'data/ucf101/list_ucf101_val_hmdb_ucf_small-frame.txt'
        prefix = '{:05d}.jpg'
    else:
        print('no such modality:' + modality)
        os.exit()
    return filename_categories, filename_sourcelist_train, filename_targetlist_train, filename_imglist_val, root_data_source, root_data_target, prefix

def return_hmdb_ucf(modelity):
    filename_categories = 'data/data_pre/classInd_hmdb_ucf.txt'
    if modelity == 'RGB':
        root_data = '/home/zhangyuanyi31/new/data/ucf101/frame'
        filename_imglist_train = 'data/ucf101/list_ucf101_train_hmdb_ucf-frame.txt'
        filename_imglist_val = 'data/ucf101/list_ucf101_val_hmdb_ucf-frame.txt'
        prefix = '{:05d}.jpg'
    else:
        print('no such modality:' + modality)
        os.exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_something(modality):
    filename_categories = 'data/sv1/category.txt'
    if modality == 'RGB':
        # root_data = '/data/vision/oliva/scratch/bzhou/video/something-something/v2/20bn-something-something-v2'
        root_data = '/home/zhangyuanyi31/new/data/sv1/20bn-something-something-v2-frames'
        # root_data = '/mnt/localssd1/aandonia/something/v2/20bn-something-something-v2-frames'
        filename_imglist_train = '/home/zhangyuanyi31/new/data/sv1/train_videofolder.txt'
        filename_imglist_val = '/home/zhangyuanyi31/new/data/sv1/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = '/data/vision/oliva/scratch/bzhou/video/something-something/flow'
        #root_data = '/mnt/localssd1/bzhou/something/flow'
        filename_imglist_train = 'something/train_videofolder.txt'
        filename_imglist_val = 'something/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    else:
        print('no such modality:' + modality)
        os.exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_somethingv2(modality):
    filename_categories = 'something/v2/category.txt'
    if modality == 'RGB':
        # root_data = '/data/vision/oliva/scratch/bzhou/video/something-something/v2/20bn-something-something-v2'
        root_data = '/mnt/localssd2/aandonia/something/v2/20bn-something-something-v2-frames'
        # root_data = '/mnt/localssd1/aandonia/something/v2/20bn-something-something-v2-frames'
        filename_imglist_train = 'something/v2/train_videofolder.txt'
        filename_imglist_val = 'something/v2/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        #root_data = '/data/vision/oliva/scratch/bzhou/video/something-something/flow'
        # root_data = '/mnt/localssd1/bzhou/something/flow'
        root_data = '/mnt/localssd2/aandonia/something/v2/flow'
        filename_imglist_train = 'something/v2/train_videofolder.txt'
        filename_imglist_val = 'something/v2/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    else:
        print('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester(modality):
    filename_categories = 'jester/category.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        #root_data = '/data/vision/oliva/scratch/bzhou/video/jester/20bn-jester-v1'
        root_data = '/mnt/localssd1/bzhou/jester/20bn-jester-v1'
        filename_imglist_train = 'jester/train_videofolder.txt'
        filename_imglist_val = 'jester/val_videofolder.txt'
    elif modality == 'Flow':
        root_data = '/data/vision/oliva/scratch/bzhou/video/jester/flow'
        #root_data = '/mnt/localssd1/bzhou/something/flow'
        filename_imglist_train = 'jester/train_videofolder.txt'
        filename_imglist_val = 'jester/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        os.exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_charades(modality):
    filename_categories = 'charades/category.txt'
    filename_imglist_train = 'charades/train_segments.txt'
    filename_imglist_val = 'charades/test_segments.txt'
    if modality == 'RGB':
        prefix = '{:06d}.jpg'
        root_data = '/data/vision/oliva/scratch/bzhou/charades/Charades_v1_rgb'
    else:
        print('no such modality:'+modality)
        os.exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_moments(modality):
    filename_categories = '/data/vision/oliva/scratch/moments/split/categoryList_nov17.csv'
    if modality == 'RGB':
        prefix = '{:06d}.jpg'
        root_data = '/data/vision/oliva/scratch/moments/moments_nov17_frames'
        filename_imglist_train = '/data/vision/oliva/scratch/moments/split/rgb_trainingSet_nov17.csv'
        filename_imglist_val = '/data/vision/oliva/scratch/moments/split/rgb_validationSet_nov17.csv'

    elif modality == 'Flow':
        root_data = '/data/vision/oliva/scratch/moments/moments_nov17_flow'
        prefix = 'flow_xyz_{:05d}.jpg'
        filename_imglist_train = '/data/vision/oliva/scratch/moments/split/flow_trainingSet_nov17.csv'
        filename_imglist_val = '/data/vision/oliva/scratch/moments/split/flow_validationSet_nov17.csv'
    else:
        os.exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
                   'charades': return_charades, 'moments': return_moments, 'hmdb_ucf_small': return_hmdb_ucf_small,
                   'hmdb_ucf': return_hmdb_ucf}
    if dataset in dict_single:
        file_categories, file_sourcelist_train, file_targetlist_train, file_imglist_val, root_data_source, root_data_target, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_sourcelist_train = os.path.join(ROOT_DATASET, file_sourcelist_train)
    file_targetlist_train = os.path.join(ROOT_DATASET, file_targetlist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    file_categories = os.path.join(ROOT_DATASET, file_categories)
    with open(file_categories) as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]
    return categories, file_sourcelist_train, file_targetlist_train, file_imglist_val, root_data_source, root_data_target, prefix
