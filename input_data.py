import os
import math
import numpy as np
import tensorflow as tf

open = []
label_open = []
close = []
label_close = []
middle = []
label_middle = []


# step1：获取所有的图片路径名，存放到
# 对应的列表中，同时贴上标签，存放到label列表中。
def get_files(file_dir, ratio):
    for file in os.listdir(file_dir + '/open'):
        open.append(file_dir + '/open' + '/' + file)
        label_open.append(0)
    for file in os.listdir(file_dir + '/middle'):
        middle.append(file_dir + '/middle' + '/' + file)
        label_middle.append(1)
    for file in os.listdir(file_dir + '/close'):
        close.append(file_dir + '/close' + '/' + file)
        label_close.append(2)
        
    image_list = np.hstack((open, middle, close))
    label_list = np.hstack((label_open, label_middle, label_close))
    
    temp = np.array([image_list, label_list])   ##### np.array([ , ])：将多个数组合并为多维数组
    temp = temp.transpose()                     ##### temp.transpose()：将数组矩阵转置
    np.random.shuffle(temp)
    
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])
    
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * ratio))  # 测试样本数
    n_train = n_sample - n_val  # 训练样本数

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images, tra_labels, val_images, val_labels
    
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # 转换类型
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=1)
    
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch