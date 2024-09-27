import os
import numpy as np
import tensorflow as tf
import input_data
import model

N_CLASSES = 3   # 分类数
IMG_W = 144     # resize图像，太大的话训练时间久
IMG_H = 144
BATCH_SIZE = 20
CAPACITY = 200
MAX_STEP = 10000  # 一般大于10K
learning_rate = 0.0001  # 一般小于0.0001

train_dir = 'D:\\澳科读书\\实习\\深度学习\\data'  # 训练样本的读入路径
logs_train_dir = 'D:\\澳科读书\\实习\\深度学习\\save'  # logs存储路径

train, train_label, val, val_label = input_data.get_files(train_dir, 0.3)
# 训练数据及标签
train_batch, train_label_batch = input_data.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
# 测试数据及标签
val_batch, val_label_batch = input_data.get_batch(val, val_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)


train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = model.losses(train_logits, train_label_batch)
train_op = model.trainning(train_loss, learning_rate)
train_acc = model.evaluation(train_logits, train_label_batch)

summary_op = tf.summary.merge_all()

sess = tf.Session()

train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    # 执行MAX_STEP步的训练，一步一个batch
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

        # 每隔50步打印一次当前的loss以及acc，同时记录log，写入writer
        if step % 10 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)
        # 每隔100步，保存一次训练好的模型
        if (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')

finally:
    coord.request_stop()