import os.path

import tensorflow as tf
import json

with open('./config_param.json') as config_file:
    config = json.load(config_file)

BATCH_SIZE = int(config['batch_size'])


def _decode_samples(image_list, shuffle=False):
    decomp_feature = {
        # image size, dimensions of 3 consecutive slices
        'dsize_dim0': tf.io.FixedLenFeature([], tf.int64),  # 256
        'dsize_dim1': tf.io.FixedLenFeature([], tf.int64),  # 256
        'dsize_dim2': tf.io.FixedLenFeature([], tf.int64),  # 3
        # labels size, dimension of the middle slice
        'lsize_dim0': tf.io.FixedLenFeature([], tf.int64),  # 256
        'lsize_dim1': tf.io.FixedLenFeature([], tf.int64),  # 256
        'lsize_dim2': tf.io.FixedLenFeature([], tf.int64),  # 1
        # image slices of size [256, 256, 3]
        'data_vol': tf.io.FixedLenFeature([], tf.string),
        # labels slice of size [256, 256, 3]
        'label_vol': tf.io.FixedLenFeature([], tf.string)}

    raw_size = [256, 256, 3]
    volume_size = [256, 256, 3]
    label_size = [256, 256, 1]  # the labels has size [256,256,3] in the preprocessed data, but only the middle slice is used

    # 创建一个数据集
    dataset = tf.data.TFRecordDataset(image_list)

    # 定义解析函数
    def _parse_function(proto):
        # 解析样本
        parsed_features = tf.io.parse_single_example(proto, decomp_feature)

        data_vol = tf.io.decode_raw(parsed_features['data_vol'], tf.float32)
        data_vol = tf.reshape(data_vol, raw_size)
        data_vol = tf.slice(data_vol, [0, 0, 0], volume_size)

        # 放缩到[-1, 1]
        # data_vol = 2*(data_vol - tf.reduce_min(data_vol)) / (tf.reduce_max(data_vol) - tf.reduce_min(data_vol))-1

        label_vol = tf.io.decode_raw(parsed_features['label_vol'], tf.float32)
        label_vol = tf.reshape(label_vol, raw_size)
        label_vol = tf.slice(label_vol, [0, 0, 1], label_size)

        batch_y = tf.one_hot(tf.cast(tf.squeeze(label_vol), tf.uint8), 5)
        return tf.expand_dims(data_vol[:, :, 1], axis=2), batch_y

    # 应用解析函数
    dataset = dataset.map(_parse_function)

    # 如果需要打乱数据
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    return dataset


def _load_samples(root, source_pth, target_pth,do_shuffle):

    with open(os.path.join(root,source_pth), 'r') as fp:
        rows = fp.readlines()
    imagea_list = [os.path.join(root,row[:-1]) for row in rows]

    with open(os.path.join(root,target_pth), 'r') as fp:
        rows = fp.readlines()
    imageb_list = [os.path.join(root,row[:-1]) for row in rows]

    dataset_source = _decode_samples(imagea_list, shuffle=do_shuffle)
    dataset_target = _decode_samples(imageb_list, shuffle=do_shuffle)

    return dataset_source, dataset_target


def load_data(root, source_pth, target_pth, do_shuffle=False):

    dataset_source, dataset_target = _load_samples(root, source_pth, target_pth, do_shuffle)
    dataset_source = dataset_source.batch(BATCH_SIZE)
    dataset_target = dataset_target.batch(BATCH_SIZE)
    return dataset_source, dataset_target
