''' CelebAHQ data loading utilities, adapted from https://github.com/openai/glow '''

from glob import glob

import numpy as np
import tensorflow as tf
import torch


_FILES_SHUFFLE = 1024
_SHUFFLE_FACTOR = 4


def parse_tfrecord_into_tf_tensor(record, resolution):
    features = tf.io.parse_single_example(record, features=dict(
        data=tf.io.FixedLenFeature([], tf.string)
    ))
    img = tf.io.decode_raw(features['data'], tf.uint8)
    img = tf.reshape(img, [resolution, resolution, 3])
    return img


def tf_to_torch(tf_tensor):
    tf_tensor = tf.transpose(tf_tensor, perm=(0, 3, 1, 2))
    return torch.tensor(tf_tensor.numpy())


class TFRecordIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 tfr_file_pattern,
                 batch_size=32,
                 resolution=256,
                 is_training=True,
                 tf_num_parallel_map=16,
                 batch_transform=None):
        super(TFRecordIterableDataset).__init__()
        
        files = tf.data.Dataset.list_files(tfr_file_pattern)
        if is_training:
            files = files.shuffle(buffer_size=_FILES_SHUFFLE)
        dset = files.apply(tf.data.TFRecordDataset)
        if is_training:
            dset = dset.shuffle(buffer_size=batch_size * _SHUFFLE_FACTOR)
        dset = dset.map(lambda x: parse_tfrecord_into_tf_tensor(x, resolution),
                        num_parallel_calls=tf_num_parallel_map)
        dset = dset.batch(batch_size)
        dset = dset.prefetch(1)
        self.batched_tf_tensor_iterator = \
            tf.compat.v1.data.make_one_shot_iterator(dset)

        if batch_transform:
            # Convert to torch, apply user-specified batch tensor transformation, and add label
            self.batch_transform = lambda tf_tensor: (batch_transform(tf_to_torch(tf_tensor)), None)
        else:
            # Convert to torch and add label
            self.batch_transform = lambda tf_tensor: (tf_to_torch(tf_tensor), None)

    def __iter__(self):           
        return map(self.batch_transform,
                   self.batched_tf_tensor_iterator)


def get_celeba_dataloader(data_dir="data",
                          split="train",
                          tf_num_parallel_map=16,
                          batch_transform=None,
                          **data_loader_kwargs):
    # r08 specifies resolution: log_2(256) = 8
    tfr_file = f'{data_dir}/celeba-tfr/{split}/{split}-r08-s-*-of-*.tfrecords'
    files = glob(tfr_file)
    assert len(files) == int(files[0].split(
        "-")[-1].split(".")[0]), "Not all tfrecords files present at %s" % tfr_prefix
    dataset = TFRecordIterableDataset(tfr_file,
                                      batch_size=data_loader_kwargs.get("batch_size", 32),
                                      resolution=256,
                                      is_training=(split == "train"),
                                      tf_num_parallel_map=tf_num_parallel_map,
                                      batch_transform=batch_transform)
    assert data_loader_kwargs.get("num_workers", 0) == 0
    return torch.utils.data.DataLoader(dataset, **data_loader_kwargs)