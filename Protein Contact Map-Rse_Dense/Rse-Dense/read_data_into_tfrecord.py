# coding: utf-8
#!/usr/bin/env python

import pickle as pickle


from libs.datasets.data_preprocessing import *
from libs.config.config import *

FLAGS = tf.app.flags.FLAGS

def read_pkl(name):
    print(name)
    with open(name,'rb') as fin:
        return pickle.load(fin,encoding='bytes')

train_infos = read_pkl(FLAGS.train_file)
print("FLAGS.data_dir:",FLAGS.data_dir)
records_dir = os.path.join(FLAGS.data_dir, 'records/')
print("records_dir:",records_dir)
add_to_tfrecord(records_dir, 'train', train_infos)
