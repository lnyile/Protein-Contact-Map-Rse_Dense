#!/usr/bin/env python

import os
import pickle as pickle
import numpy as np
import math
import tensorflow as tf
import codecs
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
from libs.config.config import *


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def to_tfexample_raw(name, seqLen, seq_feature, pair_feature, label_data):
    return tf.train.Example(features=tf.train.Features(feature={
        'name': _bytes_feature(name),
        'seqLen': _int64_feature(seqLen),
        'seq_feature': _bytes_feature(seq_feature),         # of shape (L, L, 26)
        'pair_feature': _bytes_feature(pair_feature),       # of shape (L, L, 5)
        'label_matrix': _bytes_feature(label_data),         # of shape (L, L)
    }))

def get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
    output_filename = '%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, num_shards)
    return os.path.join(dataset_dir, output_filename)

def extract_single(info):
    print("info:",info)

    name = info[b'name']
    seq = info[b'sequence']
    seqLen = len(seq)
    acc = info[b'ACC']
    ss3 = info[b'SS3']
    pssm = info[b'PSSM']
    psfm = info[b'PSFM']

    f = codecs.open("feature/sequence/"+name.decode()+".txt",'w','utf-8')
    f.write(seq.decode())

    f = codecs.open("feature/accuracy/"+name.decode()+".txt",'w','utf-8')
    f.write(str(acc))

    f = codecs.open("feature/ss3/"+name.decode()+".txt",'w','utf-8')
    f.write(str(ss3))

    f = codecs.open("feature/pssm/"+name.decode()+".txt",'w','utf-8')
    f.write(str(pssm))

    f = codecs.open("feature/psfm/"+name.decode()+".txt",'w','utf-8')
    f.write(str(psfm))


    #diso = info[b'DISO']
    print("psfm:",psfm.shape)
    #print("diso:",diso.shape)
    sequence_profile = np.concatenate((pssm, ss3, acc, psfm), axis = 1)
    ccmpred = info[b'ccmpredZ']

    f = codecs.open("feature/ccmpred/"+name.decode()+".txt",'w','utf-8')
    f.write(str(ccmpred))

    #psicov = info[b'psicovZ']
    other = info[b'OtherPairs']
    #pairwise_profile = np.dstack((ccmpred, psicov))
    #pairwise_profile = np.concatenate((pairwise_profile, other), axis = 2) #shape = (L, L, 5)

    pairwise_profile = np.dstack((ccmpred, ccmpred))
    print("pairwise_profile:",pairwise_profile.shape)
    pairwise_profile = np.concatenate((pairwise_profile, other), axis = 2) #shape = (L, L, 5)

    #datafile = "data/76CAMEO/"+name.decode()+".txt"
    datafile = "data/mems400/"+name.decode()+".txt"
    #datafile = "data/CASP11/"+name.decode()+".txt"
    f = open(datafile,'rb')
    data = pickle.load(f,encoding='bytes')
    f.close()


    true_contact = data[b'contactMatrix']

    #true_contact = [[-1] * pairwise_profile.shape[0] for _ in range(pairwise_profile.shape[1])]
    #true_contact = np.array(true_contact)
    #print("true_contact:",true_contact.shape)
    true_contact = np.array(true_contact)
    print("true_contact:",np.shape(true_contact))
    true_contact[true_contact < 0] = 0 # transfer -1 to 0, shape = (L, L)
    true_contact = np.tril(true_contact, k=-6) + np.triu(true_contact, k=6) # remove the diagnol contact
    true_contact = true_contact.astype(np.uint8)

    return name, seqLen, sequence_profile, pairwise_profile, true_contact
    
def add_to_tfrecord(records_dir, split_name, infos):
    """Loads image files and writes files to a TFRecord.
    Note: masks and bboxes will lose shape info after converting to string.
    """
    num_shards = int(len(infos) / 1000)
    num_per_shard = int(math.ceil(len(infos) / float(num_shards)))
      
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        with tf.Session('') as sess:
            for shard_id in range(num_shards):
                record_filename = get_dataset_filename(records_dir, split_name, shard_id, num_shards)
                options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
                with tf.python_io.TFRecordWriter(record_filename, options=options) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(infos))
                    print ("processing %s_data from %d to %d..." %(split_name, start_ndx, end_ndx))
                    for i in range(start_ndx, end_ndx):
                        info = infos[i]
                        name, seqLen, seq_feature, pair_feature, label = extract_single(info)
                        if seqLen > 300:
                            continue
                        #print "generate tfrecord for %s" %name
                        seq_feature = seq_feature.astype(np.float32)
                        pair_feature = pair_feature.astype(np.float32)
                        label = label.astype(np.uint8)
                        
                        example = to_tfexample_raw(name, seqLen, seq_feature.tostring(), pair_feature.tostring(), label.tostring())
                        tfrecord_writer.write(example.SerializeToString())
    
