#!/usr/bin/env python

#import libs.nets.network as network
import libs.nets.DenseNet_LSTM as network
#import libs.nets.highway_network as network
import libs.datasets.data_preprocessing as data_preprocess
from libs.config.config import *
from libs.utils.acc_cal_v2 import topKaccuracy, evaluate, output_result, output_result1

import tensorflow as tf
import numpy as np
import pickle as pickle
import os
# using GPU numbered 0
os.environ["CUDA_VISIBLE_DEVICES"]='1'

def load_test_data():
    #datafile = "data/pdb25-test-500.release.contactFeatures.pkl"
    datafile = "data/Mems400.2016.contactFeatures.pkl"

    f = open(datafile,'rb')
    data = pickle.load(f,encoding='bytes')
    f.close()
    return data

def test():
    # restore graph
    input_1d = tf.placeholder("float", shape=[None, None, 46], name="input_x1")
    input_2d = tf.placeholder("float", shape=[None, None, None, 5], name="input_x2")
    label = tf.placeholder("float", shape=None, name="input_y")
    is_training = tf.placeholder(tf.bool)
    output = network.build(is_training, input_1d, input_2d, label,
            FLAGS.filter_size_1d, FLAGS.filter_size_2d,
            FLAGS.block_num_1d, FLAGS.block_num_2d,
            regulation=True, batch_norm=True)
    prob = output['output_prob']

    # restore model
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
    #checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt-90000")
    print ("Loading model from %s" %checkpoint_path)
    restorer = tf.train.Saver()
    restorer.restore(sess, checkpoint_path)
    
    # prediction
    #predict single one
    #output_prob = sess.run(prob, feed_dict={input_1d: f_1d, input_2d: f_2d})
    data = load_test_data()
    input_acc = []
    output_acc = []
    output_recall = []
    input_recall = []
    for i in range(len(data)):
        d = data[i]
        name, seqLen, sequence_profile, pairwise_profile, true_contact = \
                data_preprocess.extract_single(d)
        print ("processing %d %s" %(i+1, name))
        sequence_profile = sequence_profile[np.newaxis, ...]
        print (sequence_profile.shape)
        pairwise_profile = pairwise_profile[np.newaxis, ...]
        print (pairwise_profile.shape)
        y_out = sess.run(prob, \
                feed_dict = {input_1d: sequence_profile, input_2d: pairwise_profile, is_training:False})
        np.savetxt("results/"+name.decode()+".deepmat", y_out[0,:,:,1])
        np.savetxt("contacts/"+name.decode()+".contacts", true_contact[0])
        input_temp_acc,input_temp_recall = evaluate(pairwise_profile[0,:,:,0], true_contact)
        input_acc.append(input_temp_acc)
        input_recall.append(input_temp_recall)
        print("y_out1:",y_out[0,:,:,1])
        print("true_contact:",true_contact)
        print("y_out1:",y_out[0,:,:,1].shape)
        print("true_contact:",true_contact.shape)
        output_temp_acc,output_temp_recall = evaluate(y_out[0,:,:,1], true_contact)
        output_acc.append(output_temp_acc)
        output_recall.append(output_temp_recall)

    print("output_acc:",output_acc)
    print("output_recall:",output_recall)

    print ("Input result:")
    output_result(np.mean(np.array(input_acc), axis=0))
    print ("\nOutput result:")
    output_result(np.mean(np.array(output_acc), axis=0))
    output_result1(np.mean(np.array(output_recall), axis=0))
    
if __name__ == "__main__":
    test()
