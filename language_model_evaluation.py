import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Gen_Data_loader_text8, Dis_dataloader, Dis_dataloader_text8
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import pickle
import os
import collections
import json
# from tqdm import tqdm
from sequence_gan import *

def convergence_experiment(sess, tested_model, data_loader):

    data_loader.reset_pointer()
    batch = data_loader.next_batch()

    N = 100000  # hard coded for the meantime
    gap = 10 # hard coded for the meantime


    pred = np.zeros([BATCH_SIZE, SEQ_LENGTH, tested_model.num_emb], dtype=np.float32)


    for i in range(N):
        pred_one_hot, real_pred = tested_model.language_model_eval_step(sess, batch)
        pred += pred_one_hot



    pred = pred / (N * 1.)

def language_model_evaluation(sess, tested_model, data_loader):

    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        N = 1000 # hard coded for the meantime

        # estimate probability vectors for batch
        pred = np.zeros([BATCH_SIZE,SEQ_LENGTH,tested_model.num_emb],dtype=np.float32)
        for i in range(N):
            pred_one_hot, real_pred = tested_model.language_model_eval_step(sess, batch)
            pred += real_pred # pred_one_hot
        pred = pred / (N * 1.)
        pred_flat = np.reshape(pred,[BATCH_SIZE * SEQ_LENGTH,tested_model.num_emb])
        batch_flat = np.reshape(batch,[BATCH_SIZE * SEQ_LENGTH])
        eps = 1e-20

        #calc bit per char
        BPC = np.average([-np.log2(pred_flat[i,batch_flat[i]] + eps) for i in range(pred_flat.shape[0])])
        print(str(BPC))

    return BPC

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    if use_real_world_data:
        vocab_size = 27
        # split to train-eval-test
        real_data_train_file = real_data_file_path + '-train'
        real_data_eval_file = real_data_file_path + '-eval'
        real_data_test_file = real_data_file_path + '-test'
        real_data_dict_file = real_data_file_path + '-dict.json'
        if not os.path.exists(real_data_train_file):
            split_text8(real_data_file_path)
        charmap, inv_charmap = create_real_data_dict(real_data_train_file,real_data_dict_file)
        gen_data_loader = Gen_Data_loader_text8(BATCH_SIZE,charmap,inv_charmap)
        dis_data_loader = Dis_dataloader_text8(BATCH_SIZE,charmap,inv_charmap)
    else:
        gen_data_loader = Gen_Data_loader(BATCH_SIZE)
        likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
        vocab_size = 5000
        dis_data_loader = Dis_dataloader(BATCH_SIZE)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)

    if not use_real_world_data:
        target_params = pickle.load(open('save/target_params.pkl'))
        target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # sess.run(tf.global_variables_initializer())

    print('#########################################################################')
    print('loading model [%0s]...'%EXPERIMENT_NAME)

    save_file = os.path.join('.', 'ckp', EXPERIMENT_NAME, EXPERIMENT_NAME)
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(list(zip([x.name.split(':')[0] for x in tf.global_variables()], tf.global_variables())))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
            else:
                print(("Not loading: %s." % saved_var_name))
    saver = tf.train.Saver(restore_vars)
    print ("Restoring vars:")
    print (restore_vars)
    saver.restore(sess, save_file)


    print('#########################################################################')
    print('Start Language Model Evaluation...')
    test_data_loader = Gen_Data_loader_text8(BATCH_SIZE,charmap,inv_charmap)
    test_data_loader.create_batches(real_data_test_file)
    language_model_evaluation(sess,generator, test_data_loader)

if __name__ == '__main__':
    main()