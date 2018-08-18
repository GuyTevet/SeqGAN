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
from tqdm import tqdm
from sequence_gan import *

SEQ_LENGTH = 200 # IF CHANGED - also delete Cache - aka 'text8-test.txt.npy'
# EXPERIMENT_NAME = 'lm_only_320_0_0'

def convergence_experiment(sess, tested_model, data_loader):

    data_loader.reset_pointer()
    batch = data_loader.next_batch()

    N_MAX = 5000  # hard coded for the meantime
    gap = 10 # hard coded for the meantime

    indexes = np.array(range(0,N_MAX+1,gap),dtype=np.float32)
    results = np.zeros_like(indexes)

    pred = np.zeros([BATCH_SIZE, SEQ_LENGTH, tested_model.num_emb], dtype=np.float32)
    weight = 0

    for tot_i in range(indexes.shape[0] - 1):
        #clculate the gap prediction
        gap_pred = np.zeros([BATCH_SIZE, SEQ_LENGTH, tested_model.num_emb], dtype=np.float32)
        for gap_i in range(gap):
            pred_one_hot, real_pred = tested_model.language_model_eval_step(sess, batch)
            gap_pred += pred_one_hot
        gap_pred /= gap * 1.

        #calculate new prediction including gap prediction
        new_pred = (pred * weight) + (gap_pred * gap)
        weight += gap
        new_pred /= weight * 1.

        #calculate inf nnorm error
        inf_norm_err = np.linalg.norm((new_pred - pred),ord=np.inf,axis=2)
        avg_err = np.average(inf_norm_err)
        results[tot_i+1] = avg_err

        #updating pred
        pred = new_pred

    return np.concatenate((np.expand_dims(indexes,axis=1),np.expand_dims(results,axis=1)),axis=1).transpose()


def language_model_evaluation_direct(sess, tested_model, data_loader):

    data_loader.reset_pointer()
    BPC_list = []

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        pred_one_hot, real_pred = tested_model.language_model_eval_step(sess, batch)
        real_pred = np.clip(real_pred,1e-20,1)
        pred_flat = np.reshape(real_pred,[BATCH_SIZE * SEQ_LENGTH,tested_model.num_emb])
        batch_flat = np.reshape(batch,[BATCH_SIZE * SEQ_LENGTH])

        #calc bit per char
        BPC = np.average([-np.log2(pred_flat[i,batch_flat[i]]) for i in range(pred_flat.shape[0])])
        BPC_list.append(BPC)


    return np.average(BPC_list)

def language_model_evaluation_by_approximation(sess, tested_model, data_loader):

    data_loader.reset_pointer()
    BPC_list = []
    N = 2000 #chosen N for the paper

    for it in tqdm(range(data_loader.num_batch)):

        batch = data_loader.next_batch()

        approx_pred = np.zeros([BATCH_SIZE,SEQ_LENGTH,tested_model.num_emb],dtype=np.float32)

        for n in range(N):
            pred_one_hot, real_pred = tested_model.language_model_eval_step(sess, batch)
            approx_pred += pred_one_hot

        approx_pred /= N * 1.
        approx_pred = np.clip(approx_pred,1e-20,1)
        pred_flat = np.reshape(approx_pred,[BATCH_SIZE * SEQ_LENGTH,tested_model.num_emb])
        batch_flat = np.reshape(batch,[BATCH_SIZE * SEQ_LENGTH])

        #calc bit per char
        BPC = np.average([-np.log2(pred_flat[i,batch_flat[i]]) for i in range(pred_flat.shape[0])])
        BPC_list.append(BPC)


    return np.average(BPC_list)

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
        gen_data_loader = Gen_Data_loader_text8(BATCH_SIZE,charmap,inv_charmap,SEQ_LENGTH)
        dis_data_loader = Dis_dataloader_text8(BATCH_SIZE,charmap,inv_charmap,SEQ_LENGTH)
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

    # for exp_name in ['lm_only_0_0_0', 'lm_only_1_0_0', 'lm_only_120_0_0' , 'regular_120_50_200']:
    for exp_name in ['regular_120_50_200','lm_only_120_0_0']:
        print('#########################################################################')
        print('loading model [%0s]...'%exp_name)

        save_file = os.path.join('.', 'ckp', exp_name, exp_name)
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

        if exp_name == 'regular_120_50_200':
            print('#########################################################################')
            print('Conducting convergence expariment...')
            test_data_loader = Gen_Data_loader_text8(BATCH_SIZE,charmap,inv_charmap,SEQ_LENGTH)
            test_data_loader.create_batches(real_data_test_file)
            results = convergence_experiment(sess, generator, test_data_loader)
            print('Saving results...')
            np.save('SeqGan_' + exp_name + '_conv_results',results)


        print('#########################################################################')
        print('Start Language Model Evaluation...')
        test_data_loader = Gen_Data_loader_text8(BATCH_SIZE,charmap,inv_charmap,SEQ_LENGTH)
        test_data_loader.create_batches(real_data_test_file)
        BPC_direct = language_model_evaluation_direct(sess,generator, test_data_loader)
        BPC_approx = language_model_evaluation_by_approximation(sess, generator, test_data_loader)
        print("[%0s] BPC_direct = %f"%(exp_name,BPC_direct))
        print("[%0s] BPC_approx = %f" % (exp_name, BPC_approx))

if __name__ == '__main__':
    main()