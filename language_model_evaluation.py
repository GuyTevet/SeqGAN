import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Gen_Data_loader_text, Dis_dataloader, Dis_dataloader_text
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

# FIXME - hard coded params
BATCH_SIZE = 64
SEED = 88
START_TOKEN = 0
use_real_world_data = True
SEQ_LENGTH = 1000 #200


def restore_param_from_config(config_file,param):
    with open(config_file,'r') as f:
        line = None
        while line != '':
            line = f.readline()
            if line.startswith(param):
                return line.replace(param,'').replace(':','').strip()

    raise ValueError('param was not found')


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

def main(FLAGS):

    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    # if not use_real_world_data:
    #     target_params = pickle.load(open('save/target_params.pkl'))
    #     target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    # discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
    #                             filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    # # tf.reset_default_graph()
    # generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # # sess.run(tf.global_variables_initializer())

    experiments_list = [exp for exp in os.listdir('ckp') if os.path.isdir(os.path.join('ckp',exp))]
    if FLAGS.epoch_exp:
        experiments_list.sort(key=lambda x: int(x.split('_epoch_')[-1]))
        stats = np.zeros([2,len(experiments_list)],dtype=np.float32)

    for i, exp_name in enumerate(experiments_list):
        print('#########################################################################')
        print('loading model [%0s]...'%exp_name)

        if FLAGS.epoch_exp:
            config = os.path.join('ckp', 'config_' + exp_name.split('_epoch_')[0] + '.txt')
        else:
            config = os.path.join('ckp', 'config_' + exp_name + '.txt')

        # restore generator arch
        try:
            EMB_DIM = int(restore_param_from_config(config, param= 'gen_emb_dim'))
            HIDDEN_DIM = int(restore_param_from_config(config, param= 'gen_hidden_dim'))
        except:
            EMB_DIM = 32
            HIDDEN_DIM = 32
            print("WARNING: CONFIG FILE WAS NOT FOUND - USING DEFAULT CONFIG")
        
        try:
            TOKEN_TYPE = restore_param_from_config(config, param= 'base_token')
            DATA_PATH = restore_param_from_config(config, param='dataset_path')
        except ValueError:
            TOKEN_TYPE = 'char'
            DATA_PATH = './data/text8/text8'
            print("WARNING: NEW CONFIGURATION WAS NOT FOUND - EVALUATING CHAR-BASED")

        try:
            NUM_LAYERS = int(restore_param_from_config(config, param= 'gen_num_recurrent_layers'))
        except ValueError:
            NUM_LAYERS = 1
            print("WARNING: NUM LAYERS NOT FOUND - USING 1")

        if use_real_world_data:
            # split to train-valid-test
            real_data_train_file = DATA_PATH + '-train'
            real_data_valid_file = DATA_PATH + '-valid'
            real_data_test_file = DATA_PATH + '-test'
            real_data_dict_file = DATA_PATH + '-dict.json'
            if not os.path.exists(real_data_train_file):
                split_text8(DATA_PATH)
            charmap, inv_charmap = create_real_data_dict(real_data_train_file, real_data_dict_file,TOKEN_TYPE)
            vocab_size = len(charmap)
        else:
            gen_data_loader = Gen_Data_loader(BATCH_SIZE)
            likelihood_data_loader = Gen_Data_loader(BATCH_SIZE)  # For testing
            vocab_size = 5000
            dis_data_loader = Dis_dataloader(BATCH_SIZE)

        tf.reset_default_graph()
        generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN,num_recurrent_layers=NUM_LAYERS)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # sess.run(tf.global_variables_initializer())

        # restore weights
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

        # if exp_name == 'regular_120_50_200':
        #     print('#########################################################################')
        #     print('Conducting convergence expariment...')
        #     test_data_loader = Gen_Data_loader_text(BATCH_SIZE,charmap,inv_charmap,SEQ_LENGTH)
        #     test_data_loader.create_batches(real_data_test_file)
        #     results = convergence_experiment(sess, generator, test_data_loader)
        #     print('Saving results...')
        #     np.save('SeqGan_' + exp_name + '_conv_results',results)

        print('###')
        print('Start Language Model Evaluation...')
        test_data_loader = Gen_Data_loader_text(BATCH_SIZE,charmap,inv_charmap,SEQ_LENGTH,TOKEN_TYPE)
        if FLAGS.test:
            test_data_loader.create_batches(real_data_test_file)
            print("USING %s TEST SET"%TOKEN_TYPE.upper())
        else:
            test_data_loader.create_batches(real_data_valid_file)
            # test_data_loader.create_batches(real_data_train_file)
            print("USING %s VALID SET"%TOKEN_TYPE.upper())
        BPC_direct = language_model_evaluation_direct(sess,generator, test_data_loader)
        print("[%0s] BPC_direct = %f"%(exp_name,BPC_direct))

        if FLAGS.test:
            BPC_approx = language_model_evaluation_by_approximation(sess, generator, test_data_loader)
            print("[%0s] BPC_approx = %f" % (exp_name, BPC_approx))

        if FLAGS.epoch_exp:
            stats[0, i] = int(exp_name.split('_epoch_')[-1])
            stats[1, i] = BPC_direct

    if FLAGS.epoch_exp:
        np.save('direct_results_epoch_exp',stats)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SeqGAN LM Test on Text8/PTB")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--epoch_exp', action='store_true')
    FLAGS = parser.parse_args()

    main(FLAGS)