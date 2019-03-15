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
import argparse
# from tqdm import tqdm

def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    print('Generating samples...')
    # Generate Samples
    generated_samples = []
    for _ in list(range(int(generated_num / batch_size))):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

def generate_real_data_samples(sess, trainable_model, batch_size, generated_num, output_file, inv_map, token_type):
    # Generate Samples
    print('Generating real data samples...')

    if token_type == 'char':
        seperator = ''
    elif token_type == 'word':
        seperator = ' '
    else:
        raise TypeError

    generated_samples = []
    for _ in list(range(int(generated_num / batch_size))):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = seperator.join([inv_map[x] for x in poem]) + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)

def split_text8(text8_orig_path):

    print('spliting text8 to train and test sets...')

    text8_train_path = text8_orig_path + '-train'
    text8_valid_path = text8_orig_path + '-valid'
    text8_test_path = text8_orig_path + '-test'

    # find each split size
    with open(text8_orig_path) as f:
        text8_size = len(f.read())
    assert text8_size == 100000000

    train_size = int(0.9 * text8_size)
    valid_size = int(0.05 * text8_size)
    test_size = int(0.05 * text8_size)

    with open(text8_orig_path,'r') as f_orig, \
            open(text8_train_path,'w') as f_train, \
            open(text8_valid_path, 'w') as f_valid, \
            open(text8_test_path,'w') as f_test:
        f_train.write(f_orig.read(train_size))
        f_valid.write(f_orig.read(valid_size))
        f_test.write(f_orig.read(test_size))

    return

def create_real_data_dict(data_file, dict_file, token_type):

    if not os.path.exists(dict_file): #create dict
        with open(data_file, 'r') as f:
            all_data = f.read()

        if token_type == 'char':
            counts = collections.Counter(char for char in all_data)
        elif token_type == 'word':
            all_data = all_data.replace('\n','<eos>')
            counts = collections.Counter(all_data.split(' '))

        map = {}
        inv_map = []

        for token, count in counts.most_common(200000):
            if token not in map:
                map[token] = len(inv_map)
                inv_map.append(token)

        #save dict
        with open(dict_file,'w') as f:
            f.write(json.dumps(map))

    else: # load dict
        with open(dict_file, 'r') as f:
            map = json.loads(f.read())

        inv_map = [None] * len(map)
        for key in list(map.keys()):
            inv_map[int(map[key])] = str(key)

    return map, inv_map
    

def main(FLAGS):
    #########################################################################################
    #  Generator  Hyper-parameters
    ######################################################################################
    EMB_DIM = FLAGS.gen_emb_dim # 32  # embedding dimension
    HIDDEN_DIM = FLAGS.gen_hidden_dim # 32  # hidden state dimension of lstm cell
    SEQ_LENGTH = FLAGS.seq_len # 20  # sequence length
    START_TOKEN = 0
    PRE_EPOCH_NUM = FLAGS.gen_pretrain_epoch_num  # 120 # supervise (maximum likelihood estimation) epochs for generator
    DISC_PRE_EPOCH_NUM = FLAGS.dis_pretrain_epoch_num  # 50 # supervise (maximum likelihood estimation) epochs for descriminator
    SEED = 88
    BATCH_SIZE = FLAGS.batch_size #64
    gen_dropout_keep_prob = FLAGS.gen_dropout_keep_prob # 0.75

    #########################################################################################
    #  Discriminator  Hyper-parameters
    #########################################################################################
    dis_embedding_dim = FLAGS.dis_emb_dim # 64
    dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
    dis_dropout_keep_prob = 0.75
    dis_l2_reg_lambda = 0.2
    dis_batch_size = FLAGS.batch_size #64

    #########################################################################################
    #  Basic Training Parameters
    #########################################################################################
    EXPERIMENT_NAME = FLAGS.experiment_name
    TOTAL_BATCH = FLAGS.num_epochs  # 200 #num of adversarial epochs
    positive_file = 'save/real_data_%0s.txt'%EXPERIMENT_NAME
    negative_file = 'save/generator_sample_%0s.txt'%EXPERIMENT_NAME
    eval_file = "save/eval_file_%0s"%EXPERIMENT_NAME
    generated_num = 10000  # 10000

    #########################################################################################
    #  Data configurations
    #########################################################################################
    use_real_world_data = True
    real_data_file_path = FLAGS.dataset_path # './data/text8/text8'
    dataset_name = os.path.basename(real_data_file_path)
    base_token = FLAGS.base_token # 'char'


    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    if use_real_world_data:

        real_data_train_file = real_data_file_path + '-train'
        real_data_valid_file = real_data_file_path + '-valid'
        real_data_test_file = real_data_file_path + '-test'
        real_data_dict_file = real_data_file_path + '-{}-dict.json'.format(base_token)

        if not os.path.exists(real_data_train_file):
            split_text8(real_data_file_path)

        map, inv_map = create_real_data_dict(real_data_train_file, real_data_dict_file, base_token)
        vocab_size = len(map)

        if dataset_name == 'text8' and base_token == 'char':
            assert vocab_size == 27 # SORRY FOR THE HARD CODING
        elif dataset_name == 'ptb' and base_token == 'word':
            assert vocab_size == 10001 # SORRY FOR THE HARD CODING
        elif dataset_name == 'toy' and base_token == 'word':
            assert vocab_size == 9 # SORRY FOR THE HARD CODING
        elif dataset_name == 'wt2' and base_token == 'word':
            assert vocab_size == 33279 # SORRY FOR THE HARD CODING
        else:
            raise TypeError

        gen_data_loader = Gen_Data_loader_text(BATCH_SIZE, map, inv_map, seq_len=SEQ_LENGTH, token_type=base_token)
        dis_data_loader = Dis_dataloader_text(BATCH_SIZE, map, inv_map, seq_len=SEQ_LENGTH, token_type=base_token)

    else:
        gen_data_loader = Gen_Data_loader(BATCH_SIZE)
        likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
        vocab_size = 5000
        dis_data_loader = Dis_dataloader(BATCH_SIZE)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, dropout_keep_prob=gen_dropout_keep_prob)

    if not use_real_world_data:
        target_params = pickle.load(open('save/target_params.pkl'))
        target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    saver = tf.train.Saver(tf.trainable_variables(),max_to_keep=999999)
    sess.run(tf.global_variables_initializer())

    if use_real_world_data:
        # gen_data_loader.create_batches(real_data_train_file)
        pass
    else:
        # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
        generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
        gen_data_loader.create_batches(positive_file)

    log = open('save/experiment-log.txt', 'w')
    #  pre-train generator
    print('Start pre-training...')
    log.write('pre-training...\n')
    for epoch in range(PRE_EPOCH_NUM):
        print("start epoch %0d" % epoch)

        if epoch % FLAGS.save_each_epochs == 0:
            print('#########################################################################')
            print('saving model...')
            save_file = os.path.join('.', 'ckp', EXPERIMENT_NAME + '_pretrain_epoch_%0d'%epoch , EXPERIMENT_NAME + '_pretrain_epoch_%0d'%epoch)
            saver.save(sess, save_file)

        if use_real_world_data:
            gen_data_loader.create_batches(real_data_train_file,limit_num_samples=generated_num)

        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 1 == 0:
            if use_real_world_data:
                generate_real_data_samples(sess, generator, BATCH_SIZE, generated_num, eval_file + "_epoch_%0d.txt"%epoch ,inv_map, base_token)
                test_loss = 0 # FIXME - TEMP
            else:
                generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
                likelihood_data_loader.create_batches(eval_file)
                test_loss = target_loss(sess, target_lstm, likelihood_data_loader)

            print('pre-train epoch ', epoch, 'test_loss ', test_loss)
            buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
            log.write(buffer)

    print('Start pre-training discriminator...')
    # Train 3 epoch on the generated data and do this for 50 times
    for epoch in range(DISC_PRE_EPOCH_NUM):
        print("start epoch %0d"%epoch)
        if use_real_world_data:
            generate_real_data_samples(sess, generator, BATCH_SIZE, generated_num , negative_file,inv_map, base_token)
            dis_data_loader.load_train_data(real_data_train_file, negative_file)
        else:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _ = sess.run(discriminator.train_op, feed)

    rollout = ROLLOUT(generator, 0.8)

    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        print("start epoch %0d" % total_batch)

        if total_batch % FLAGS.save_each_epochs == 0:
            print('#########################################################################')
            print('saving model...')
            save_file = os.path.join('.', 'ckp', EXPERIMENT_NAME + '_epoch_%0d'%total_batch , EXPERIMENT_NAME + '_epoch_%0d'%total_batch)
            saver.save(sess, save_file)

        for it in range(1):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)

        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            if not use_real_world_data:
                generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
                likelihood_data_loader.create_batches(eval_file)
                test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
                print('total_batch: ', total_batch, 'test_loss: ', test_loss)
                log.write(buffer)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        for _ in range(5):

            if use_real_world_data:
                generate_real_data_samples(sess, generator, BATCH_SIZE, generated_num, negative_file, inv_map, base_token)
                dis_data_loader.load_train_data(real_data_train_file, negative_file)
            else:
                generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
                dis_data_loader.load_train_data(positive_file, negative_file)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)


    print('#########################################################################')
    print('saving model...')
    save_file = os.path.join('.','ckp',EXPERIMENT_NAME,EXPERIMENT_NAME)
    saver.save(sess, save_file)

    #
    # print '#########################################################################'
    # print 'Start Language Model Evaluation...'
    # test_data_loader = Gen_Data_loader_text(BATCH_SIZE,map,inv_map)
    # test_data_loader.create_batches(real_data_test_file)
    # language_model_evaluation(sess,generator, test_data_loader)

    log.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SeqGAN Train for real text datasets")

    ######################################################################################
    #  General
    ######################################################################################
    parser.add_argument('experiment_name', type=str, help='experiment name')
    parser.add_argument('--dataset_path', type=str, default='./data/text8/text8',  help='dataset path', choices=['./data/text8/text8', './data/ptb/ptb', './data/toy/toy', './data/wt2/wt2'])
    parser.add_argument('--base_token', type=str, default='char', help='base token', choices=['char', 'word'])
    parser.add_argument('--num_epochs', type=int, default=200, help='number of adversarial epochs [200]')
    parser.add_argument('--seq_len', type=int, default=20, help='sequence length (must be >= 20 to fit disc arc) [20]')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size [64]')
    parser.add_argument('--gpu_inst', type=str, default='', help='choose GPU instance. empty string == run on CPU []')
    parser.add_argument('--save_each_epochs', type=int, default=999999, help='save model each X epochs [999999]')

    ######################################################################################
    #  Generator  Hyper-parameters
    ######################################################################################
    parser.add_argument('--gen_emb_dim', type=int, default=32, help='generator embedding dimension [32]')
    parser.add_argument('--gen_hidden_dim', type=int, default=32, help='hidden state dimension of lstm cell [32]')
    parser.add_argument('--gen_pretrain_epoch_num', type=int, default=120, help='supervise (maximum likelihood estimation) epochs for generator [120]')
    parser.add_argument('--gen_dropout_keep_prob', type=float, default=.75, help='dropout keep probability [0.75]')

    #########################################################################################
    #  Discriminator  Hyper-parameters
    #########################################################################################
    parser.add_argument('--dis_emb_dim', type=int, default=64, help='discriminator embedding dimension [64]')
    parser.add_argument('--dis_pretrain_epoch_num', type=int, default=50, help='supervise (maximum likelihood estimation) epochs for descriminator [50]')

    FLAGS = parser.parse_args()

    #choose GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_inst

    #check valid name
    if os.path.isdir(os.path.join('ckp',FLAGS.experiment_name)):
        raise NameError("experiment_name [%0s] already exists - choose another one!")

    # print FLAGS
    args_dict = vars(FLAGS)
    config_file = os.path.join('ckp','config_' + FLAGS.experiment_name + '.txt')
    if not os.path.isdir('ckp'):
        os.mkdir('ckp')
    with open(config_file,'w') as f:
        for arg in args_dict.keys():
            s = "%0s :\t\t\t%0s"%(arg,str(args_dict[arg]))
            print(s)
            f.write(s + '\n')


    # run
    main(FLAGS)
