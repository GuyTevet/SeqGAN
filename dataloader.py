import numpy as np
import os

class Gen_Data_loader(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []

    def create_batches(self, data_file):
        self.token_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

class Gen_Data_loader_text8(Gen_Data_loader):

    def __init__(self, batch_size, charmap, inv_charmap):
        super(Gen_Data_loader_text8, self).__init__(batch_size)
        self.charmap, self.inv_charmap = charmap, inv_charmap
        self.seq_len = 20

    def create_batches(self, data_file):

        if os.path.exists(data_file + '.npy'):
            self.token_stream = np.load(data_file + '.npy')
        else:
            self.token_stream = []

            with open(data_file, 'r') as f:
                line = f.read(self.seq_len)
                while len(line) == self.seq_len:
                    tokens = [int(self.charmap[char]) for char in line]
                    assert len(tokens) == self.seq_len
                    self.token_stream.append(tokens)

                    line = f.read(self.seq_len)

            np.save(data_file,np.array(self.token_stream))

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

        print "done create_batches - [num_batch=%0d]"%self.num_batch

    def reset_pointer(self):
        self.shuffle()
        self.pointer = 0

    def shuffle(self):
        print "GEN shuffling data..."
        permut = np.random.permutation(self.token_stream.shape[0])
        self.token_stream = self.token_stream[permut]


class Dis_dataloader(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                positive_examples.append(parse_line)
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    negative_examples.append(parse_line)
        self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader_text8(Dis_dataloader):

    def __init__(self, batch_size, charmap, inv_charmap):
        super(Dis_dataloader_text8, self).__init__(batch_size)
        self.charmap, self.inv_charmap = charmap, inv_charmap
        self.seq_len = 20

    def load_train_data(self, positive_file, negative_file):

        # #LOAD NEGATIVE
        # positive file is constant while negative file is changed during train!
        # if os.path.exists(negative_file + '.npy'):
        #     negative_examples = np.load(negative_file + '.npy')
        # else:
        negative_examples = []

        # remove \n
        with open(negative_file, 'r') as f:
            all_negative = f.read()
        with open(negative_file, 'w') as f:
            f.write(all_negative.replace('\n',''))

        with open(negative_file, 'r') as f:
            line = f.read(self.seq_len)
            while len(line) == self.seq_len:
                tokens = [int(self.charmap[char]) for char in line]
                assert len(tokens) == self.seq_len
                negative_examples.append(tokens)

                line = f.read(self.seq_len)

        # np.save(negative_file,np.array(negative_examples))
        negative_examples = np.array(negative_examples)
        num_positive_samples = negative_examples.shape[0]


        #LOAD POSITIVE
        if os.path.exists(positive_file + '.npy'):
            positive_examples = np.load(positive_file + '.npy')
        else:
            positive_examples = []

            with open(positive_file, 'r') as f:
                line = f.read(self.seq_len)
                while len(line) == self.seq_len:
                    tokens = [int(self.charmap[char]) for char in line]
                    assert len(tokens) == self.seq_len
                    positive_examples.append(tokens)

                    line = f.read(self.seq_len)

            np.save(positive_file,np.array(positive_examples))

        #choose only num_positive_samples from them
        permut = np.random.permutation(positive_examples.shape[0])[:num_positive_samples]
        positive_examples = positive_examples[permut]

        # CONCAT
        negative_examples = np.array(negative_examples)
        positive_examples = np.array(positive_examples)
        assert negative_examples.shape == positive_examples.shape
        self.sentences = np.concatenate((positive_examples,negative_examples),axis=0)

        # Generate labels
        positive_labels = [[0, 1]] * positive_examples.shape[0]
        negative_labels = [[1, 0]] * negative_examples.shape[0]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # # Shuffle the data
        # print "DISC shuffling data..."
        # shuffle_indices = np.random.permutation(self.sentences.shape[0])
        # self.sentences = self.sentences[shuffle_indices]
        # self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0

        print "done create_batches - [num_batch=%0d]"%self.num_batch

    def reset_pointer(self):
        self.shuffle()
        self.pointer = 0

    def shuffle(self):
        print "DISC shuffling data..."
        shuffle_indices = np.random.permutation(self.sentences.shape[0])
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

