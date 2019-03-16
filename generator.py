import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class Generator(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token,
                 reward_gamma=0.95,
                 dropout_keep_prob = 1., num_recurrent_layers=1):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.learning_rate = tf.placeholder(tf.float32, [])
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.d_params = []
        self.temperature = 1.0
        self.grad_clip = 5.0
        self.dropout_keep_prob = dropout_keep_prob
        self.num_recurrent_layers = num_recurrent_layers

        self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))

        with tf.variable_scope('generator'):
            self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.g_params.append(self.g_embeddings)
            # self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)   # maps h_tm1 to h_t for generator
            self.g_recurrent_unit = [self.create_recurrent_unit(self.g_params, i) for i in range(self.num_recurrent_layers)]
            self.g_output_unit = self.create_output_unit(self.g_params)  # maps h_t to o_t (output token logits)

        # placeholder definition
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length]) # sequence of tokens generated by generator
        self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.sequence_length]) # get from rollout policy and discriminator

        self.is_lm_eval = tf.placeholder(tf.bool)

        # processed for batch
        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x), perm=[1, 0, 2])  # seq_length x batch_size x emb_dim

        # Initial states
        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = [tf.stack([self.h0, self.h0])] * self.num_recurrent_layers

        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
            h_t = []
            curr_x = x_t
            for layer_i in range(self.num_recurrent_layers):
                h_t.append(self.g_recurrent_unit[layer_i](curr_x, h_tm1[layer_i]))  # hidden_memory_tuple
                curr_hidden_state, curr_prev = tf.unstack(h_t[-1])
                curr_hidden_state_drop = tf.nn.dropout(curr_hidden_state,keep_prob=self.dropout_keep_prob)
                h_t_droped = tf.stack((curr_hidden_state_drop, curr_prev))
                curr_x = curr_hidden_state_drop
            o_t = self.g_output_unit(h_t_droped)  # batch x vocab , logits not prob
            # h_t = self.g_recurrent_unit[0](x_t, h_tm1[0])  # hidden_memory_tuple
            # h_t_droped = tf.nn.dropout(h_t, keep_prob=self.dropout_keep_prob)
            # o_t = self.g_output_unit(h_t_droped)  # batch x vocab , logits not prob
            log_prob = tf.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.num_emb, 1.0, 0.0),
                                                             tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, gen_o, gen_x

        _, _, _, self.gen_o, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, gen_o, gen_x))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length

        # supervised pretraining for generator
        g_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
            h_t = []
            curr_x = x_t
            for layer_i in range(self.num_recurrent_layers):
                h_t.append(self.g_recurrent_unit[layer_i](curr_x, h_tm1[layer_i]))  # hidden_memory_tuple
                curr_hidden_state, curr_prev = tf.unstack(h_t[-1])
                curr_hidden_state_drop = tf.nn.dropout(curr_hidden_state,keep_prob=self.dropout_keep_prob)
                h_t_droped = tf.stack((curr_hidden_state_drop, curr_prev))
                curr_x = curr_hidden_state_drop
            o_t = self.g_output_unit(h_t_droped)
            # h_t = self.g_recurrent_unit[0](x_t, h_tm1)  # hidden_memory_tuple
            # h_t_droped = tf.nn.dropout(h_t, keep_prob=self.dropout_keep_prob)
            # o_t = self.g_output_unit(h_t_droped)  # batch x vocab , logits not prob
            g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # batch x vocab_size
            x_tp1 = ta_emb_x.read(i)
            return i + 1, x_tp1, h_t, g_predictions

        _, _, _, self.g_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, g_predictions))

        self.g_predictions = tf.transpose(self.g_predictions.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size
        self.g_pred_argmax = tf.argmax(self.g_predictions,axis=2) # batch_size x seq_length
        # self.g_pred_one_hot = tf.one_hot(self.g_pred_argmax,depth=self.g_predictions.shape[2]) # batch_size x seq_length x vocab_size

        # pretraining loss
        self.pretrain_loss = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
            )
        ) / (self.sequence_length * self.batch_size)

        # training updates
        pretrain_opt = self.g_optimizer(self.learning_rate)

        self.pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pretrain_loss, self.g_params), self.grad_clip)
        self.pretrain_updates = pretrain_opt.apply_gradients(list(zip(self.pretrain_grad, self.g_params)))

        #######################################################################################################
        #  Unsupervised Training
        #######################################################################################################
        self.g_loss = -tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                    tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
                ), 1) * tf.reshape(self.rewards, [-1])
        )

        g_opt = self.g_optimizer(self.learning_rate)

        self.g_grad, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.g_params), self.grad_clip)
        self.g_updates = g_opt.apply_gradients(list(zip(self.g_grad, self.g_params)))

        #######################################################################################################
        #  LM evaluation
        #######################################################################################################

        self.g_pred_sampled = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        self.g_pred_for_eval = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        def _g_lm_eval(i, x_t, h_tm1, g_pred_for_eval, g_pred_sampled):
            h_t = []
            curr_x = x_t
            for layer_i in range(self.num_recurrent_layers):
                h_t.append(self.g_recurrent_unit[layer_i](curr_x, h_tm1[layer_i]))  # hidden_memory_tuple
                curr_hidden_state, curr_prev = tf.unstack(h_t[-1])
                # curr_hidden_state_drop = tf.nn.dropout(curr_hidden_state,keep_prob=self.dropout_keep_prob)
                # h_t_droped = tf.stack((curr_hidden_state_drop, curr_prev))
                curr_x = curr_hidden_state
            o_t = self.g_output_unit(h_t[-1])
            # h_t = self.g_recurrent_unit[0](x_t, h_tm1)  # hidden_memory_tuple #no dropout here
            # o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
            log_prob = tf.log(tf.nn.softmax(o_t))
            g_pred_for_eval = g_pred_for_eval.write(i, tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            g_pred_sampled = g_pred_sampled.write(i, next_token)
            x_tp1 = ta_emb_x.read(i)
            return i + 1, x_tp1, h_t, g_pred_for_eval, g_pred_sampled

        _, _, _, self.g_pred_for_eval, self.g_pred_sampled = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_lm_eval,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, self.g_pred_for_eval, self.g_pred_sampled))

        self.g_pred_for_eval = tf.transpose(self.g_pred_for_eval.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size
        self.g_pred_sampled = self.g_pred_sampled.stack()  # seq_length x batch_size
        self.g_pred_sampled = tf.transpose(self.g_pred_sampled, perm=[1, 0])  # batch_size x seq_length

        self.g_pred_one_hot = tf.one_hot(self.g_pred_sampled,self.num_emb, 1.0, 0.0)



    def generate(self, sess):
        outputs = sess.run(self.gen_x)
        return outputs

    def pretrain_step(self, sess, x, lr):
        outputs = sess.run([self.pretrain_updates, self.pretrain_loss], feed_dict={self.x: x, self.learning_rate: lr})
        return outputs

    def language_model_eval_step(self, sess, x):
        # outputs = sess.run([self.g_pred_one_hot, self.g_predictions], feed_dict={self.x: x})
        outputs = sess.run([self.g_pred_one_hot, self.g_pred_for_eval], feed_dict={self.x: x})
        return outputs

    def init_matrix(self, shape):
        # return tf.random_uniform(shape, minval=-0.05, maxval=0.05)
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)

    def create_recurrent_unit(self, params, num=0):

        if num == 0:
            self.Wi = [None] * self.num_recurrent_layers
            self.Ui = [None] * self.num_recurrent_layers
            self.bi = [None] * self.num_recurrent_layers
            self.Wf = [None] * self.num_recurrent_layers
            self.Uf = [None] * self.num_recurrent_layers
            self.bf = [None] * self.num_recurrent_layers
            self.Wog = [None] * self.num_recurrent_layers
            self.Uog = [None] * self.num_recurrent_layers
            self.bog = [None] * self.num_recurrent_layers
            self.Wc = [None] * self.num_recurrent_layers
            self.Uc = [None] * self.num_recurrent_layers
            self.bc = [None] * self.num_recurrent_layers

        # Weights and Bias for input and hidden tensor
        self.Wi[num] = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]), name='Wi_{}'.format(num))
        self.Ui[num] = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]), name='Ui_{}'.format(num))
        self.bi[num] = tf.Variable(self.init_matrix([self.hidden_dim]), name='bi_{}'.format(num))

        self.Wf[num] = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]), name='Wf_{}'.format(num))
        self.Uf[num] = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]), name='Uf_{}'.format(num))
        self.bf[num] = tf.Variable(self.init_matrix([self.hidden_dim]), name='bf_{}'.format(num))

        self.Wog[num] = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]), name='Wog_{}'.format(num))
        self.Uog[num] = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]), name='Uog_{}'.format(num))
        self.bog[num] = tf.Variable(self.init_matrix([self.hidden_dim]), name='bog_{}'.format(num))

        self.Wc[num] = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]), name='Wc_{}'.format(num))
        self.Uc[num] = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]), name='Uc_{}'.format(num))
        self.bc[num] = tf.Variable(self.init_matrix([self.hidden_dim]), name='bc_{}'.format(num))


        params.extend([
            self.Wi[num], self.Ui[num], self.bi[num],
            self.Wf[num], self.Uf[num], self.bf[num],
            self.Wog[num], self.Uog[num], self.bog[num],
            self.Wc[num], self.Uc[num], self.bc[num]])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi[num]) +
                tf.matmul(previous_hidden_state, self.Ui[num]) + self.bi[num]
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf[num]) +
                tf.matmul(previous_hidden_state, self.Uf[num]) + self.bf[num]
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog[num]) +
                tf.matmul(previous_hidden_state, self.Uog[num]) + self.bog[num]
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc[num]) +
                tf.matmul(previous_hidden_state, self.Uc[num]) + self.bc[num]
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self, params):
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.num_emb]))
        self.bo = tf.Variable(self.init_matrix([self.num_emb]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer(*args, **kwargs)
