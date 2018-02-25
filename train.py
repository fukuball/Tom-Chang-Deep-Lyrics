#coding:utf-8
import tensorflow as tf
import sys,time
import numpy as np
import cPickle

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1


file = sys.argv[1]
data = open(file,'r').read()
data = data.decode('utf-8')
chars = list(set(data)) # Char vocabulary

data_size, _vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, _vocab_size)
char_to_idx = { ch:i for i,ch in enumerate(chars) }
idx_to_char = { i:ch for i,ch in enumerate(chars) }

model_path = 'model/Model' # The path of model that need to save or load

cPickle.dump((char_to_idx, idx_to_char), open(model_path+'.voc','w'), protocol=cPickle.HIGHEST_PROTOCOL)

class Config(object):
    def __init__(self):
        self.init_scale = 0.04
        self.learning_rate = 0.001
        self.max_grad_norm = 15
        self.num_layers = 3
        self.num_steps = 25 # number of steps to unroll the RNN for
        self.hidden_size = 1000 # size of hidden layer of neurons
        self.iteration = 50
        self.save_freq = 5 # The step (counted by the number of iterations) at which the model is saved to hard disk.
        self.keep_prob = 0.5
        self.batch_size = 32
        self.vocab_size = _vocab_size

def get_config():
    return Config()

context_of_idx = [char_to_idx[ch] for ch in data]

def data_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)] # data 的 shape 是 (batch_size, batch_len)，每一行是連貫的一段，一次可輸入多個段

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1] # y 就是 x 的錯一位，即下一個詞
        yield (x, y)

class Model(object):
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        self.lr = config.learning_rate

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps]) # 宣告輸入變數 x, y


        lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=False)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=False)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size]) # size 是 wordembedding 的維度
            inputs = tf.nn.embedding_lookup(embedding, self._input_data) # 返回一個 tensor，shape 是 (batch_size, num_steps, size)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)


        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state) # inputs[:, time_step, :] 的 shape 是 (batch_size, size)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        """
        output 是一個 list，n*(batch_size, hidden_size)，tf.concat(outputs, 1) 返回一個矩陣 (batch_size, n*hidden_size)
        reshape(..., [-1, size])
        """
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b # logits 應該是 (batch_size*time_step, vocab_size)，順序是第一段的第一個詞、第二個詞...，然後是第二段的第一個詞，...
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state
        self._logits = logits

        if not is_training:
            self._prob = tf.nn.softmax(logits)
            return

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def train_op(self):
        return self._train_op


def run_epoch(session, m, data, eval_op, state=None, is_generation=False):
    """Runs the model on the given data."""
    if not is_generation:
        epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
        start_time = time.time()
        costs = 0.0
        iters = 0
        state = m.initial_state.eval()
        for step, (x, y) in enumerate(data_iterator(data, m.batch_size,
                                                        m.num_steps)):
            cost, state, _ = session.run([m.cost, m.final_state, eval_op], # x 和 y 的 shape 都是 (batch_size, num_steps)
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
            costs += cost
            iters += m.num_steps

            if step and step % (epoch_size // 10) == 0:
                print("%.2f perplexity: %.3f cost-time: %.2f s" %
                    (step * 1.0 / epoch_size, np.exp(costs / iters),
                     (time.time() - start_time)))
                start_time = time.time()

        return np.exp(costs / iters)
    else:
        x = data.reshape((1,1))
        prob, _state, _ = session.run([m._prob, m.final_state, eval_op],
                             {m.input_data: x,
                              m.initial_state: state})
        return prob, _state

def main(_):
    train_data = context_of_idx

    config = get_config()

    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        cPickle.dump(config, open(model_path+'.fig','w'), protocol=cPickle.HIGHEST_PROTOCOL)

        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = Model(is_training=True, config=config)

        tf.global_variables_initializer().run()

        model_saver = tf.train.Saver(tf.global_variables())

        for i in range(config.iteration):
            print("Training Epoch: %d ..." % (i+1))
            train_perplexity = run_epoch(session, m, train_data, m.train_op)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            if (i+1) % config.save_freq == 0:
                print 'model saving ...'
                model_saver.save(session, model_path+'-%d'%(i+1))
                print 'Done!'

if __name__ == "__main__":
    tf.app.run()
