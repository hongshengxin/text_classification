# -*- coding:utf-8 -*-

import tensorflow as tf

class TRNNConfig(object):
    """RNN配置参数"""
    # 模型参数
    embedding_dim=64
    seq_length=600
    num_classes = 10        # 类别数
    vocab_size = 5000       # 词汇表大小

    num_layers=2
    hidden_dim=128
    rnn='gru'

    dropout_keep_prob=0.8
    learning_rate=1e-3

    batch_size=64
    num_epochs=10000

    print_per_batch=100
    save_per_batch=10  # 每多少轮存入tensorboard
class TextRNN(object):
    def __init__(self,config):
        self.config=config
        self.input_x=tf.placeholder(tf.int32,shape=[None,self.config.seq_length],name="input_x")
        self.input_y=tf.placeholder(tf.int32,shape=[None,self.config.num_classes],name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.rnn()
    def rnn(self):
        """rnn模型"""

        def lstm_cell():
            return tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim,state_is_tuple=True)
        def gru_cell():
            return tf.nn.rnn_cell.GRUCell(self.config.hidden_dim)
        def dropout():
            if self.config.rnn=="lstm":
                cell=lstm_cell()
            else:
                cell=gru_cell()
            # return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.config.dropout_keep_prob)
        with tf.device('/cpu:0'):
            # embbedding=tf.get_variable(name="embedding",shape=[self.config.vocab_size,self.config.embedding_dim],
            #                            initializer=tf.initializers.random_normal)
            embbedding=tf.get_variable(name="embedding",shape=[self.config.vocab_size,self.config.embedding_dim])
            embbedding_inputs=tf.nn.embedding_lookup(embbedding,self.input_x)
        with tf.name_scope('rnn'):
            celss=[dropout() for _ in range(self.config.num_layers)]
            rnn_cell=tf.nn.rnn_cell.MultiRNNCell(celss,state_is_tuple=True)
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embbedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope("score"):
            """全连接层"""
            fc=tf.layers.dense(last,self.config.hidden_dim,name="fc1")
            fc=tf.contrib.layers.dropout(fc,self.keep_prob)
            fc=tf.nn.relu(fc)

            self.logits=tf.layers.dense(fc,self.config.num_classes,name='fc2')
            self.y_pred_cls=tf.argmax(tf.nn.softmax(self.logits),1)
        with tf.name_scope("optimize"):
            """损失函数"""
            cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.input_y)
            self.loss=tf.reduce_mean(cross_entropy)
            # self.optim=tf.train.AdamOptimizer()
            self.optim=tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        with tf.name_scope("accuracy"):
            """准确率的计算"""
            correct_pred=tf.equal(tf.argmax(self.input_y,1),self.y_pred_cls)
            self.acc=tf.reduce_mean(tf.cast(correct_pred,tf.int32))
if __name__ == '__main__':

    import tensorflow as tf
    import numpy as np
    config=TRNNConfig()
    rn=TextRNN(config)

    # input_ids = tf.placeholder(dtype=tf.int32, shape=[None,2])
    #
    # embedding = tf.Variable(np.identity(5, dtype=np.int32))
    # # input_embedding1 = tf.nn.embedding_lookup(embedding, input_ids)
    #
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # print(embedding.eval())
    # print("*"*100)
    # # print(sess.run(input_embedding1, feed_dict={input_ids:[1, 2, 3, 0, 3, 2, 1]}))
    # print("*"*100)
    # input_embedding = tf.nn.embedding_lookup(embedding, input_ids)
    # print(sess.run(input_embedding, feed_dict={input_ids: [[1, 2], [2, 1], [3, 3]]}))





