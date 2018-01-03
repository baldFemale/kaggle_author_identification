import tensorflow as tf
import numpy as np

class fasttext:
    """
    因为我安装不上fasttext，所以就参考了git上别人实现的源码，自己写一遍吧。
    我找不到facebook的文献，所以只能看git。但是git上又有很多不同版本的代码，里面的思想好像差了有点多。
    一些用nce_loss，一些用cross_entropy loss，我搞不太清楚。因为nce_loss是做embed里用的，cross_entropy loss是做分类的。
    但是根据我以前的经验，我决定用的是cross_entropy loss。
    但不管怎么说，fasttext是个很简单的模型：把ngram向量化，做个embed取平均然后做个softmax就行了。
    这个类不负责数据预处理，只做训练模型。
    """
    def __init__(self,label_size,learning_rate,batch_size,decay_step,decay_rate,num_sampled,vocab_size,embed_size,istraining,sentence_len):
        """初始化参数"""
        """不reset会报错，应该是在同一张图上进行了两次操作，所以会有变量已经存在的错误"""
        tf.reset_default_graph()
        self.label_size = label_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.istraing = istraining
        self.learning_rate = learning_rate
        self.sentence_len = sentence_len

        # 初始化placeholder
        self.sentence = tf.placeholder(tf.int32,[None,self.sentence_len],name='sentence')
        print(self.sentence.shape)
        self.labels = tf.placeholder(tf.int32,[None],name="labels")

        self.global_step = tf.Variable(0,trainable=False,name='Global_step')
        self.epoch_step = tf.Variable(0,trainable=False,name='Epoch_step')
        self.epoch_increment = tf.assign(self.epoch_step,tf.constant(1))
        self.decat_step,self.decay_rate = decay_step,decay_rate

        # self.epoch_step = tf.Variable(0, trainable=False, name='Epoch_step')
        self.instantiate_weight()
        self.logits = self.inference()
        if not istraining:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.prediction = tf.argmax(self.logits,axis=1,name="prediction")
        correct_prediction = tf.equal(tf.cast(self.prediction,tf.int32),self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name="Accuracy")


    def instantiate_weight(self):
        with tf.name_scope('embed'):
            self.Embedding = tf.get_variable(name='Embed',shape=[self.vocab_size,self.embed_size])
            self.W = tf.get_variable('W',[self.embed_size,self.label_size])
            self.b = tf.get_variable('b',[self.label_size])

    def inference(self):
        # 一样不懂
        sentence_embed = tf.nn.embedding_lookup(self.Embedding,self.sentence)
        self.sentence_embed = tf.reduce_mean(sentence_embed,axis=1)
        logit = tf.matmul(self.sentence_embed,self.W)+self.b
        return logit

    def loss(self):
        if self.istraing:
            labels = tf.reshape(self.labels, [-1])
            labels = tf.expand_dims(labels, 1)
            labels_one_hot = tf.one_hot(self.labels,self.label_size)
            # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot,logits=self.logits)
            # print(loss.shape)
            loss = tf.reduce_mean(tf.nn.nce_loss(weights=tf.transpose(self.W),
                                  biases=self.b,
                                  labels=labels,
                                  inputs=self.sentence_embed,
                                  num_sampled=self.num_sampled,
                                  num_classes=self.label_size,partition_strategy='div'
                                  ))
        else:
            labels_one_hot = tf.one_hot(self.labels,self.label_size)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot,logits=self.logits)
            print('loss0:',loss)
            loss = tf.reduce_mean(loss,axis=1)
            print("loss1:",loss)
        return loss

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate,self.global_step,self.decat_step,self.decay_rate,staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val,global_step = self.global_step,learning_rate=learning_rate,optimizer="Adam")
        return train_op
