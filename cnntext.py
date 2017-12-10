import tensorflow as tf

class cnntext:
    """
    这是个2014年的模型，原理也很简单，把定长的句子用不同的filter做卷积，得到的结果取max，在合并成一个向量，最后做一个softmax.
    同样参考了github上的代码
    """
    def __init__(self,num_class,sequence_length,num_filter,filter_sizes,embed_size,vocab_size,batch_size,
                 learning_rate,decay_rate,decay_step,clip_gradients=5.0,initializer=tf.random_normal_initializer(stddev=0.1)):
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.initializer = initializer
        self.num_class = num_class
        self.num_filters = num_filter
        self.filter_sizes = filter_sizes
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.clip_gradients = clip_gradients
        # 参考原文，每个不同尺寸的filter都有多个
        self.num_filters_total = self.num_filters*len(filter_sizes)

        self.input_x = tf.placeholder(tf.int32,[None,self.sequence_length],name="input_x")
        self.input_y = tf.placeholder(tf.int32,[None,],name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0,trainable=False,name="Epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step,tf.constant(1))

        self.initiate_weight()
        self.logits = self.inference()
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, 1, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

    def initiate_weight(self):
        with tf.name_scope("embed"):
            self.embed = tf.get_variable("embed",shape=[self.vocab_size,self.embed_size],initializer=self.initializer)
            self.W = tf.get_variable("W",shape=[self.num_filters_total,self.num_class],initializer=self.initializer)
            self.b = tf.get_variable("b",shape=[self.num_class])

    def inference(self):
        self.embed_words = tf.nn.embedding_lookup(self.embed,self.input_x)
        self.sentence_embed_expand = tf.expand_dims(self.embed_words,-1)

        pool_output = []
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" %filter_size):
                """
                关于filter可以参考官方文档，四维依次是高，宽，in_channel,out_channel。
                stride表示[batch，in_height，in_width，in_channel]，其中batch和in_channel必须为0
                padding 表示要不要进行边缘填充
                要具体理解conv，filter，pool，drop
                """
                filter = tf.get_variable("filter-%s"%filter_size,[filter_size,self.embed_size,1,self.num_filters],initializer=self.initializer)
                conv = tf.nn.conv2d(self.sentence_embed_expand,filter=filter,strides=[1,1,1,1],padding="VALID",name="conv")
                b=tf.get_variable("b-%s"%filter_size,[self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
                pooled = tf.nn.max_pool(h,ksize=[1,self.sequence_length-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID',name="pool")
                pool_output.append(pooled)
        self.h_pool = tf.concat(pool_output, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat,keep_prob=self.dropout_keep_prob)
        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop,self.W) + self.b
        return logits

    def loss(self,l2_lamba=0.0001):
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])*l2_lamba
            loss = loss+l2_loss
        return loss

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate,self.global_step,self.decay_step,self.decay_rate,staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val,global_step=self.global_step,learning_rate=learning_rate,optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op