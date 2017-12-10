import cnntext
import tensorflow as tf
import numpy as np
import pandas as pd

train_data = pd.read_excel('D:\大学\大三上\数据挖掘\作业/train4cc.xlsx')

def str_int(x):
    a = []
    for i in x:
        a.append(int(i))
    return a

def test():
    num_classes=3
    learning_rate=0.01
    batch_size=128
    decay_steps=20000
    decay_rate=0.9
    sequence_length=38
    vocab_size=31663
    embed_size=100
    is_training=True
    dropout_keep_prob=1 #0.5
    filter_sizes=[3,4,5]
    num_filters=128
    textCNN=cnntext.cnntext(num_class=num_classes,sequence_length=sequence_length,num_filter=num_filters,filter_sizes=filter_sizes,
                    embed_size=embed_size,vocab_size=vocab_size,batch_size=batch_size,learning_rate=learning_rate,
                    decay_rate=decay_rate,decay_step=decay_steps,clip_gradients=5.0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        i=0
        input_x = []
        input_y = []
        while(True):
            a = []
            index = i%19579
            for j in range(38):
                a.append(train_data['word2int_'+str(j)][index])
            input_x.append(a)
            input_y.append(int(train_data.author[index]))
            i+=1
            if (i%128==0):
                input_x = np.array(input_x,dtype=np.int32)
                input_y = np.array(input_y,dtype=np.int32)
                loss,acc,predict,W_projection_value,_=sess.run([textCNN.loss_val,textCNN.accuracy,textCNN.predictions,textCNN.W,textCNN.train_op],
                                                           feed_dict={textCNN.input_x:input_x,textCNN.input_y:input_y,textCNN.dropout_keep_prob:dropout_keep_prob})
                # print("loss:",loss,"acc:",acc,"label:",input_y,"prediction:",predict)
                print("loss:", loss, "acc:", acc)
                input_x = []
                input_y = []
            if (i%128==0):
                saver = tf.train.Saver()
                saver.save(sess,'D:/tensorflow/model.ckpt')

def main():
    test()

if __name__=='__main__':
    main()