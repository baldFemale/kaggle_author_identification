import cnntext
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics

data_combine_ = pd.read_csv('C:/Users\Administrator\Desktop/data4cc.csv')
test_data = pd.read_excel('D:/大学\大三上\数据挖掘\作业/test4cc.xlsx')
train_ = pd.read_excel('D:/大学\大三上\数据挖掘\作业/train4cc.xlsx')



def load(train_data):
    num_classes = 3
    learning_rate = 0.01
    batch_size = 128
    decay_steps = 20000
    decay_rate = 0.9
    sequence_length = 38
    vocab_size = 31663
    embed_size = 100
    is_training = True
    dropout_keep_prob = 1
    filter_sizes = [3, 4, 5]
    num_filters = 128
    textCNN = cnntext.cnntext(num_class=num_classes, sequence_length=sequence_length, num_filter=num_filters,
                              filter_sizes=filter_sizes,
                              embed_size=embed_size, vocab_size=vocab_size, batch_size=batch_size,
                              learning_rate=learning_rate,
                              decay_rate=decay_rate, decay_step=decay_steps, clip_gradients=5.0)
    loaded_graph = tf.Graph()
    with tf.Session() as sess:
        i = 22000
        input_x = []
        input_y = []
        saver = tf.train.Saver()
        saver.restore(sess,'D:/tensorflow/model.ckpt')
        while(True):
            a = []
            index = i % 19579
            for j in range(38):
                a.append(train_data['word2int_' + str(j)][index])
            input_x.append(a)
            input_y.append(int(train_data.author[index]))
            i += 1
            if (i % 128 == 0):
                input_x = np.array(input_x, dtype=np.int32)
                input_y = np.array(input_y, dtype=np.int32)
                loss, acc, predict, W_projection_value, _ = sess.run(
                    [textCNN.loss_val, textCNN.accuracy, textCNN.predictions, textCNN.W, textCNN.train_op],
                    feed_dict={textCNN.input_x: input_x, textCNN.input_y: input_y,
                               textCNN.dropout_keep_prob: dropout_keep_prob})
                # print("loss:",loss,"acc:",acc,"label:",input_y,"prediction:",predict)
                print("loss:", loss, "acc:", acc)
                input_x = []
                input_y = []
            if (i % 128 == 0):
                saver = tf.train.Saver()
                saver.save(sess, 'D:/tensorflow/model.ckpt')


def test(train_data):
    num_classes=3
    learning_rate=0.01
    batch_size=128
    decay_steps=20000
    decay_rate=0.9
    sequence_length=38
    vocab_size=31663
    embed_size=100
    is_training=True
    dropout_keep_prob=1
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
            index = i%19579
            a = train_data[index]
            input_x.append(a[1:])
            input_y.append(int(a[0]))
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

def predict(data_pre):
    data = []
    num_classes = 3
    learning_rate = 0.01
    batch_size = 128
    decay_steps = 20000
    decay_rate = 0.9
    sequence_length = 38
    vocab_size = 31663
    embed_size = 100
    is_training = True
    dropout_keep_prob = 1  # 0.5
    filter_sizes = [3, 4, 5]
    num_filters = 128
    textCNN = cnntext.cnntext(num_class=num_classes, sequence_length=sequence_length, num_filter=num_filters,
                              filter_sizes=filter_sizes,
                              embed_size=embed_size, vocab_size=vocab_size, batch_size=batch_size,
                              learning_rate=learning_rate,
                              decay_rate=decay_rate, decay_step=decay_steps, clip_gradients=5.0)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, 'D:/tensorflow/model.ckpt')
        for i in range(len(data_pre)):
            a = []
            if len(data_pre[i])==38:
                a = data_pre[i]
            else:
                a = data_pre[i][1:]
            input_x = np.array([a])
            logit = sess.run(textCNN.logits,feed_dict={textCNN.input_x: input_x,
                                                       textCNN.dropout_keep_prob:dropout_keep_prob})
            print(logit)
            data.append(logit[0])
        return data



def main():
    cv = []
    pred_full_test = np.zeros([test_data.shape[0], 3])
    pred_train = np.zeros([train_.shape[0], 3])
    kf = KFold(shuffle=True, n_splits=3, random_state=2017)
    for val_index, dva_index in kf.split(train_.values):
        # print(val_index)
        # print(len(val_index))
        # print(dva_index)
        # print(len(dva_index))
        train_val, train_dva = train_.values[val_index], train_.values[dva_index]
        train_y = train_.author.values[dva_index]
        test(train_val)
        print("训练结束")
        data1 = predict(train_dva)
        print(data1)
        pred_train[dva_index, :] = data1
        data2 = predict(test_data.values)
        pred_full_test += data2
        cv.append(metrics.log_loss(train_y, data1))
    print(cv)
    pred_full_test /= 3
    pred_train = pd.DataFrame(pred_train)
    pred_full_test = pd.DataFrame(pred_full_test)
    result = pd.concat([pred_train, pred_full_test], axis=0)
    result.to_csv('C:/Users\Administrator\Desktop/cnntext_pred.csv')

if __name__=='__main__':
    main()