from fasttext import fasttext
import pandas as pd
from nltk import word_tokenize
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics

data_combine_ = pd.read_csv('C:/Users\Administrator\Desktop/data4cc.csv')
test_data = pd.read_excel('D:/大学\大三上\数据挖掘\作业/test4cc.xlsx')
train_ = pd.read_excel('D:/大学\大三上\数据挖掘\作业/train4cc.xlsx')


def test(train_data):
    """
    为了简化操作，我会对每个句子做个预处理，把句子分成长度差不多的三类，然后截断。
    :return:null 
    """
    #########################################################################################
    num_classes = 3
    learning_rate = 0.01
    batch_size = 128
    decay_steps = 10000
    decay_rate = 0.9
    sequence_length = 38
    vocab_size = 31663
    embed_size = 100
    is_training = True
    fastText = fasttext(num_classes, learning_rate, batch_size, decay_steps, decay_rate, 2, vocab_size, embed_size,
                        is_training, sequence_length)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        i = 0
        input_x = []
        input_y = []
        while (i<400):
            a = []
            index = i % 19579
            a = train_data[index]
            input_x.append(a[1:])
            input_y.append(int(a[0]))
            i += 1
            if (i % 128 == 0):
                input_x = np.array(input_x, dtype=np.int32)
                input_y = np.array(input_y, dtype=np.int32)
                loss, acc, predict, _ = sess.run(
                    [fastText.loss_val, fastText.accuracy, fastText.prediction, fastText.train_op],
                    feed_dict={
                        fastText.labels: input_y,
                        fastText.sentence: input_x})
                # print("loss:",loss,"acc:",acc,"label:",input_y,"prediction:",predict)
                print("loss:", loss, "acc:", acc)
                input_x = []
                input_y = []
            if (i%128==0):
                saver = tf.train.Saver()
                saver.save(sess,'D:/tensorflow/fasttext/model.ckpt')
        sess.close()

def load(train_data):
    num_classes = 3
    learning_rate = 0.01
    batch_size = 128
    decay_steps = 10000
    decay_rate = 0.9
    sequence_length = 38
    vocab_size = 31663
    embed_size = 100
    is_training = True
    fastText = fasttext(num_classes, learning_rate, batch_size, decay_steps, decay_rate, 2, vocab_size, embed_size,
                        is_training, sequence_length)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess,'D:/tensorflow/fasttext/model.ckpt')
        i = 0
        input_x = []
        input_y = []
        while (True):
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
                loss, acc, predict, _ = sess.run(
                    [fastText.loss_val, fastText.accuracy, fastText.prediction, fastText.train_op],
                    feed_dict={
                        fastText.labels: input_y,
                        fastText.sentence: input_x})
                # print("loss:",loss,"acc:",acc,"label:",input_y,"prediction:",predict)
                print("loss:", loss, "acc:", acc)
                input_x = []
                input_y = []
            if (i%128==0):
                saver = tf.train.Saver()
                saver.save(sess,'D:/tensorflow/fasttext/model.ckpt')

def predict(data_pre):
    data = []
    num_classes = 3
    learning_rate = 0.01
    batch_size = 1
    decay_steps = 10000
    decay_rate = 0.9
    sequence_length = 38
    vocab_size = 31663
    embed_size = 100
    is_training = True
    fastText = fasttext(num_classes, learning_rate, batch_size, decay_steps, decay_rate, 2, vocab_size, embed_size,
                        False, sequence_length)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, 'D:/tensorflow/fasttext/model.ckpt')
        for i in range(len(data_pre)):
            if len(data_pre[i])==38:
                a = data_pre[i]
            else:
                a = data_pre[i][1:]
            input_x = np.array([a])
            logit = sess.run(fastText.logits,feed_dict={fastText.sentence: input_x})
            print(logit)
            data.append(logit[0])
        print("预测结束")
        return data

def main():
    cv = []
    pred_full_test = np.zeros([test_data.shape[0],3])
    pred_train = np.zeros([train_.shape[0], 3])
    kf = KFold(shuffle=True,n_splits=3,random_state=2017)
    for val_index,dva_index in kf.split(train_.values):
        # print(val_index)
        # print(len(val_index))
        # print(dva_index)
        # print(len(dva_index))
        train_val,train_dva = train_.values[val_index],train_.values[dva_index]
        train_y = train_.author.values[dva_index]
        test(train_val)
        print("训练结束")
        data1 = predict(train_dva)
        print(data1)
        pred_train[dva_index,:] = data1
        data2 = predict(test_data.values)
        pred_full_test+=data2
        cv.append(metrics.log_loss(train_y,data1))
    print(cv)
    pred_full_test/=3
    pred_train = pd.DataFrame(pred_train)
    pred_full_test = pd.DataFrame(pred_full_test)
    result = pd.concat([pred_train,pred_full_test],axis=0)
    result.to_csv('C:/Users\Administrator\Desktop/fasttext_pred.csv')
    #test()
    # load()


if __name__ == '__main__':
    main()