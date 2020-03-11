from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Conv1D, MaxPool1D
from eunjeon import Mecab
from chanhaeUtil.MyUtil import MyUtil
from tensorflow.keras.models import Sequential
import tensorflow.keras.utils as kerasUtil
from tensorflow.keras.preprocessing.text import Tokenizer
import datetime as dt

import numpy as np
import tensorflow as tf
import re
import matplotlib.pyplot as plt

seed = 0
#np.random.seed(seed)
#tf.random.set_seed(3)

m = Mecab()
util = MyUtil()


'''
데이터 전처리
'''
# 기사 수
documentCount = 5760
resultDocumentCount = 1440

# 기사 한 문장으로 합치기
contentsText = ''
# 기사 배열
articleMemory = []
resultArticleMemory = []

f = open("stopwordsKorean.txt", 'r', -1, "utf-8")
stopwoardList = f.read()
f.close()

# 기사 불러오기
for i in range(documentCount):
    if i < 1920:
        f = open("Finance/Finance%05d.txt" % i, 'r', -1, "utf-8")
    elif i < 3840:
        f = open("Social/Social%05d.txt" % (i-1920), 'r', -1, "utf-8")
    else:
        f = open("Science/Science%05d.txt" % (i-3840), 'r', -1, "utf-8")
    data = f.read()
    f.close()

    data = data[139:]

    data = re.sub("[-=+,#/\?:%$.@*\"※~&%!\\'|\(\)\[\]\<\>`\'\\\\n\\\\t{}◀▶▲☞“”ⓒ◇]", "", data)
    data = data[:-117]
    data = m.morphs(data)

    data = util.stopWord(data, stopwoardList)

    articleMemory.append(data)

# 정답데이터 기사 불러오기
for i in range(resultDocumentCount):
    if i < 480:
        f = open("Science/Science%05d.txt" % (i + 1920), 'r', -1, "utf-8")
    elif i < 960:
        f = open("Finance/Finance%05d.txt" % (i + 1920 - 480), 'r', -1, "utf-8")
    else:
        f = open("Social/Social%05d.txt" % (i + 1920 - 960), 'r', -1, "utf-8")
    data = f.read()
    f.close()

    data = data[139:]
    data = re.sub("[-=+,#/\?:%$.@*\"※~&%!\\'|\(\)\[\]\<\>`\'\\\\n\\\\t{}◀▶▲☞“”ⓒ◇]", "", data)
    data = data[:-117]
    data = m.morphs(data)
    data = util.stopWord(data, stopwoardList)
    articleMemory.append(data)

token = Tokenizer()
token.fit_on_texts(articleMemory)
word_size = len(token.word_index)+1
tempX = token.texts_to_sequences(articleMemory)
max_len = max(len(l) for l in tempX)
padded_x = sequence.pad_sequences(tempX, maxlen=2000)
x_train = padded_x[0:5760]
x_test = padded_x[5760:]

# 정답 데이터 작성
y_class = np.arange(documentCount)
for i in range(1920):
    y_class[i] = 0
for i in range(1920):
    y_class[i+1920] = 1
for i in range(1920):
    y_class[i+3840] = 2

y_test = np.arange(resultDocumentCount)
for i in range(480):
    y_test[i] = 2
for i in range(480):
    y_test[i+480] = 0
for i in range(480):
    y_test[i+960] = 1

y_train = y_class

x_train, y_train = util.randomBox(x_train, y_train, 2000)
#x_test, y_test = util.randomBox(x_test, y_test, 500)

y_train = kerasUtil.to_categorical(y_train)
y_test = kerasUtil.to_categorical(y_test)

x_test = np.array(x_test)



new_model = tf.keras.models.load_model('C://Users/och5351/Desktop/tensorflowSave/2020.03.11/my_model3.h5')
#new_model.summary()
#loss, acc = new_model.evaluate(x_test[0], y_test[0])
#print("Restored model, accuracy: {:5.2f}%".format(100*acc))
a = 930
predict = new_model.predict_classes(np.array([x_test[a]]))

for i in range(1):
    p = ''
    r = ''
    if predict[i] == 0:
        p = '경제'
    elif predict[i] == 1:
        p = '사회'
    else:
        p = '과학'
    if y_test[a].argmax() == 0:
        r = '경제'
    elif y_test[a].argmax() == 1:
        r = '사회'
    else:
        r = '과학'

    print(str(i+1)+"번 :"+"\n예측값 : {pred}".format(pred=p)+"\t\t 정답 : {res}".format(res=r))
