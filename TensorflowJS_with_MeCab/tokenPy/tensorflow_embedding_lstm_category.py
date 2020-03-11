from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Conv1D, MaxPooling1D
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
np.random.seed(seed)
tf.random.set_seed(3)

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
        f = open("Finance/Finance%05d.txt" % (i + 1920), 'r', -1, "utf-8")
    elif i < 960:
        f = open("Social/Social%05d.txt" % (i + 1920 - 480), 'r', -1, "utf-8")
    else:
        f = open("Science/Science%05d.txt" % (i + 1920 - 960), 'r', -1, "utf-8")
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
    y_test[i] = 0
for i in range(480):
    y_test[i+480] = 1
for i in range(480):
    y_test[i+960] = 2

y_train = y_class

#x_train, y_train = util.randomBox(x_train, y_train, 2000)
#x_test, y_test = util.randomBox(x_test, y_test, 500)

y_train = kerasUtil.to_categorical(y_train)
y_test = kerasUtil.to_categorical(y_test)

print(word_size)

'''
##전처리 끝


model = Sequential()
model.add(Embedding(word_size, 100, input_length=2000))
model.add(Dropout(0.5))
model.add(Conv1D(64,5,padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(55, activation='tanh'))
model.add(Dense(3, activation='softmax'))
tf.keras.losses.CategoricalCrossentropy()

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=50, epochs=5, shuffle=True,validation_data=(x_test, y_test))

print("\n Test Accuracy : %.4f" % (model.evaluate(x_test, y_test)[1]))

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch'
           "\n Test Accuracy : %.4f" % (model.evaluate(x_test, y_test)[1]))
plt.ylabel('loss')
plt.show()
h = str(dt.datetime.hour)
m = str(dt.datetime.minute)
model.save('C://Users/och5351/Desktop/tensorflowSave/2020.03.11/my_model3.h5')
'''