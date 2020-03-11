from eunjeon import Mecab
from chanhaeUtil.MyUtil import MyUtil
import re
import numpy as np
from numpy import array
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding
import matplotlib.pyplot as plt

tf.random.set_seed(1234)

m = Mecab()
util = MyUtil()
# 기사 수
documentCount = 900
resultDocumentCount = 300

# 정답 데이터 작성
y_class = np.arange(documentCount)
for i in range(300):
    y_class[i] = 0
for i in range(300):
    y_class[i+300] = 1
for i in range(300):
    y_class[i+600] = 2

y_test = np.arange(resultDocumentCount)
for i in range(100):
    y_test[i] = 0
for i in range(100):
    y_test[i+100] = 1
for i in range(100):
    y_test[i+200] = 2

y_class = tf.keras.utils.to_categorical(y_class)
y_test = tf.keras.utils.to_categorical(y_test)



# 기사 한 문장으로 합치기
contentsText = ''
# 기사 배열
articleMemory = []
resultArticleMemory = []



# 기사 불러오기
for i in range(documentCount):
    if i < 300:
        f = open("Finance/Finance%05d.txt" % i, 'r', -1, "utf-8")
    elif i < 600:
        f = open("Social/Social%05d.txt" % (i-300), 'r', -1, "utf-8")
    else:
        f = open("Science/Science%05d.txt" % (i-600), 'r', -1, "utf-8")
    data = f.read()
    f.close()

    data = data[139:]
    contentsText += data
    data = re.sub("[-=+,#/\?:%$.@*\"※~&%!\\'|\(\)\[\]\<\>`\'\\\\n\\\\t{}◀▶▲☞“”ⓒ◇]", "", data)
    data = m.morphs(data)
    articleMemory.append(data)

# 특수 문자 & 기호 제거
token = re.sub("[-=+,#/\?:%$.@*\"※~&%!\\'|\(\)\[\]\<\>`\'\\\\n\\\\t{}◀▶▲☞“”ⓒ◇]","",contentsText) # 특수 기호 제외
# 형태소 분리
temp = m.morphs(token)
# BagOfWords 적용
sequencesText = util.BagOfWords(temp)
# StopWord 적용
sequencesText = util.stopWord(sequencesText[0], sequencesText[1], deleteRate=0.02)
# TF-IDF 구하기
tfidf = util.tf_idf(articleMemory, documentCount, list(sequencesText[0].keys()))

articleMemory = []

# 기사 불러오기
for i in range(resultDocumentCount):
    if i < 100:
        f = open("Finance/Finance%05d.txt" % (i + 300), 'r', -1, "utf-8")
    elif i < 200:
        f = open("Social/Social%05d.txt" % (i + 300 - 100), 'r', -1, "utf-8")
    else:
        f = open("Science/Science%05d.txt" % (i + 300 - 200), 'r', -1, "utf-8")
    data = f.read()
    f.close()

    data = data[139:]

    contentsText += data
    data = re.sub("[-=+,#/\?:%$.@*\"※~&%!\\'|\(\)\[\]\<\>`\'\\\\n\\\\t{}◀▶▲☞“”ⓒ◇]", "", data)
    data = m.morphs(data)
    articleMemory.append(data)

# 특수 문자 & 기호 제거
token = re.sub("[-=+,#/\?:%$.@*\"※~&%!\\'|\(\)\[\]\<\>`\'\\\\n\\\\t{}◀▶▲☞“”ⓒ◇]","",contentsText) # 특수 기호 제외
# 형태소 분리
temp = m.morphs(token)
# BagOfWords 적용
sequencesText = util.BagOfWords(temp)
# StopWord 적용
sequencesText = util.stopWord(sequencesText[0], sequencesText[1], deleteRate=0.02)
# TF-IDF 구하기
resultTfidf = util.tf_idf(articleMemory, resultDocumentCount, list(sequencesText[0].keys()))

#util.gradient_descent(tfidf,y_class,201)

model = Sequential()
model.add(Dense(6, input_dim=tfidf.shape[1], activation='relu'))
model.add(Dense(3, input_dim=tfidf.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
resultTfidf = pad_sequences(resultTfidf, tfidf.shape[1])
hist = model.fit(tfidf, y_class, batch_size=1, epochs=200, shuffle=False, validation_data=(resultTfidf, y_test))
#word_size = len(sequencesText[0]) + 1


loss_and_metrics = model.evaluate(resultTfidf, y_test, batch_size=1)
print("="*20,"검증")
print(loss_and_metrics)

xhat = resultTfidf
yhat = model.predict(xhat)
print('## yhat ##')
print(yhat)



'''
classes = array([1,1,1,1,1,0,0,0,0,0])

# 정규화
token = re.sub("(\.)","",text)
token2 = util.wordNormalization(docs)

# 형태소 분리
token = m.morphs(token)
token2 = util.divideMorpheme(token2)

sequencsText = util.BagOfWords(token2)
word_size = len(sequencsText[0]) + 1

#max 찾아내기
maxV = len(sequencsText[0])
for count in sequencsText:
    if maxV < len(count):
        maxV = len(count)

padded_data = tf.keras.preprocessing.sequence.pad_sequences(sequencsText[2], maxV)

model = Sequential()
model.add(Embedding(word_size, 8, input_length=maxV))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_data, classes, epochs=20)

print("\n Accuracy: %.4f" % (model.evaluate(padded_data, classes)[1]))
'''