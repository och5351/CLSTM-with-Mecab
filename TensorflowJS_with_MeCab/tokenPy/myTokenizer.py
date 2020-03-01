#from eunjeon import Mecab
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf


docs = ['먼저 텍스트의 각 단어를 나누어 토큰화합니다.',
        '텍스트의 단어로 토큰화해야 딥러닝에서 인식됩니다.',
        '토큰화한 결과는 딥러닝에서 사용할 수 있습니다.']

m = Mecab()
print('[MeCab 형태소 분석기]')
print(m.morphs('제가이렇게띄어쓰기를전혀하지않고글을썼다고하더라도글을이해할수있습니다.'))
print('[Keras 문장 단어 분석기]')
print(text_to_word_sequence('제가이렇게띄어쓰기를전혀하지않고글을썼다고하더라도글을이해할수있습니다.'))
token = Tokenizer()
token.fit_on_texts('제가이렇게띄어쓰기를전혀하지않고글을썼다고하더라도글을이해할수있습니다.')
print("[몇개의 단어]")
print(token.word_counts) # 몇개의 단어가 나오는지
print("[몇개의 문장]")
print(token.document_count) # 몇개의 문장이 나오는지
print("[각 단어들이 몇 개의 문장에 나오는가]")
print(token.word_docs) # 각 단어들이 몇 개의 문장에 나오는가
print("[각 단어에 매겨진 인덱스 값]")
print(token.word_index) # 각 단어에 매겨진 인덱스 값

'''
print("[원 핫 인코딩]")
print(token.word_index) # 각 단어에 매겨진 인덱스 값
print(token.texts_to_sequences(docs))

one_hot_encoding_text = token.texts_to_sequences(docs)
count = len(one_hot_encoding_text)

#max 찾아내기
maxV = 0
for count in range(len(one_hot_encoding_text)):
    if count == 0:
        maxV = len(one_hot_encoding_text[count])
    else:
        if maxV < len(one_hot_encoding_text[count]):
            maxV = len(one_hot_encoding_text[count])


padded_data = tf.keras.preprocessing.sequence.pad_sequences(one_hot_encoding_text, maxV)
print(padded_data)

'''
'''
for rePrintCount in range(count):
    token = Tokenizer()
    token.fit_on_texts([docs[rePrintCount]])
    one_hot_encoding_text = token.texts_to_sequences([docs[rePrintCount]])

    x = tf.keras.utils.to_categorical(one_hot_encoding_text, num_classes=len(token.word_index) + 1)
    print("[{}번째 결과]".format(rePrintCount+1))
    print(x)
'''



'''
padded_data = tf.keras.preprocessing.sequence.pad_sequences(one_hot_encoding_text, 5)
print(padded_data)
'''
'''
for rePrintCount in range(len(one_hot_encoding_text)):
    x = to_categorical([one_hot_encoding_text[rePrintCount]], num_classes=len(one_hot_encoding_text[rePrintCount])+1)
    print("[{}번째 결과]".format(rePrintCount))
    print(x)
'''