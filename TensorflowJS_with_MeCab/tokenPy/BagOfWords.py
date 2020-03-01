from eunjeon import Mecab
import re

m = Mecab()

text = "정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다."
token = re.sub("(\.)","",text)
token = m.morphs(token)
print(token)