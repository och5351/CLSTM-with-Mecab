from eunjeon import Mecab
from chanhaeUtil.MyUtil import MyUtil
import re

m = Mecab()
util = MyUtil()

text = "정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다."
# 정규화
token = re.sub("(\.)","",text)
# 형태소 분리
token = m.morphs(token)
print(token)

util.BagOfBow(token)
