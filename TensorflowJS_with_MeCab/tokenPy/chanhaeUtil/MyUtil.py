
class MyUtil:

    def __init__(self):
        return print("MyUtil Call")


    def BagOfBow(self, token):
        word2index = {}
        bow = []
        for voca in token:
            if voca not in word2index.keys():
                word2index[voca] = len(word2index)
                # token을 읽으면서, word2index에 없는 (not in) 단어는 새로 추가하고, 이미 있는 단어는 넘깁니다.
                bow.insert(len(word2index) - 1, 1)
            # BoW 전체에 전부 기본값 1을 넣어줍니다. 단어의 개수는 최소 1개 이상이기 때문입니다.
            else:
                index = word2index.get(voca)
                # 재등장하는 단어의 인덱스를 받아옵니다.
                bow[index] = bow[index] + 1
                # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더해줍니다. (단어의 개수를 세는 것입니다.)
        print(word2index)
        print(bow)