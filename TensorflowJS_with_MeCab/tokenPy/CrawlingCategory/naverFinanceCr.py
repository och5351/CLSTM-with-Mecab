from CrawlingCategory.crawlingUtil import CrawUtil
import os

class naverFinanceCrawling:

    crawlingUtil = CrawUtil()

    def __init__(self, today):
        URL = 'https://news.naver.com/main/ranking/popularDay.nhn?rankingType=popular_day&sectionId=102&date=' + today
        links = self.crawlingUtil.get_link(URL)
        fileNum = len(os.walk(
            'C://Users/och5351/Desktop/github_och/Using-RNN-with-Tensorflow.js/TensorflowJS_with_MeCab/\
            tokenPy/finance').__next__()[2])

        p = 0

        for count in range(len(links)):

            result_text = self.crawlingUtil.get_text('https://news.naver.com/' + links[count])
            if result_text == '':
                p += 1
            else:
                OUTPUT_FILE_NAME = 'finance/finance%05d.txt' % (count + fileNum - p)
                print(OUTPUT_FILE_NAME)
                open_output_file = open(OUTPUT_FILE_NAME, 'w', -1, "utf-8")
                open_output_file.write(result_text)
                open_output_file.close()


