from bs4 import BeautifulSoup
import urllib.request

class CrawUtil:

    def get_link(self, URL):
        source_code_from_URL = urllib.request.urlopen(URL)
        soup = BeautifulSoup(source_code_from_URL, 'html.parser', from_encoding='utf-8')
        link = []
        for list in soup.find("ol", class_="ranking_list").find_all("li"):
            div = list.find("div", class_="ranking_headline")
            link.append(div.find("a")["href"])

        return link

    def get_text(self, URL):
        source_code_from_URL = urllib.request.urlopen(URL)
        soup = BeautifulSoup(source_code_from_URL, 'html.parser', from_encoding='utf-8')
        text = ''

        for item in soup.find_all('div', id='articleBodyContents'):
            text = text + str(item.find_all(text=True))

        return text