from datetime import datetime
from CrawlingCategory.naverFinanceCr import naverFinanceCrawling
from CrawlingCategory.naverScienceCr import naverScienceCrawling
from CrawlingCategory.naverSocialCr import naverSocialCrawling

if __name__ == '__main__':
    today = str(datetime.today().year) + "%02d" %datetime.today().month + "%02d" %datetime.today().day
    naverFinanceCrawling(today)
    naverScienceCrawling(today)
    naverSocialCrawling(today)