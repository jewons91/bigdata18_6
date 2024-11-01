# 네이버페이증권 소맥 일별종가 스크래핑
    # 3번째 컬럼 '전일대비' 앞에 +, - 부호 붙일 수 있도록 코드 개선 희망
    # 밀 가격과 밀접한 글로벌기업, 국내식품기업 주가 변동과 어느정도 일치성 가지는지 확인 희망

import numpy as np
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
import folium
from bs4 import BeautifulSoup

from tqdm import tqdm
import json
import requests

date1 = []
closing1 = []
change1 = []
percent1 = []

driver = webdriver.Chrome()

page_num = '0'

for _ in tqdm(range(400)):
    # 셀레니움이 다음 페이지 url로 넘어가기
    page_num = str(int(page_num)+1)
    auto_url = f'https://finance.naver.com/marketindex/worldDailyQuote.naver?fdtc=2&marketindexCd=CMDT_W&page={page_num}'
    driver.get(auto_url)
    # html 따고 파싱하기.
    auto_html = driver.page_source
    auto_soup = BeautifulSoup(auto_html, 'html.parser')
    # 리스트에 담기
    for i in auto_soup.find_all(class_='date'):
        date1.append(i.text.strip())
    for i,j in enumerate(auto_soup.find_all(class_='num')):
        if i%3 == 0:
            closing1.append(j.text.strip())
        elif i%3 == 1:
            change1.append(j.text.strip())
        else:
            percent1.append(j.text.strip())
driver.close()

# 데이터프레임 생성하기
data = {
    '날짜':date1
    , '종가':closing1
    , '전일대비':change1
    , '등락율':percent1
}
df = pd.DataFrame(data)

# CSV파일로 저장하기
recent = df['날짜'].values[0]        # 제일 최근날짜
past = df['날짜'].values[-1]         # 제일 과거날짜
df.to_csv(
    f'wheat_price_scraping_{recent}_{past}.csv'
    , encoding='utf-8'
)
