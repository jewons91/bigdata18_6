import requests
import json
import numpy as np
import pandas as pd
from hangukInvestAPI.access_token import auth
import os
import time
import logging

file_path=os.getenv('ini_file_path', '/root/airflow/dags/hangukInvestAPI')
logging.info(f'### Current working directory: {os.getcwd()} ###')
logging.info(f'### Config file path: {file_path} ###')

with open(file_path+'/api.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
APP_KEY = lines[0].strip()
APP_SECRET = lines[1].strip()

ACCESS_TOKEN = ''

try:
    with open(file_path+'/token.txt', 'r', encoding='utf-8') as f:
        tokenLines = f.readlines()
    ACCESS_TOKEN = tokenLines[0].strip()
except Exception as e:
    logging.error(f'{e}')

URL_BASE = "https://openapi.koreainvestment.com:9443" #실전투자
# URL_BASE = "https://openapi.koreainvestment.com:29443" #모의투자

# 주식 일별 데이터
def get_daily_price(stock_no, start_date, end_date):
    PATH = "uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    URL = f"{URL_BASE}/{PATH}"
    
    global ACCESS_TOKEN
    
    # 헤더 설정
    headers = {"Content-Type":"application/json", 
            "authorization": f"Bearer {ACCESS_TOKEN}",
            "appKey":APP_KEY,
            "appSecret":APP_SECRET,
            "tr_id":"FHKST03010100",
            "custtype": 'P'}

    params = {
        "FID_COND_MRKT_DIV_CODE": "J"
        , "FID_INPUT_ISCD": stock_no
        , "FID_INPUT_DATE_1": start_date
        , "FID_INPUT_DATE_2": end_date
        , "FID_PERIOD_DIV_CODE": "D"
        , "FID_ORG_ADJ_PRC": "0"
    }

    # 호출
    res = requests.get(URL, headers=headers, params=params)
    # print(ACCESS_TOKEN)
    if res.status_code == 200 and res.json()["rt_cd"] == "0" :
        return(res.json())
    elif res.json()['rt_cd'] == '1':
        ACCESS_TOKEN = auth()
        logging.info('만료된 토큰 또는 토큰이 없으므로 토큰 발급 후 실행')
        time.sleep(1)
        return get_daily_price(stock_no, start_date, end_date)
    else:
        logging.info("Error Code : " + str(res.status_code) + " | " + res.text)
        return None

# if __name__ == '__main__':
#     result = get_daily_price("005930", '20240501', '20240730')
#     print(type(result))
#     df = pd.DataFrame(result['output2'])

#     df.to_csv('stock_daily.csv', encoding='utf-8')
