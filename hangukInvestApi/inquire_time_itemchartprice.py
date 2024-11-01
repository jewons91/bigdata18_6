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

with open(file_path + '/api.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
APP_KEY = lines[0].strip()
APP_SECRET = lines[1].strip()

ACCESS_TOKEN = ''

try:
    with open(file_path + '/token.txt', 'r', encoding='utf-8') as f:
        tokenLines = f.readlines()
    ACCESS_TOKEN = tokenLines[0].strip()
except Exception as e:
    logging.error(f'{e}')
    
URL_BASE = "https://openapi.koreainvestment.com:9443" #실전투자
# URL_BASE = "https://openapi.koreainvestment.com:29443" #모의투자

# 주식 분봉 데이터
def get_time_price(stock_no, input_time=''):
    PATH = "uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
    URL = f"{URL_BASE}/{PATH}"
    
    global ACCESS_TOKEN

    # 헤더 설정
    headers = {"Content-Type":"application/json", 
            "authorization": f"Bearer {ACCESS_TOKEN}",
            "appKey":APP_KEY,
            "appSecret":APP_SECRET,
            "tr_id":"FHKST03010200",
            "custtype": 'P'}

    params = {
        "FID_ETC_CLS_CODE": "",
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_no,
        "FID_INPUT_HOUR_1": input_time,
        "FID_PW_DATA_INCU_YN": "N"
    }

    # 호출
    res = requests.get(URL, headers=headers, params=params)
    if res.status_code == 200 and res.json()["rt_cd"] == "0" :
        return(res.json())
    elif res.json()['rt_cd'] == '1':
        ACCESS_TOKEN = auth()
        logging.info('만료된 토큰이므로 토큰 재발급 후 재실행')
        time.sleep(1)
        return get_time_price(stock_no, input_time)
    else:
        logging.info("Error Code : " + str(res.status_code) + " | " + res.text)
        return None

# if __name__ == '__main__':
#     result = get_time_price("005930")
#     print(result)
#     # df = pd.DataFrame(result)
#     # print(df)
#     # with pd.ExcelWriter('./news_data.xlsx1') as w:
#     #     df.to_excel(w, sheet_name='TEST')