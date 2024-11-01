
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import sys
import urllib.request
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import logging
import configparser
import json

from connection import conn
from naverNewsCrawling.insert_krnews_naverapi import main


# 0.
    # congif_naverapi.ini 파일로부터 네이버API 연결에 필요한 두가지 값 리턴.
def load_naver_api_config():

    file_path=os.getenv('ini_file_path', '/root/airflow/dags/naverNewsCrawling/config_naverapi.ini')
    logging.info(f'### Current working directory: {os.getcwd()} ###')
    logging.info(f'### Config file path: {file_path} ###')

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            logging.info("File contents:")
            logging.info(f.read())
    except FileNotFoundError:
        logging.info(f"config file 찾을 수 없음. 파일경로: {file_path}")
        return {} 
    config = configparser.ConfigParser()
    config.read(file_path, encoding='utf-8')
    logging.info(f"config file에서 찾은 섹션: {config.sections()}")
    if 'NAVER_API' not in config.sections():
        logging.info("config file에서 NAVER_API 섹션을 찾을 수 없음.")
        return {}
    return {
        'client_id': config.get('NAVER_API', 'client_id'),
        'client_secret': config.get('NAVER_API', 'client_secret')
    }

# 함수 1.
    # DB에 있는 주식종목명들 가져와서 search_list 리스트 만들기.
def fetch_select_krstock_tb():
    search_list = []
    try:
        logging.info('### 1번 함수 fetch_select_krstock_tb 시작 ###')
        conn.global_cursor.execute("SELECT STC_NAME FROM TB_STOCKCLASSIFY")
        search_list = [row[0] for row in conn.global_cursor.fetchall()]
        logging.info('########## 데이터 확인 ##########')
        logging.info(search_list)
        logging.info('########## 데이터 확인 ##########')
    except conn.pymysql.Error as e:
        logging.error(f'DB로부터 검색종목 목록 가져오다가 오류 발생: {e}')
    return search_list

# 함수 2.
def fetch_krnews_naverapi(search_list):
    # 0.
    api_config = load_naver_api_config()
    client_id = api_config['client_id']
    client_secret = api_config['client_secret']
    
    display = 10
    
    data = []
    for search_word in search_list:
        ################
        df = main(client_id, client_secret, search_word, display)
        ################
        logging.info("########################### search df ###########################")
        logging.info(df)
        logging.info("########################### search df ###########################")
        if df is not None and not df.empty:
            data.append(df)
        else:
            logging.warning(f"이번 종목에서 DB 저장 대상인 뉴스가 없음. : {search_word}")
    
    if data:
        try:
            df = pd.concat(data, ignore_index=True)
            idx_json_data = df.to_json(orient='records', date_format='iso')
            return idx_json_data
        except ValueError as e:
            logging.error(f"Concatenation failed: {e}")
            return "Fail fetch_krnews_naverapi"
    else:
        logging.error("No valid dataframes to concatenate.")
        return "Fail fetch_krnews_naverapi"
    
