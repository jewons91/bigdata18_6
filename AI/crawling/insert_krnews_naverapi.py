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
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

from konlpy.tag import Okt
import jpype
import jpype.imports
from jpype.types import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import logging
import configparser
import json

from connection import conn

# 변경1

def get_naver_news_urls(client_id, client_secret, search_word, display):
    encText = urllib.parse.quote(search_word)
    url = f"https://openapi.naver.com/v1/search/news?query={encText}&display={display}&start=1&sort=sim"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    
    if rescode == 200:
        response_body = response.read()
        response_data = json.loads(response_body.decode('utf-8-sig'))
        url_list = [item['link'] for item in response_data['items'] if 'https://n.news' in item['link']]
        logging.info(f"원본 JSON데이터: {response_body.decode('utf-8-sig')}")
        return url_list
    else:
        logging.info(f'Error Code: {str(rescode)}')
        return []

def scrape_news(url_list, GSTC_CODE, INVEST_CODE):
    logging.info("###################################### scrape_news(url_list, GSTC_CODE, INVEST_CODE) 시작")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)

    data = {
          'KRNEWS_CODE': []
        , 'GSTC_CODE': []
        , 'INVEST_CODE': []
        , 'KRNEWS_TITLE': []
        , 'KRNEWS_CONTENT': []
        , 'KRNEWS_DATE': []
        , 'KRNEWS_PRESS': []
        , 'KRNEWS_URL': [] 
    }
    logging.info("######################### url list #########################")
    logging.info(url_list)
    logging.info("######################### url list #########################")
    
    for url in tqdm(url_list):
        driver.get(url)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            data['KRNEWS_CODE'].append(datetime.now().strftime('%f')[:3])
        except AttributeError as e:
            data['KRNEWS_CODE'].append('N/A')
        
        data['GSTC_CODE'].append(GSTC_CODE)
        data['INVEST_CODE'].append(INVEST_CODE)
        
        try:
            data['KRNEWS_TITLE'].append(soup.find('h2', id='title_area').text)
        except AttributeError as e:
            logging.error(f'뉴스제목 AttributeError e: {e}')
            data['KRNEWS_TITLE'].append('N/A')
        
        try:
            article = soup.find('article')
            for img_desc in article.find_all('em', class_='img_desc'):
                img_desc.decompose()
            article_text = article.get_text()
            data['KRNEWS_CONTENT'].append(article_text.strip())
        except AttributeError as e:
            logging.error(f'뉴스본문 AttributeError e: {e}')
            data['KRNEWS_CONTENT'].append('N/A')
        
        try:
            data['KRNEWS_DATE'].append(soup.find('span', class_='media_end_head_info_datestamp_time').get('data-date-time'))
        except AttributeError as e:
            logging.error(f'뉴스날짜 AttributeError e: {e}')
            data['KRNEWS_DATE'].append('N/A')
        
        try:
            data['KRNEWS_PRESS'].append(soup.find('img', class_='media_end_head_top_logo_img').get('title'))
        except AttributeError as e:
            logging.error(f'언론사명 AttributeError e: {e}')
            data['KRNEWS_PRESS'].append('N/A')
        
        data['KRNEWS_URL'].append(url)
    logging.info('#########################################################################################################')
    logging.info('#########################################################################################################')
    logging.info(data)
    logging.info('#########################################################################################################')
    logging.info('#########################################################################################################')
    driver.quit()
    
    #print(len(url_list), len(data['KRNEWS_CODE']), len(data['KRNEWS_TITLE']), len(data['KRNEWS_CONTENT']), len(data['KRNEWS_DATE']), len(data['KRNEWS_PRESS']), len(data['KRNEWS_URL']))
    # 리스트 길이 같은지 확인
    if len(url_list) == len(data['KRNEWS_CODE']) == len(data['KRNEWS_TITLE']) == len(data['KRNEWS_CONTENT']) == len(data['KRNEWS_DATE']) == len(data['KRNEWS_PRESS']) == len(data['KRNEWS_URL']):
        logging.info('리스트의 길이가 일치합니다.')
    else:
        logging.info('리스트의 길이 중 일치하지 않는 것이 있습니다. 데이터 확인이 필요합니다.')
    
    
    df = pd.DataFrame(data)
    for i, row in df.iterrows():
        date_object = datetime.strptime(row['KRNEWS_DATE'], '%Y-%m-%d %H:%M:%S')
        date_string = date_object.strftime('%Y%m%d%H%M%S')
        df.at[i, 'KRNEWS_CODE'] = 'KRN' + date_string + str(row['KRNEWS_CODE'])
    logging.info("###################################### scrape_news(url_list, GSTC_CODE, INVEST_CODE) 종료")    
    return df

# 5.
def process_dataframe(df):
    rows_before = len(df)
    
    # Remove 'N/A' values('N/A'포함 행 제거)
    df = df[~df.isin(['N/A']).any(axis=1)]
    # Remove NaN values(NaN값 있는 행 제거)
    df = df.dropna()
    # Remove rows with NaN in NEWS_CONTENT(뉴스내용 컬럼이 NaN인 행 제거)
    df = df.dropna(subset=['KRNEWS_CONTENT'])
    
    # Remove duplicate titles(제목중복제거)
    df.drop_duplicates(subset='KRNEWS_TITLE', inplace=True)
    rows_after = len(df)
    logging.info(f'df_new에서 NaN값을 포함한 기사 및 제목중복 기사를 {rows_before-rows_after}개 삭제함.')
    
    # 오늘날짜, 어제날짜 아닌 데이터는 삭제.
    today = datetime.now().date()
    oneday_ago = today - timedelta(days=1)
    
    
    today_str = today.strftime('%Y-%m-%d')
    oneday_ago_str = oneday_ago.strftime('%Y-%m-%d')
    
    rows_before2 = len(df)
    
    df = df[df['KRNEWS_DATE'].str.startswith(today_str) | df['KRNEWS_DATE'].str.startswith(oneday_ago_str)]
    rows_after2 = len(df)
    logging.info(f'df_new에서 오늘, 어제보다 더 오래된 기사 {rows_before2-rows_after2}개를 삭제함.')
    
    
    return df


# 6.
def get_old_news_from_db(GSTC_CODE):
    today = datetime.now().date()
    oneday_ago = today - timedelta(days=1)
    
    try:
        oldnews_query = """
        SELECT * 
        FROM TB_KRNEWS a 
        WHERE a.GSTC_CODE = %s 
        AND (a.KRNEWS_DATE LIKE %s OR a.KRNEWS_DATE LIKE %s)
        """
        query_params = (GSTC_CODE, f'{today}%', f'{oneday_ago}%')
        conn.global_cursor.execute(oldnews_query, query_params)
        
        results = conn.global_cursor.fetchall()
        columns_names = [desc[0] for desc in conn.global_cursor.description]
        return pd.DataFrame(results, columns=columns_names)
    except conn.pymysql.Error as e:
        logging.info(f"데이터 불러오던 중 오류 발생: {e}")
        conn.rollback_changes()
        return pd.DataFrame()

def initialize_jvm():
    if not jpype.isJVMStarted():
        logging.info("JVM이 시작되지 않았으므로 시작합니다.")
        jvmpath = "/usr/local/java/jdk-17.0.8+7/lib/server/libjvm.so"
        konlpy_java_path = "/root/anaconda3/envs/ml-dev/lib/python3.10/site-packages/konlpy/java"
        classpath = f"{konlpy_java_path}/bin:{konlpy_java_path}/aho-corasick.jar:{konlpy_java_path}/jhannanum-0.8.4.jar:{konlpy_java_path}/komoran-3.0.jar:{konlpy_java_path}/kkma-2.0.jar:{konlpy_java_path}/open-korean-text-2.1.0.jar:{konlpy_java_path}/scala-library-2.12.3.jar:{konlpy_java_path}/shineware-common-1.0.jar:{konlpy_java_path}/shineware-ds-1.0.jar:{konlpy_java_path}/snakeyaml-1.12.jar:{konlpy_java_path}/twitter-text-1.14.7.jar"


        try:
            jpype.startJVM(jvmpath, f"-Djava.class.path={classpath}", convertStrings=True)
            logging.info("JVM이 성공적으로 시작되었습니다.")
        except Exception as e:
            logging.error(f"JVM 시작 중 오류 발생: {e}")
            return False
    return True


# 12.
def calculate_similarity(df):
    
    logging.info("################### df == df_oldnew ################")
    logging.info(df)
    logging.info("################### df == df_oldnew ################")
        
    if not initialize_jvm():
        return None

    # JClass로 kr.lucypark.okt.OktInterface 클래스를 직접 로드합니다.
    try:
        # OktInterface = jpype.JClass('kr.lucypark.okt.OktInterface')
        # okt = OktInterface()
        okt = Okt()
    except Exception as e:
        logging.error(f"JClass를 사용한 OktInterface 로드 중 오류 발생: {e}")
        return None
    
    def tokenize_korean(text):
        if isinstance(text, str):  # 입력이 문자열인지 확인
            try:
                # Okt의 morphs 함수 호출
                tokens = okt.morphs(text, stem=True)
                logging.info('########################## token #######################')
                logging.info(tokens)
                logging.info('########################## token #######################')
                return tokens
            except Exception as e:
                logging.error(f"Error in tokenize_korean function: {e}")
                return []
        else:
            logging.error("입력된 텍스트가 문자열이 아닙니다.")
            return []

    tfidf = TfidfVectorizer(tokenizer=tokenize_korean)
    logging.info("################### tfidf ################")
    logging.info(tfidf)
    logging.info("################### tfidf ################")   
    try:
        logging.info("################### df['KRNEWS_CONTENT'] ################")
        logging.info(df['KRNEWS_CONTENT'])
        logging.info("################### df['KRNEWS_CONTENT'] ################")          
        tfidf_matrix = tfidf.fit_transform(df['KRNEWS_CONTENT'])
    except ValueError as e:
        logging.error(f"df['KRNEWS_CONTENT']가 비었을 경우 ValueError 발생: {e}")
        return None

    return cosine_similarity(tfidf_matrix, tfidf_matrix)


# 13.
def mark_similar_news(df, cosine_sim, threshold=0.7):
    unique_indices = []
    removed_indices = []
    for idx in range(cosine_sim.shape[0]):
        if not any(cosine_sim[idx, unique_indices] > threshold):
            unique_indices.append(idx)
        else:
            removed_indices.append(idx)

    df['Similarity'] = np.nan
    df['Removed'] = 'x'
    for idx in removed_indices:
        similar_indices = [i for i in unique_indices if cosine_sim[idx, i] > threshold]
        if similar_indices:
            df.at[idx, 'Similarity'] = cosine_sim[idx, similar_indices[0]]
            df.at[idx, 'Removed'] = 'o'
    
    return df

# 14.
def save_dataframe_to_csv(df, filename):
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    logging.info(f'{filename}을 csv파일로 저장')




##########################
def main(client_id, client_secret, search_word, display):
    logging.info("########################### main(client_id, client_secret, search_word, display)  ###########################")
    logging.info(client_id)
    logging.info(client_secret)
    logging.info(search_word)
    logging.info(display)
    logging.info("########################### main(client_id, client_secret, search_word, display) ###########################")   
    
    # 2.
    try:
        conn.global_cursor.execute("SELECT a.GSTC_CODE FROM TB_STOCKCLASSIFY a WHERE a.STC_NAME = %s", (search_word,))
        result = conn.global_cursor.fetchone()
        if result:
            GSTC_CODE = result[0]
        else:
            logging.info(f"GSTC_CODE not found for {search_word}")
            return

        conn.global_cursor.execute("SELECT a.INVEST_CODE FROM TB_STOCKCLASSIFY a WHERE a.STC_NAME = %s", (search_word,))
        result = conn.global_cursor.fetchone()
        if result:
            INVEST_CODE = result[0]
        else:
            logging.info(f"INVEST_CODE not found for {search_word}")
            return

        logging.info(f"GSTC_CODE: {GSTC_CODE}")
        logging.info(f"INVEST_CODE: {INVEST_CODE}")
    except conn.pymysql.Error as e:
        logging.error(f"GSTC_CODE, INVEST_CODE를 SELECT하다가 오류 발생: {e}")
        conn.rollback_changes()
        return

    # 3.
    url_list = get_naver_news_urls(client_id, client_secret, search_word, display)
    if url_list is None or len(url_list) == 0:
        logging.warning('이번 종목에 대한 url_list가 비었음.')
        return None
    
    # 4.
    df_new = scrape_news(url_list, GSTC_CODE, INVEST_CODE)
    
    # 5.
    df_new = process_dataframe(df_new)
    
    # 6.
    df_old = get_old_news_from_db(GSTC_CODE)
    
    # 7.
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # save_dataframe_to_csv(df_old, f'df_old_{GSTC_CODE}_{timestamp}.csv')
    # save_dataframe_to_csv(df_new, f'df_new_{GSTC_CODE}_{timestamp}.csv')
    
    # 8.
    df_new = df_new[~df_new['KRNEWS_TITLE'].isin(df_old['KRNEWS_TITLE'])]
    
    # 9.
    df_old['OLD_NEW'] = 'old'
    df_new['OLD_NEW'] = 'new'
    
    # 10.
    df_oldnew = pd.concat([df_old, df_new], axis=0, ignore_index=True)
    
    # 11. 
    # save_dataframe_to_csv(df_oldnew, f'df_oldnew_{GSTC_CODE}_{timestamp}.csv')
    
    # 12.
    logging.info("################### df_oldnew ################")
    logging.info(df_oldnew)
    logging.info("################### df_oldnew ################")
    cosine_sim = calculate_similarity(df_oldnew)
    
    logging.info("################### cosine_sim ################")
    logging.info(cosine_sim)
    logging.info("################### cosine_sim ################")
    if cosine_sim is not None:
        
        # 13.
        df_oldnew = mark_similar_news(df_oldnew, cosine_sim)

        # 14.
        # save_dataframe_to_csv(df_oldnew, f'df_oldnew2_{GSTC_CODE}_{timestamp}.csv')
                 
        # remove_ = len(df_oldnew[(df_oldnew[df_oldnew['OLD_NEW'] == 'new']) & (df_oldnew['Removed'] == 'o')])
        # insert_ = len(df_old[(df_oldnew[df_oldnew['OLD_NEW'] == 'new']) & (df_oldnew['Removed'] == 'x')])
        # df_oldnew에서 'new'  'x' 달린 애들 개수 (DB 저장 대상)
    
        # 'new'이고 'Removed'가 'x'인 행들을 필터링하여 df_to_insert에 저장
        df_to_insert = df_oldnew[(df_oldnew['OLD_NEW'] == 'new') & (df_oldnew['Removed'] == 'x')]

        # 'new' 태그가 붙은 행의 총 개수
        total_ = len(df_oldnew[df_oldnew['OLD_NEW'] == 'new'])

        # 'new' 태그가 붙고 'Removed'가 'o'인 행의 개수 (탈락 대상)
        remove_ = len(df_oldnew[(df_oldnew['OLD_NEW'] == 'new') & (df_oldnew['Removed'] == 'o')])

        # 'new' 태그가 붙고 'Removed'가 'x'인 행의 개수 (DB 저장 대상)
        insert_ = len(df_oldnew[(df_oldnew['OLD_NEW'] == 'new') & (df_oldnew['Removed'] == 'x')])

        # 로그 메시지 출력
        logging.info(f'유사도 검사 결과, 총 {total_}개의 뉴스 중 {remove_}개를 삭제하고 {insert_}개를 DB 저장예정.')

        return df_to_insert
        
    
    '''
    # 16.
    similarity_df = pd.DataFrame(cosine_sim, 
                             index=df_oldnew['KRNEWS_TITLE'], 
                             columns=df_oldnew['KRNEWS_TITLE'])
    print(similarity_df)
    similarity_df.to_csv(f'similarity_matrix_{GSTC_CODE}_{timestamp}.csv', encoding='utf-8-sig')
    '''


# 15.
# 함수 3.
def insert_krnews_to_db(json_data):
    
    
    try:
        data = json.loads(json_data)
        
        insert_query = """
        INSERT INTO TB_KRNEWS (KRNEWS_CODE, GSTC_CODE, INVEST_CODE, KRNEWS_TITLE, KRNEWS_CONTENT, KRNEWS_DATE, KRNEWS_PRESS, KRNEWS_URL) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = [
            (
                item['KRNEWS_CODE'],
                item['GSTC_CODE'],
                item['INVEST_CODE'],
                item['KRNEWS_TITLE'],
                item['KRNEWS_CONTENT'],
                item['KRNEWS_DATE'],
                item['KRNEWS_PRESS'],
                item['KRNEWS_URL']
            )
            for item in data
        ]
        conn.global_cursor.executemany(insert_query, values)
        conn.commit_changes()
        logging.info(f"데이터 {len(values)}건이 성공적으로 MariaDB에 삽입되었습니다.")
    except json.JSONDecodeError as je:
        logging.error(f"Error decoding JSON: {je}")
    except Exception as e:
        logging.error(f"데이터 적재 중 오류 발생: {e}")
        conn.rollback_changes()
    finally:
        conn.close_database_connection()
        logging.error('프로그램 완전 종료!')
