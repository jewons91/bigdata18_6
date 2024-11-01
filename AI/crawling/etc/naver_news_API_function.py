import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import sys
import urllib.request
import json
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import conn

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
        print("원본 JSON데이터:", response_body.decode('utf-8-sig'))
        return url_list
    else:
        print("Error Code:" + str(rescode))
        return []

def scrape_news(url_list, GSTC_CODE, INVEST_CODE):
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
            print('뉴스제목 AttributeError e: ', e)
            data['KRNEWS_TITLE'].append('N/A')
        
        try:
            article = soup.find('article')
            for img_desc in article.find_all('em', class_='img_desc'):
                img_desc.decompose()
            article_text = article.get_text()
            data['KRNEWS_CONTENT'].append(article_text.strip())
        except AttributeError as e:
            print('뉴스본문 AttributeError e: ', e)
            data['KRNEWS_CONTENT'].append('N/A')
        
        try:
            data['KRNEWS_DATE'].append(soup.find('span', class_='media_end_head_info_datestamp_time').get('data-date-time'))
        except AttributeError as e:
            print('뉴스날짜 AttributeError e: ', e)
            data['KRNEWS_DATE'].append('N/A')
        
        try:
            data['KRNEWS_PRESS'].append(soup.find('img', class_='media_end_head_top_logo_img').get('title'))
        except AttributeError as e:
            print('언론사명 AttributeError e: ', e)
            data['KRNEWS_PRESS'].append('N/A')
        
        data['KRNEWS_URL'].append(url)

    driver.quit()
    
    print(len(url_list), len(data['KRNEWS_CODE']), len(data['KRNEWS_TITLE']), len(data['KRNEWS_CONTENT']), len(data['KRNEWS_DATE']), len(data['KRNEWS_PRESS']), len(data['KRNEWS_URL']))
    # 리스트 길이 같은지 확인
    if len(url_list) == len(data['KRNEWS_CODE']) == len(data['KRNEWS_TITLE']) == len(data['KRNEWS_CONTENT']) == len(data['KRNEWS_DATE']) == len(data['KRNEWS_PRESS']) == len(data['KRNEWS_URL']):
        print('리스트의 길이가 일치합니다.')
    else:
        print('리스트의 길이 중 일치하지 않는 것이 있습니다. 데이터 확인이 필요합니다.')
    
    
    df = pd.DataFrame(data)
    for i, row in df.iterrows():
        date_object = datetime.strptime(row['KRNEWS_DATE'], '%Y-%m-%d %H:%M:%S')
        date_string = date_object.strftime('%Y%m%d%H%M%S')
        df.at[i, 'KRNEWS_CODE'] = 'KRN' + date_string + str(row['KRNEWS_CODE'])
    return df

# 5.
def process_dataframe(df):
    # Remove 'N/A' values('N/A'포함 행 제거)
    df = df[~df.isin(['N/A']).any(axis=1)]
    # Remove NaN values(NaN값 있는 행 제거)
    df = df.dropna()
    # Remove rows with NaN in NEWS_CONTENT(뉴스내용 컬럼이 NaN인 행 제거)
    df = df.dropna(subset=['KRNEWS_CONTENT'])
    
    # Remove duplicate titles(제목중복제거)
    rows_before = len(df)
    df.drop_duplicates(subset='KRNEWS_TITLE', inplace=True)
    rows_after = len(df)
    print(f'df_new 내에서 제목 중복 제거한 행의 개수 : {rows_before-rows_after}개')
    
    # 오늘날짜, 어제날짜 아닌 데이터는 삭제.
    today = datetime.now().date()
    oneday_ago = today - timedelta(days=1)
    
    today_str = today.strftime('%Y-%m-%d')
    oneday_ago_str = oneday_ago.strftime('%Y-%m-%d')
    
    df = df[df['KRNEWS_DATE'].str.startswith(today_str) | df['KRNEWS_DATE'].str.startswith(oneday_ago_str)]
    
    return df


# 6.
def get_old_news_from_db(GSTC_CODE):
    today = datetime.now().date()
    oneday_ago = today - timedelta(days=1)
    
    try:
        oldnews_query = """
        SELECT * 
        FROM TB_KRNEWS_TEST a 
        WHERE a.GSTC_CODE = %s 
        AND (a.KRNEWS_DATE LIKE %s OR a.KRNEWS_DATE LIKE %s)
        """
        query_params = (GSTC_CODE, f'{today}%', f'{oneday_ago}%')
        conn.global_cursor.execute(oldnews_query, query_params)
        
        results = conn.global_cursor.fetchall()
        columns_names = [desc[0] for desc in conn.global_cursor.description]
        return pd.DataFrame(results, columns=columns_names)
    except conn.pymysql.Error as e:
        print(f"데이터 불러오던 중 오류 발생: {e}")
        conn.rollback_changes()
        return pd.DataFrame()

def calculate_similarity(df):
    okt = Okt()
    def tokenize_korean(text):
        return okt.morphs(text, stem=True)

    tfidf = TfidfVectorizer(tokenizer=tokenize_korean)
    try:
        tfidf_matrix = tfidf.fit_transform(df['KRNEWS_CONTENT'])
    except ValueError as e:
        print("df['KRNEWS_CONTENT']가 비었을 경우 ValueError 발생 e : ", e)
        return None
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

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

def save_dataframe_to_csv(df, filename):
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f'{filename}을 csv파일로 저장')

# 추가
    # df을 받아서 JSON 문자열로 바꿔서 반환하는 함수.
def df_to_json(df):
    json_data = df.to_json(orient='records', date_format='iso')
    return json_data

# 15.
def insert_news_to_db(json_data):
    
    data = json.loads(json_data)
    
    try:
        insert_query = """
        INSERT INTO TB_KRNEWS_TEST (KRNEWS_CODE, GSTC_CODE, INVEST_CODE, KRNEWS_TITLE, KRNEWS_CONTENT, KRNEWS_DATE, KRNEWS_PRESS, KRNEWS_URL) 
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
        print("데이터가 성공적으로 MariaDB에 삽입되었습니다.")
    except json.JSONDecodeError as je:
        print(f"Error decoding JSON: {je}")
    except Exception as e:
        print(f"데이터 적재 중 오류 발생: {e}")
        conn.rollback_changes()


##########################
def main(client_id, client_secret, search_word, display):
    
    
    # 2.
    try:
        conn.global_cursor.execute("SELECT a.GSTC_CODE FROM TB_STOCKCLASSIFY a WHERE a.STC_NAME = %s", (search_word,))
        result = conn.global_cursor.fetchone()
        if result:
            GSTC_CODE = result[0]
        else:
            print(f"GSTC_CODE not found for {search_word}")
            return

        conn.global_cursor.execute("SELECT a.INVEST_CODE FROM TB_STOCKCLASSIFY a WHERE a.STC_NAME = %s", (search_word,))
        result = conn.global_cursor.fetchone()
        if result:
            INVEST_CODE = result[0]
        else:
            print(f"INVEST_CODE not found for {search_word}")
            return

        print("GSTC_CODE:", GSTC_CODE)
        print("INVEST_CODE:", INVEST_CODE)
    except conn.pymysql.Error as e:
        print(f"GSTC_CODE, INVEST_CODE를 SELECT하다가 오류 발생: {e}")
        conn.rollback_changes()
        return

    # 3.
    url_list = get_naver_news_urls(client_id, client_secret, search_word, display)
    
    # 4.
    df_new = scrape_news(url_list, GSTC_CODE, INVEST_CODE)
    
    # 5.
    df_new = process_dataframe(df_new)
    
    # 6.
    df_old = get_old_news_from_db(GSTC_CODE)
    
    # 7.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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
    cosine_sim = calculate_similarity(df_oldnew)
    if cosine_sim is not None:
    
    # 13.
        df_oldnew = mark_similar_news(df_oldnew, cosine_sim)
    
    # 14.
        # save_dataframe_to_csv(df_oldnew, f'df_oldnew2_{GSTC_CODE}_{timestamp}.csv')
        

        
        df_to_insert = df_oldnew[(df_oldnew['OLD_NEW'] == 'new') & (df_oldnew['Removed'] == 'x')]
        
        # 추가 1.
        json_to_insert = df_to_json(df_to_insert)
        
        # 15.
        insert_news_to_db(json_to_insert)
    
    '''
    # 16.
    similarity_df = pd.DataFrame(cosine_sim, 
                             index=df_oldnew['KRNEWS_TITLE'], 
                             columns=df_oldnew['KRNEWS_TITLE'])
    print(similarity_df)
    similarity_df.to_csv(f'similarity_matrix_{GSTC_CODE}_{timestamp}.csv', encoding='utf-8-sig')
    '''
    
    
