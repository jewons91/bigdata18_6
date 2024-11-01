from connection import conn
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import bs4
from bs4 import BeautifulSoup
import requests
from datetime import datetime
from random import randint
import time as time_module
import logging
# 유사도 검사 함수
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from investingCrawling.fetch_select_pdstock_tb import fetch_pdstock_names
from investingCrawling.insert_pdnews_to_db import insert_pdnews_to_db

# 수정 20240911
def search_PDT_CODE(search_word, config_file_path):
    conn.connect_to_database(file_path=config_file_path) 
    try:
        conn.global_cursor.execute("SELECT PDT_CODE FROM TB_PRODUCTCLASSIFY WHERE PDT_NAME = %s", (search_word,))
        result = conn.global_cursor.fetchone()
        
        if result:
            PDT_CODE = result[0]
            logging.info(f'첫번째 SELECT문 결과: {PDT_CODE}')
            return PDT_CODE
        else:
            logging.info(f'{search_word}에 해당하는 PDT_CODE를 찾을 수 없습니다.')
            return None
        
        return PDT_CODE
    except conn.pymysql.Error as e:
        logging.error(f'DB로부터 PDT_CODE 가져오는 중 오류 발생: {e}')
        return None

def get_old_pdnews_from_db(PDT_CODE, config_file_path): 
    conn.connect_to_database(file_path=config_file_path) 
    today = datetime.now().date()
    oneday_ago = today - timedelta(days=1)
    twoday_ago = today -timedelta(days=2)
    
    try:
        if not PDT_CODE:
            logging.info("유효하지 않은 PDT_CODE입니다.")
            return pd.DataFrame()
        
        oldnews_query = """
        SELECT * 
        FROM TB_USPRODUCTNEWS 
        WHERE PDT_CODE = %s 
        AND (USNEWS_DATE LIKE %s OR USNEWS_DATE LIKE %s OR USNEWS_DATE LIKE %s)
        """
        query_params = (PDT_CODE, f'{today}%', f'{oneday_ago}%', f'{twoday_ago}%')
        
        conn.global_cursor.execute(oldnews_query, query_params)
        
        results = conn.global_cursor.fetchall()
        columns_names = [desc[0] for desc in conn.global_cursor.description]
        return pd.DataFrame(results, columns=columns_names)
    except conn.pymysql.Error as e:
        logging.error(f"데이터 불러오던 중 오류 발생: {e}")
        conn.rollback_changes()
        return pd.DataFrame()
    finally : conn.close_database_connection()

def investing_pd(url_start):
    request_headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36',
        'Referer':'https://www.google.com/',
        'Accept-Language':'en-US,en;q=0.9',
        'Connection':'keep-alive', 
    }
    url_base = 'https://investing.com'
    url_sub = url_start
    url = url_base + url_sub

    href = []
    url_adds = []
    Press_soup = []
    press_soup = []

    logging.info('URL, Press추출 시작')

    for page in range(1,2): # (1,2) => 2-1까지 불러온다
        url_ = f'{url}{page}'
        response = requests.get(url=url_, headers=request_headers).text
        time_module.sleep(randint(0, 3))  # 1초에서 5초 사이의 랜덤 지연
        soup = BeautifulSoup(response, 'html.parser')
        # 뉴스 URL을 가져오는 부분
        list_soup = soup.find_all('a', class_='block text-base font-bold leading-5 hover:underline sm:text-base sm:leading-6 md:text-lg md:leading-7')
        # 언론사를 가져오는 부분
        press_soup = soup.find_all('li', class_='overflow-hidden text-ellipsis')

    for i, item in enumerate(list_soup):
        # 각 URL을 가져온 후에 1초 대기
        url_adds.append(item['href'])
        time_module.sleep(randint(0,3))  # 1초 대기

        # press_soup에서 크롤링한 데이터가 있는지 확인하고, 없으면 None으로 처리
        if i < len(press_soup):
            Press_soup.append(press_soup[i].get_text())
        else:
            Press_soup.append(None)

    # DataFrame 생성
    data = pd.DataFrame({
        '주소': url_adds,
        '뉴스': Press_soup
    })
    Press_soup = [item.replace('By', '') for item in Press_soup]
    
    logging.info('###############################################################')
    logging.info(data)
    logging.info('###############################################################')
    
    title = []
    content = []
    time_t = []
    logging.info('추출 완료')
    logging.info('나머지 추출')

    for idx in tqdm(data.index):
        url_str = data['주소'][idx]
    
        response = requests.get(url=url_str, headers=request_headers).text
        soup_tmp = BeautifulSoup(response, 'html.parser')
        
        title_element = soup_tmp.find('h1', id='articleTitle')
        time_element = soup_tmp.find('div', class_='flex flex-col gap-2 text-warren-gray-700 md:flex-row md:items-center md:gap-0')
        content_element = soup_tmp.find('div', class_='article_WYSIWYG__O0uhw article_articlePage__UMz3q text-[18px] leading-8')
    
        Title_tmp = title_element.get_text() if title_element else 'None'
        Time_tmp = time_element.get_text() if time_element else 'None'
    
        if Time_tmp and len(Time_tmp.split()) >= 3:
            Time_tmp = Time_tmp.split()[1:4]
        else:
            Time_tmp = ['None', '00:00', 'AM']

        Content_tmp = content_element.get_text() if content_element else 'None'
    
        title.append(Title_tmp)
        content.append(Content_tmp)
        time_t.append(Time_tmp)

    for idx in time_t:
        if len(idx) >= 3:
            idx[2] = idx[2].replace('Updated', ' ')

    time_result = []
    for idx in time_t:
        date_ = idx[0].replace(',', '').strip()
        time_ = idx[1].strip()
        period_ = idx[2].strip() if len(idx) >= 3 else 'AM'
    
        if period_ == 'PM' and not time_.startswith('12'):
            hour, minute = time_.split(':')
            time_ = f"{int(hour) + 12}:{minute}"
        elif period_ == 'AM' and time_.startswith('12'):
            hour, minute = time_.split(':')
            time_ = f"00:{minute}"
    
        result = f'{date_} {time_}'
        time_result.append(result)

    chtime = []
    for idx in time_result:
        if idx.split()[0] != 'None':
            try:
                date_org = datetime.strptime(idx, '%m/%d/%Y %H:%M')
                date_org += timedelta(hours=13)
                chtime.append(date_org)
            except ValueError:
                chtime.append(None)
        else:
            chtime.append(None)

    data1 = pd.DataFrame({
        'USNEWS_TITLE': title,
        'USNEWS_CONTENT': content,
        'USNEWS_DATE': chtime,
        'USNEWS_PRESS': Press_soup,
        'USNEWS_URL': url_adds,
    })
    
    data1 = data1[data1['USNEWS_CONTENT'] != 'None']
    df = data1
    df['New_Index'] = range(len(df))
    # 오늘 날짜 필터링
    today = datetime.today().date()
    today_df = df[df['USNEWS_DATE'].apply(lambda x: x.date() == today if x else False)]
    return today_df

def insert_pd_db(search_word, url_start, config_file_path):
    today = datetime.today().strftime('%Y-%m-%d')
    PDT_CODE = search_PDT_CODE(search_word, config_file_path)
    if PDT_CODE:
        db_df = get_old_pdnews_from_db(PDT_CODE, config_file_path) 
    print('#######################################')
    print(db_df)
    print('#######################################')
    news_df = investing_pd(url_start)
    conn.connect_to_database(file_path=config_file_path) 
    try:
        conn.global_cursor.execute("SELECT PDT_CODE FROM TB_PRODUCTCLASSIFY WHERE PDT_NAME = %s", (search_word,))
        PDT_CODE = conn.global_cursor.fetchone()[0]
        logging.info(f'첫번째 SELECT문 {PDT_CODE}')
        conn.global_cursor.execute("SELECT INVEST_CODE FROM TB_PRODUCTCLASSIFY WHERE PDT_NAME = %s", (search_word,))
        INVEST_CODE = conn.global_cursor.fetchone()[0]
        logging.info(f'두번째 SELECT문 {INVEST_CODE}')
    except conn.pymysql.Error as e:
        logging.error(f'PDT_CODE, INVEST_CODE를 SELECT하다가 오류 발생: {e}')
        conn.rollback_changes()
    finally : conn.close_database_connection()
    
    # data 변수를 미리 빈 DataFrame으로 초기화
    data = pd.DataFrame()

    for idx, row in news_df.iterrows():
        time_str = row['USNEWS_DATE'].strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(row['USNEWS_DATE']) else None
        new_index_str = str(row['New_Index']) # int -> str
        row_data = pd.DataFrame({ 
            'USPDTN_CODE' : [PDT_CODE + time_str.replace('-', '').replace(':', '').replace(' ', '') + new_index_str if time_str else 'USN00000000000000'], 
            'PDT_CODE' : [PDT_CODE], 
            'INVEST_CODE' : [INVEST_CODE], 
            'USNEWS_TITLE' : [row['USNEWS_TITLE']], 
            'USNEWS_CONTENT' : [row['USNEWS_CONTENT']],
            'USNEWS_DATE' : [time_str],
            'USNEWS_PRESS' : [row['USNEWS_PRESS']], 
            'USNEWS_URL' : [row['USNEWS_URL']]
        })
        # 루프에서 각 row_data를 누적하여 data에 추가
        data = pd.concat([data, row_data], ignore_index=True)
    return data, db_df

def consim_pd(data, db_df, config_file_path): 
    combined_df = pd.concat([data, db_df], axis=0, ignore_index=True)
    # 'USNEWS_CONTENT'와 'USNEWS_TITLE'이 존재하는지 확인
    if 'USNEWS_CONTENT' not in combined_df.columns or 'USNEWS_TITLE' not in combined_df.columns:
        print("경고: 'USNEWS_CONTENT' 또는 'USNEWS_TITLE' 열이 데이터프레임에 없습니다.")
        return pd.DataFrame()
    combined_df['USNEWS_DATE'] = pd.to_datetime(combined_df['USNEWS_DATE'], errors='coerce')
    # NaN 값, 빈 row 제거
    combined_df = combined_df.dropna(subset=['USNEWS_CONTENT', 'USNEWS_TITLE'])
    combined_df = combined_df[(combined_df['USNEWS_CONTENT'].str.strip() != '') & (combined_df['USNEWS_TITLE'].str.strip() != '')]
    # 기사가 없으면 종료
    if combined_df.empty:
        logging.info("'USNEWS_CONTENT' 또는 'USNEWS_TITLE' 열이 비어있거나 유효한 데이터가 없습니다.")
        pass
    else:
        # 제목과 내용을 결합하여 유사도 계산
        combined_df['combined_text'] = combined_df['USNEWS_TITLE'] + " " + combined_df['USNEWS_CONTENT']

        # TF-IDF 행렬 
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(combined_df['combined_text'])
        # tfidf_matrix = vectorizer.fit_transform(combined_df['USNEWS_CONTENT'])

        # 코사인 유사도 계산
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # 유사도 결과를 DataFrame으로 저장
        similarity_df = pd.DataFrame(cosine_sim, index=combined_df.index, columns=combined_df.index)

        # 유사한 기사 인덱스 저장 리스트
        rows_to_drop = []

        # 유사도 검사: 유사도가 0.9 이상이거나 1.0인 경우 제거 대상에 추가
        for i in range(similarity_df.shape[0]):
            for j in range(i + 1, similarity_df.shape[1]):
                if cosine_sim[i, j] >= 0.9:
                    logging.info(f"유사한 기사 발견: {i}와 {j} (유사도: {cosine_sim[i, j]})")
                    rows_to_drop.append(j) # append
                    rows_to_drop.append(i)
        # 유사한 기사 삭제
        if rows_to_drop:
            logging.info(f"유사한 기사 있음. 해당 기사는 삭제됩니다. 삭제 인덱스: {rows_to_drop}")
            combined_df = combined_df.drop(rows_to_drop, axis=0)
        else:
            logging.info("유사한 기사 없음.")
        # 날짜 변환해서 오늘 기사만 남기고 모두 삭제
        today = datetime.today().date()
        news_json_df = combined_df[combined_df['USNEWS_DATE'].apply(lambda x: x.date() == today if pd.notnull(x) else False)]

        # 최종 결과를 JSON으로 변환하여 데이터베이스에 삽입
        if not news_json_df.empty:
            news_json = news_json_df.to_json(orient='records', date_format='iso')
            logging.info('#####################################################')
            logging.info('#####################################################')
            logging.info(news_json)
            logging.info('#####################################################')
            logging.info('#####################################################')
            insert_pdnews_to_db(news_json, config_file_path) 
        else:
            logging.info('기사 없어서 json변환 건너뜀')



def process_and_compare_pdnews(config_file_path):
    source = [
    # ('비트코인/달러', '/crypto/bitcoin/news/'), 이상함
    ('브렌트유', '/commodities/brent-oil-news/'), 
    ('WTI원유', '/commodities/crude-oil-news/'), 
    # ('이더리움/달러', '/crypto/ethereum/news/'),이상함
    ('COMEX_금', '/commodities/gold-news/'),
    ('COMEX_구리', '/commodities/copper-news/'),
    ('CBOT_밀', '/commodities/us-wheat-news/'),
    ('원/달러', '/currencies/usd-krw-news/')
]
    insert = pd.DataFrame(source, columns=['search_word', 'url_start'])
    for idx, row in insert.iterrows():
        data, db_df = insert_pd_db(row['search_word'], row['url_start'], config_file_path) 
        combined_df = consim_pd(data, db_df, config_file_path) 


# process_and_compare_pdnews()  # insert_db 호출