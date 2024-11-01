from connection import conn
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import bs4
from bs4 import BeautifulSoup
import requests
from datetime import datetime
import logging
# 유사도 검사 함수
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from investingCrawling.fetch_select_idstock_tb import fetch_stock_names
from investingCrawling.insert_idnews_to_db import insert_usnews_to_db

def search_GINDEX_CODE(search_word, config_file_path):
    conn.connect_to_database(file_path=config_file_path)
    try:
        conn.global_cursor.execute("SELECT GINDEX_CODE FROM TB_INDEXCLASSIFY WHERE GINDEX_NAME = %s", (search_word,))
        result = conn.global_cursor.fetchone()

        if result:
            GINDEX_CODE = result[0]
            logging.info(f'첫번째 SELECT문 결과: {GINDEX_CODE}')
            return GINDEX_CODE
        else:
            logging.info(f'{search_word}에 해당하는 GINDEX_CODE를 찾을 수 없습니다.')
            return None

        return GINDEX_CODE
    except conn.pymysql.Error as e:
        logging.error(f"DB로부터 GINDEX_CODE 가져오는 중 오류 발생: {e}")
        return None
    finally : conn.close_database_connection()

def get_old_news_from_db(GINDEX_CODE, config_file_path):
    conn.connect_to_database(file_path=config_file_path)
    today = datetime.now().date()
    oneday_ago = today - timedelta(days=1)
    twoday_ago = today -timedelta(days=2)

    try:
        if not GINDEX_CODE:
            logging.info('유효하지 않은 GINDEX_CODE입니다.')
            return pd.DataFrame()

        oldnews_query = """
        SELECT * 
        FROM TB_USINDEXNEWS 
        WHERE GINDEX_CODE = %s 
        AND (USNEWS_DATE LIKE %s OR USNEWS_DATE LIKE %s OR USNEWS_DATE LIKE %s)
        """
        query_params = (GINDEX_CODE, f'{today}%', f'{oneday_ago}%', f'{twoday_ago}%')

        conn.global_cursor.execute(oldnews_query, query_params)

        results = conn.global_cursor.fetchall()
        columns_names = [desc[0] for desc in conn.global_cursor.description]
        return pd.DataFrame(results, columns=columns_names)
    except conn.pymysql.Error as e:
        logging.error(f'데이터 불러오던 중 오류 발생: {e}')
        conn.rollback_changes()
        return pd.DataFrame()
    finally : conn.close_database_connection()

def investing(url_start):
    request_headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    url_base = 'https://investing.com'
    url_sub = url_start
    url = url_base + url_sub

    href = []
    url_adds = []
    Press_soup = []
    press_soup = []

    logging.info('URL, Press추출 시작')

    for page in range(1, 2):
        url_ = f'{url}{page}'
        response = requests.get(url=url_, headers=request_headers).text
        soup = BeautifulSoup(response, 'html.parser')
        list_soup = soup.find_all('a', class_='block text-base font-bold leading-5 hover:underline sm:text-base sm:leading-6 md:text-lg md:leading-7')
        press_soup = soup.find_all('li', class_='overflow-hidden text-ellipsis')

        for i, item in enumerate(list_soup):
            href.append(item['href'])
            url_adds.append(item['href']) 

            if i < len(press_soup):
                Press_soup.append(press_soup[i].get_text())
            else:
                Press_soup.append(None)

    data = pd.DataFrame({
        '주소': url_adds,
        '뉴스': Press_soup
    })
    Press_soup = [item.replace('By', '') for item in Press_soup]

    title = []
    content = []
    time = []
    ms = []
    logging.info('추출 완료')
    logging.info('나머지 추출')

    for idx in tqdm(data.index):
        url_str = data['주소'][idx]

        response = requests.get(url=url_str, headers=request_headers).text
        soup_tmp = BeautifulSoup(response, 'html.parser')
        # PK를 위한 밀리초
        try:
            ms.append(datetime.now().strftime('%f')[:3])
        except AttributeError as e:
            ms.append('N/A')

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
        time.append(Time_tmp)

    for idx in time:
        if len(idx) >= 3:
            idx[2] = idx[2].replace('Updated', ' ')

    time_result = []
    for idx in time:
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
        'Ms_': ms
    })

    data1 = data1[data1['USNEWS_CONTENT'] != 'None']

    # 오늘 날짜 필터링
    today = datetime.today().date()
    today_df = data1[data1['USNEWS_DATE'].apply(lambda x: x.date() == today if x else False)]
    return today_df

def insert_db(search_word, url_start, config_file_path):
    today = datetime.today().strftime('%Y-%m-%d')
    GINDEX_CODE = search_GINDEX_CODE(search_word, config_file_path)
    if GINDEX_CODE:
        db_df = get_old_news_from_db(GINDEX_CODE, config_file_path)
    news_df = investing(url_start)
    conn.connect_to_database(file_path=config_file_path)
    try:
        conn.global_cursor.execute("SELECT a.GINDEX_CODE FROM TB_INDEXCLASSIFY a WHERE a.GINDEX_NAME = %s", (search_word,))
        GINDEX_CODE = conn.global_cursor.fetchone()[0]
        logging.info(f'첫번째 SELECT문 {GINDEX_CODE}')
        conn.global_cursor.execute("SELECT a.INVEST_CODE FROM TB_INDEXCLASSIFY a WHERE a.GINDEX_NAME = %s", (search_word,))
        INVEST_CODE = conn.global_cursor.fetchone()[0]
        logging.info(f'두번째 SELECT문 {INVEST_CODE}')
    except conn.pymysql.Error as e:
        logging.error(f'GINDEX_CODE, INVEST_CODE를 SELECT하다가 오류 발생: {e}')
        conn.rollback_changes()
    finally : conn.close_database_connection()

    # data 변수를 미리 빈 DataFrame으로 초기화
    data = pd.DataFrame()

    for idx, row in news_df.iterrows():
        time_str = row['USNEWS_DATE'].strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(row['USNEWS_DATE']) else None
        row_data = pd.DataFrame({ 
            'USIDXN_CODE' : ['USN'+ time_str.replace('-', '').replace(':', '').replace(' ', '') + row['Ms_'] if time_str else 'USN00000000000000'], 
            'GINDEX_CODE' : [GINDEX_CODE], 
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

def consim(data, db_df, config_file_path):
    combined_df = pd.concat([data, db_df], axis=0, ignore_index=True)
    # 'USNEWS_CONTENT'와 'USNEWS_TITLE'이 존재하는지 확인
    if 'USNEWS_CONTENT' not in combined_df.columns or 'USNEWS_TITLE' not in combined_df.columns:
        logging.info("'USNEWS_CONTENT' 또는 'USNEWS_TITLE' 열이 데이터프레임에 없습니다.")
        return pd.DataFrame()
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
                    print(f"유사한 기사 발견: {i}와 {j} (유사도: {cosine_sim[i, j]})")
                    rows_to_drop.append(j) # append
                    rows_to_drop.append(i)
        # 유사한 기사 삭제
        if rows_to_drop:
            logging.info(f'유사한 기사 있음. 해당 기사는 삭제됩니다. 삭제 인덱스: {rows_to_drop}')
            combined_df = combined_df.drop(rows_to_drop, axis=0)
        else:
            logging.info('유사한 기사 없음.')
        # 날짜 변환해서 오늘 기사만 남기고 모두 삭제
        today = datetime.today().date()
        news_json_df = combined_df[combined_df['USNEWS_DATE'].apply(lambda x: x.date() == today if pd.notnull(x) else False)]
        # 최종 결과를 JSON으로 변환하여 데이터베이스에 삽입
        if not combined_df.empty:
            news_json = news_json_df.to_json(orient='records', date_format='iso')
            logging.info('#####################################################')
            logging.info('#####################################################')
            logging.info(news_json)
            logging.info('#####################################################')
            logging.info('#####################################################')
            insert_usnews_to_db(news_json, config_file_path)
        else:
            logging.info('기사 없어서 json변환 건너뜀')



def process_and_compare_news(config_file_path):
    source = [
        ('SNP500', '/indices/us-spx-500-news/'), 
        ('코스피200', '/indices/kospi-news/'), 
        ('나스닥100', '/indices/nq-100-news/'), 
        ('미국_국채_10년물', '/rates-bonds/u.s.-10-year-bond-yield-news/')
    ]
    insert = pd.DataFrame(source, columns=['search_word', 'url_start'])
    for idx, row in insert.iterrows():
        data, db_df = insert_db(row['search_word'], row['url_start'], config_file_path)
        combined_df = consim(data, db_df, config_file_path) 

# process_and_compare_news()  # insert_db 호출