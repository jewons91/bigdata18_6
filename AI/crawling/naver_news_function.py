# 240903 10시

import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt
import conn
# 크롬드라이버 headless로 실행
from selenium.webdriver.chrome.options import Options

def naver_news_scraping(search_word, start_date, end_date):
    # 날짜 포맷 설정 및 날짜 리스트 생성
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')

    date_list = [(start_date + timedelta(days=x)).strftime('%Y%m%d') for x in range(0, (end_date - start_date).days + 1)]

    # 크롬드라이버 headless 옵션
    chrome_options = Options()
    # 브라우저를 GUI 없이 백그라운드에서 실행.
#    chrome_options.add_argument("--headless")
    # 크롬 샌드박스 보안기능을 비활성화.
        # 리눅스 root 사용자로 크롬 실행할 때 필요.
        # 보안상 사용을 피하는게 좋음.
#    chrome_options.add_argument("--no-sandbox")
    # /dev/shm : 리눅스 시스템에서 사용되는 공유메모리파일시스템.
        # /dev/shm 비활성화하고 /tmp 디렉토리 사용하도록 함.
        # 도커 컨테이너 등에서 발생할 수 있는 메모리부족문제 해결.
#    chrome_options.add_argument("--disable-dev-shm-usage")
#    driver = webdriver.Chrome(options=chrome_options)  # WebDriver 초기화
    # headless 안 쓸거면 driver = webdriver.Chrome() 1줄만.
    driver = webdriver.Chrome()
    scrolls = 1  # 스크롤다운 횟수
    def scroll_down(driver, scrolls):
        for _ in range(scrolls):
            driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
            time.sleep(1)

    # 총 몇 건의 뉴스 저장 중인지.
    count = 0
    
    # DB 연결
    conn.connect_to_database()

    # GSTC_CODE = 'KR7005930003'
    # INVEST_CODE = '01'

    try:
        conn.global_cursor.execute("SELECT a.GSTC_CODE FROM TB_STOCKCLASSIFY a WHERE a.STC_NAME = %s", (search_word,))
        GSTC_CODE = conn.global_cursor.fetchone()[0]
        print("첫번째 SELECT문", GSTC_CODE)
        conn.global_cursor.execute("SELECT a.INVEST_CODE FROM TB_STOCKCLASSIFY a WHERE a.STC_NAME = %s", (search_word,))
        INVEST_CODE = conn.global_cursor.fetchone()[0]
        print("두번째 SELECT문", INVEST_CODE)
    except conn.pymysql.Error as e:
        print(f"GSTC_CODE, INVEST_CODE를 SELECT하다가 오류 발생: {e}")
        conn.rollback_changes()

    # 날짜 범위에 대해 크롤링 및 유사도 분석 수행
    for i in range(0, len(date_list), 5):
        current_dates = date_list[i:i+5]

        # 스크래핑해서 담을 리스트들. => 데이터프레임, DB의 컬럼명.
        KRNEWS_TITLE = []    # 제목
        KRNEWS_CONTENT = []  # 본문
        KRNEWS_DATE = []     # 작성일
        KRNEWS_PRESS = []    # 언론사명
        KRNEWS_URL = []      # URL
        ms = []

        for search_date in tqdm(current_dates):
            formatted_date_dot = datetime.strptime(search_date, '%Y%m%d').strftime('%Y.%m.%d')

            url = f'https://search.naver.com/search.naver?where=news&query={search_word}&sort=0&photo=0&field=0&pd=3&ds={formatted_date_dot}&de={formatted_date_dot}&docid=&related=0&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so%3Ar%2Cp%3Afrom{search_date}to{search_date}&is_sug_officeid=0&office_category=0&service_area=0'
            driver.get(url)

            # 스크롤다운 사용자정의함수 실행
            scroll_down(driver, scrolls)

            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            naver_news_links = soup.find_all('a', string='네이버뉴스', class_='info')

            for link in naver_news_links:
                # 스포츠뉴스 제외
                if not 'm.sports' in link['href']:
                    driver.get(link['href'])
                    html = driver.page_source
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # 기사 따는 순간의 밀리초 ms에 넣기
                        # 기사의 밀리초가 아니라 이 순간의 밀리초
                    try:
                        ms.append(datetime.now().strftime('%f')[:3])
                    except AttributeError as e:
                        ms.append('N/A')
                    
                    # 각 항목 리스트에 데이터 추가, 에러 발생 시 'N/A' 추가
                    try:
                        KRNEWS_TITLE.append(soup.find('h2', id='title_area').text)
                    except AttributeError as e:
                        print('뉴스제목 AttributeError e: ', e)
                        KRNEWS_TITLE.append('N/A')
                    try:
                        article = soup.find('article')
                        for img_desc in article.find_all('em', class_='img_desc'):
                            img_desc.decompose()
                        article_text = article.get_text()
                        KRNEWS_CONTENT.append(article_text.strip())
                    except AttributeError as e:
                        print('뉴스본문 AttributeError e: ', e)
                        KRNEWS_CONTENT.append('N/A')
                    try:
                        KRNEWS_DATE.append(soup.find('span', class_='media_end_head_info_datestamp_time').get('data-date-time'))
                    except AttributeError as e:
                        print('뉴스날짜 AttributeError e: ', e)
                        KRNEWS_DATE.append('N/A')
                    try:
                        KRNEWS_PRESS.append(soup.find('img', class_='media_end_head_top_logo_img').get('title'))
                    except AttributeError as e:
                        print('언론사명 AttributeError e: ', e)
                        KRNEWS_PRESS.append('N/A')
                    try:
                        KRNEWS_URL.append(link['href'])
                    except AttributeError as e:
                        print('뉴스URL AttributeError e: ', e)
                        KRNEWS_URL.append('N/A')

        print(f'{current_dates} 완료')
        if KRNEWS_TITLE==[]:
            print('5일치 검색결과 KRNEWS_TITLE가 비었으므로 다음 단계로 continue함.')
            continue
        
        # '컬럼명':데이터 들어있는 리스트
        data = {
            'KRNEWS_TITLE': KRNEWS_TITLE,
            'KRNEWS_CONTENT': KRNEWS_CONTENT,
            'KRNEWS_DATE': KRNEWS_DATE,
            'KRNEWS_PRESS': KRNEWS_PRESS,
            'KRNEWS_URL': KRNEWS_URL,
            'ms' : ms
        }
        
        # 리스트 길이 같은지 확인
        if len(KRNEWS_TITLE) == len(KRNEWS_CONTENT) == len(KRNEWS_DATE) == len(KRNEWS_PRESS) == len(KRNEWS_URL) == len(ms):
            print('리스트의 길이가 일치합니다.')
        else:
            print('리스트의 길이 중 일치하지 않는 것이 있습니다. 데이터 확인이 필요합니다.')

        # 데이터프레임 생성
        df = pd.DataFrame(data)

        ### 3가지 방법으로 비어있는 뉴스기사 제거
        # 1. 'N/A'가 포함된 행 출력하고 제거
        na_rows = df[df.isin(['N/A']).any(axis=1)]
        # print('"N/A" 값을 가진 행 출력 후 제거:')
        # print(na_rows)
        df = df[~df.isin(['N/A']).any(axis=1)]

        # 2. NaN 값이 있는 행 출력하고 제거
        nan_rows = df[df.isna().any(axis=1)]
        # print("NaN 값을 가진 행 출력 후 제거:")
        # print(nan_rows)
        df = df.dropna()

        # 3. NEWS_CONTENT 컬럼 값이 NaN인 행을 찾아서 삭제
        nan_content = df[df['KRNEWS_CONTENT'].isna()]
        # print("NEWS_CONTENT 컬럼의 값이 NaN인 행 출력 후 제거:")
        # print(nan_content)
        df = df.dropna(subset=['KRNEWS_CONTENT'])

        # 제목 중복 제거
        rows_before = len(df)
        df.drop_duplicates(subset='KRNEWS_TITLE', inplace=True)
        rows_after = len(df)
        print(f'제목 중복 제거한 행의 개수 : {rows_before-rows_after}개')
        
        # 한국어 형태소 분석기 Okt를 이용하여 한국어 불용어 처리
        okt = Okt()

        def tokenize_korean(text):
            tokens = okt.morphs(text, stem=True)
            return tokens

        # TF-IDF 벡터화 (불용어 사용하지 않음)
        tfidf = TfidfVectorizer(tokenizer=tokenize_korean)
        try:
            tfidf_matrix = tfidf.fit_transform(df['KRNEWS_CONTENT'])
        # 위에서 if문으로 5일치 KRNEWS_TITLE가 비었으면 continue하니까 이거 필요없을듯.
        except ValueError as e:
            print("df['KRNEWS_CONTENT']가 비어있어서 tfidf.fit_transform()에서 ValueError 발생 e: ", e)
        
        # 코사인 유사도 계산
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # 유사도 기반으로 제거 여부 표기
        threshold = 0.65  # 유사도 임계값 설정
        unique_indices = []
        removed_indices = []
        for idx in range(cosine_sim.shape[0]):
            if not any(cosine_sim[idx, unique_indices] > threshold):
                unique_indices.append(idx)
            else:
                removed_indices.append(idx)

        # 원본 데이터프레임에 유사도 컬럼과 제거 여부 컬럼 추가
        df['Similarity'] = np.nan
        df['Removed'] = 'x'
        for idx in removed_indices:
            similar_indices = [i for i in unique_indices if cosine_sim[idx, i] > threshold]
            if similar_indices:
                df.at[idx, 'Similarity'] = cosine_sim[idx, similar_indices[0]]
                df.at[idx, 'Removed'] = 'o'

        # 유사도와 'o' 표시된 항목이 제거되지 않은 전체 파일 저장
        df_with_similarity = df.copy()
        execute_date = datetime.now().strftime("%Y%m%d")
        ###

        # 'o'로 표시된 행 제거
        df = df[df['Removed'] != 'o']

        # 데이터프레임 출력 확인
        print('유사도 기반 중복 제거 여부 표기 후 데이터프레임:')
        print(df.head())
        
        # print('1번 with similarity csv파일 저장 완료')
        # df_with_similarity.to_csv(f'naverSearchNews_{search_word}_{current_dates[0]}_{current_dates[-1]}_with_similarity.csv', encoding='utf-8-sig', index=False)
        # print('2번 removed csv파일 저장 완료')
        # df.to_csv(f'naverSearchNews_{search_word}_{current_dates[0]}_{current_dates[-1]}_removed.csv', encoding='utf-8-sig', index=False)
        

        ### DB에 넣기
        
        # DB에 넣을 데이터프레임
        df = df.drop(['Similarity', 'Removed'], axis=1)
        
        # max_number_query = f"""
        # SELECT MAX(CONVERT(SUBSTRING(KRNEWS_CODE, 14, 12), UNSIGNED)) 
        # FROM TB_KRNEWS 
        # WHERE KRNEWS_CODE LIKE '{GSTC_CODE}_%'
        # """
        
        try:
            # conn.global_cursor.execute(max_number_query)
            # max_code = conn.global_cursor.fetchone()[0]
            
            # start_code = '000000000001' if max_code is None else str(int(max_code) + 1).zfill(12)
            # print(f"쿼리를 통해 받아온 start_code : {start_code}")
            
            # 데이터 삽입 쿼리
            insert_query = """
            INSERT INTO TB_KRNEWS (KRNEWS_CODE, GSTC_CODE, INVEST_CODE, KRNEWS_TITLE, KRNEWS_CONTENT, KRNEWS_DATE, KRNEWS_PRESS, KRNEWS_URL)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """

            # 데이터프레임의 각 행을 순회하며 데이터 삽입
            for index, (row_index, row) in enumerate(df.iterrows()):
                
                date_object = datetime.strptime(row['KRNEWS_DATE'], '%Y-%m-%d %H:%M:%S')
                date_string = date_object.strftime('%Y%m%d%H%M%S')
                # print('date_string : ', date_string)
                # print('밀리초 : ', str(row['ms']))
                data = (
                    'KRN' + date_string + str(row['ms'])
                    , GSTC_CODE
                    , INVEST_CODE
                    , row['KRNEWS_TITLE']
                    , row['KRNEWS_CONTENT']
                    , row['KRNEWS_DATE']
                    , row['KRNEWS_PRESS']
                    , row['KRNEWS_URL']
                )
                conn.global_cursor.execute(insert_query, data)
                # print(f'이번에 start_code에 붙인 index : {index}')

            # 변경사항 커밋
            conn.commit_changes()
            print(f"데이터 {df.shape[0]}건이 성공적으로 MariaDB에 삽입되었습니다.")

        except conn.pymysql.Error as e:
            print(f"데이터 적재 중 오류 발생: {e}")
            conn.rollback_changes()
        
        # 총 몇 건의 뉴스 저장 중인지.
        count += df.shape[0]
    ### for문 end ###

    print(f'총 {count}건의 뉴스 데이터 스크래핑을 완료하였습니다.')
    
    # 드라이버 연결 종료    
    driver.quit
    # 데이터베이스 연결 종료
    conn.close_database_connection()

    print("프로그램 완전 종료")
    ### fucntion end ###


if __name__ == "__main__":
    search_word = '농심'
    start_date = '20240801'
    end_date = '20240801'
    naver_news_scraping(search_word, start_date, end_date)

