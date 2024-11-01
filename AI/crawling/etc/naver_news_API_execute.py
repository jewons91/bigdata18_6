import configparser
from naver_news_API_function import main
import os
import json
import conn

# congif_naverapi.ini 파일로부터 네이버API 연결에 필요한 두가지 값 리턴.
def load_naver_api_config(file_path='./config_naverapi.ini', encoding='utf-8'):
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    print(f"config 파일 읽기 시도 중. 파일경로: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            print("File contents:")
            print(f.read())
    except FileNotFoundError:
        print(f"config file 찾을 수 없음. 파일경로: {file_path}")
        return {} 
    config = configparser.ConfigParser()
    config.read(file_path, encoding='utf-8')
    print("config file에서 찾은 섹션:")
    print(config.sections())
    if 'NAVER_API' not in config.sections():
        print("config file에서 NAVER_API 섹션을 찾을 수 없음.")
        return {}
    return {
        'client_id': config.get('NAVER_API', 'client_id'),
        'client_secret': config.get('NAVER_API', 'client_secret')
    }

# DB에 있는 주식종목명들 가져와서 search_list 리스트 만들기.
def fetch_stock_names():
    search_list = []
    try:
        conn.global_cursor.execute("SELECT STC_NAME FROM TB_STOCKCLASSIFY")
        search_list = [row[0] for row in conn.global_cursor.fetchall()]  
        print('DB로부터 가져온 검색종목 목록 : search_list')
        print(search_list)
    except conn.pymysql.Error as e:
        print(f'DB로부터 검색종목 목록 가져오다가 오류 발생: {e}')
    return search_list

def naver_news_api_kafka():
    # 1.
    conn.connect_to_database()

    api_config = load_naver_api_config()
    client_id = api_config['client_id']
    client_secret = api_config['client_secret']

    search_list = fetch_stock_names()

    display = 10

    for search_word in search_list:
        main(client_id, client_secret, search_word, display)

    # 17. 
    conn.close_database_connection()
    print("프로그램 완전 종료!")



# 1.
conn.connect_to_database()

api_config = load_naver_api_config()
client_id = api_config['client_id']
client_secret = api_config['client_secret']

search_list = fetch_stock_names()

display = 10

for search_word in search_list:
    main(client_id, client_secret, search_word, display)

# 17. 
conn.close_database_connection()
print("프로그램 완전 종료!")
