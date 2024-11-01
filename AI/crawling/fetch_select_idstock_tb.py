import conn
import logging

def fetch_stock_names(search_word):
    conn.connect_to_database()
    search_list = []
    if conn.global_cursor is None:
        logging.info("데이터베이스에 연결되지 않았습니다.")
        return search_list
    
    try:
        conn.global_cursor.execute("SELECT GINDEX_NAME FROM TB_INDEXCLASSIFY a")
        search_list = [row[0] for row in conn.global_cursor.fetchall()]  
        logging.info(f'DB로부터 가져온 검색종목 목록 {search_list}')
    except conn.pymysql.Error as e:
        logging.error(f'DB로부터 검색종목 목록 가져오다가 오류 발생: {e}')
    return search_list

