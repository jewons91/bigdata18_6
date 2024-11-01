import logging


# 사용자정의함수 7개
from stock_news_predictor_function import get_recent_opening_date
from stock_news_predictor_function import get_recent_newsdata_dataframe
from stock_news_predictor_function import get_recent_SOX_result
from stock_news_predictor_function import df_preprocessing
from stock_news_predictor_function import prediction_operating

from stock_news_predictor_function import stock_news_predictor
from stock_news_predictor_function import prediction_result_db_insert

# DB
from connection import conn

ticker = '005930.KS'
search_word = '삼성전자'
ticker_SOX = '^SOX'

try:
    # DB 연결
    conn.connect_to_database()
    logging.info("DB 연결 성공")
    
    # 메인함수 실행
    logging.info("--- 메인함수 시작 ---")
    print("--- 메인함수 시작 ---")
    GSTC_CODE, INVEST_CODE, average_probability_1, average_probability_0, execution_time = stock_news_predictor(ticker, search_word, ticker_SOX, conn)
    logging.info("--- 메인함수 종료 ---")
    print("--- 메인함수 종료 ---")
    
    
    # 결과값을 DB에 insert
    logging.info("--- INSERT함수 시작 ---")
    print("--- INSERT함수 시작 ---")
    prediction_result_db_insert(GSTC_CODE, INVEST_CODE, average_probability_1, average_probability_0, execution_time, conn)
    logging.info("--- INSERT함수 종료 ---")
    print("--- INSERT함수 종료 ---")
    
except Exception as e:
    logging.error(f"메인모듈 실행문에서 오류 발생: {e}")
finally:
    # DB 해제
    conn.close_database_connection()
    logging.info("DB 연결 종료")