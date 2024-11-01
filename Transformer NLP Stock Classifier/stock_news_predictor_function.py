import numpy as np
import pandas as pd

import yfinance as yf

from datetime import datetime, timedelta

from tensorflow.keras.models import load_model
from transformers import AutoTokenizer
from tensorflow.keras.regularizers import l2

from stock_news_predictor_custom_layer import PositionalEncoding, SelfAttention  # 사용자 정의 레이어 파일 임포트

import logging
import sys

# 삼성전자
ticker = '005930.KS'
search_word = '삼성전자'

# 필라델피아 반도체 지수
ticker_SOX = '^SOX'

# 함수 1.
    # 파라미터 : 주식종목 ticker
    # 리턴 : 제일 최근 폐장 시간을 문자열로
def get_recent_opening_date(ticker):
    # 현재 날짜와 시간 확인
    now = datetime.now()
    print(f"함수 1. 과정: 현재 일시 - {now}")
    
    try:
        # 삼성전자 제일 최근 주식 개장일 정보 가져오기
        stock_data = yf.download(ticker, period='5d', interval='1d')
        
        # 다운로드에 실패한 경우, 빈 데이터프레임을 처리하는 예외처리 추가
        if stock_data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")

        # 오늘이 평일이고 주식시장이 아직 열려있는 경우, 전날 폐장 시간(15:30)을 반환
            # now.weekday()가 현재 요일을 숫자로 반환.
            # 0~4가 월~금이므로 평일인지 확인하는 것.
        if now.weekday() < 5 and now.time() < datetime.strptime("15:30:00", "%H:%M:%S").time():
            recent_date = stock_data.index[-2]  # 전날 데이터 가져오기
        else:
            recent_date = stock_data.index[-1]  # 가장 최근 날짜 가져오기
        print('3')
        print(recent_date)
        # 날짜를 문자열로 변환하고 15:30 시간 붙이기
        recent_date_str = recent_date.strftime('%Y-%m-%d') + ' 15:30:00'

        # 결과 확인
        logging.info(f"함수 1. 결과: {recent_date_str}")
        print(f"함수 1. 결과: {recent_date_str}")

        return recent_date_str
        # ex) '2024-09-20 15:30:00'
    
    except Exception as e:
        print(f"함수 1 - Error in get_recent_opening_date: {e}")
        logging.error(f"함수 1 - Error in get_recent_opening_date: {e}")
        raise e




# 함수 2.
    # 파라미터 : search_word, 함수 1의 리턴값
    # 리턴 : 원하는 뉴스 데이터가 담긴 데이터프레임
def get_recent_newsdata_dataframe(search_word, recent_datetime_str, conn):
    try:
        # DB에서 GSTC_CODE 가져오기
        conn.global_cursor.execute("SELECT a.GSTC_CODE FROM TB_STOCKCLASSIFY a WHERE a.STC_NAME = %s", (search_word,))
        GSTC_CODE = conn.global_cursor.fetchone()[0]
        print(f"DB로부터 가져온 GSTC_CODE: {GSTC_CODE}")
        conn.global_cursor.execute("SELECT a.INVEST_CODE FROM TB_STOCKCLASSIFY a WHERE a.STC_NAME = %s", (search_word,))
        INVEST_CODE = conn.global_cursor.fetchone()[0]
        print(f"DB로부터 가져온 INVEST_CODE: {INVEST_CODE}")

        # 실행할 query문 설정
        query = "SELECT KRNEWS_DATE, KRNEWS_TITLE, KRNEWS_CONTENT FROM TB_KRNEWS a WHERE a.GSTC_CODE = %s AND a.KRNEWS_DATE >= %s"
        
        # query 실행
        conn.global_cursor.execute(query, (GSTC_CODE, recent_datetime_str))
        
        # DB에서 가져온 값을 데이터프레임으로
        df = pd.read_sql(query, conn.global_conn, params=(GSTC_CODE, recent_datetime_str))

        # 결과 확인
        logging.info(f"함수 2. 결과: {df}")
        print(f"함수 2. 결과: {df}")
        
        # 결과가 비었을 경우 프로그램 종료
        if df.empty:
            print("주식시장 종료 후 게재된 뉴스가 없음.")
            logging.error("주식시장 종료 후 게재된 뉴스가 없음.")
            sys.exit(1)  # 비정상 종료 상태 코드 1 사용
        
        return df, GSTC_CODE, INVEST_CODE
        # KRNEWS_DATE, KRNEWS_TITLE, KRNEWS_CONTENT 3개 컬럼만 가져온 데이터프레임
    
    except Exception as e:
        print(f"함수 2 - Error in get_recent_newsdata_dataframe: {e}")
        logging.error(f"함수 2 - Error in get_recent_newsdata_dataframe: {e}")
        raise e


# 함수 3.
    # 파라미터 : ticker_SOX
    # 리턴 : 문자열 '필상' 또는 '필하'
def get_recent_SOX_result(ticker_SOX):
    try:
        # 야후파이낸스에서 제일 최근 SOX 결과값 가져오기
        stock_data_SOX = yf.download(ticker_SOX, period='1d', interval='1d')
        
        if stock_data_SOX.empty:
            raise ValueError("No data found for the Philadelphia Semiconductor Index (SOX).")
        
        if stock_data_SOX['Close'].values - stock_data_SOX['Open'].values >= 1:
            return '필상'
        else:
            return '필하'
    
    except Exception as e:
        print(f"함수 3 - Error in get_recent_SOX_result: {e}")
        logging.error(f"함수 3 - Error in get_recent_SOX_result: {e}")
        raise e



# 함수 4.
    # 뉴스 데이터프레임을 모델에 집어넣을 수 있게 데이터전처리
def df_preprocessing(df, SOX_result):
    # SOX 결과를 TITLE 앞에 붙이기
    df['KRNEWS_TITLE'] = SOX_result + ' ' + df['KRNEWS_TITLE']
    # TITLE과 CONTENT를 합친 새로운 컬럼 만들기
    df['KRNEWS'] = df['KRNEWS_TITLE'] + ' ' + df['KRNEWS_CONTENT']
    
    df_krnews_series = df['KRNEWS']
    
    return df_krnews_series




# 함수 5.
def prediction_operating(df_krnews_series):
    try:
        # 저장해둔 모델 로딩
        best_model = load_model(r'C:\big18_yoon\07_DeepLearning_dl-dev\best_model_6-1_.h5', custom_objects={
            'PositionalEncoding': PositionalEncoding,
            'SelfAttention': SelfAttention,
            'l2': l2
        })

        # 저장해둔 토크나이저
        tokenizer = AutoTokenizer.from_pretrained(r'C:\big18_yoon\07_DeepLearning_dl-dev\tokenizer_directory_6-1_')
        tokenizer.pad_token = tokenizer.eos_token

        # 시리즈를 넘파이배열로
        predict_texts = df_krnews_series.to_numpy()
        
        # 넘파이배열 값들을 리스트에 담기
        inputs_predict_texts = []
        for i in predict_texts:
            inputs_predict_texts.append(i)

        
        # 텍스트 토큰화, 인덱싱
        encodings = tokenizer(inputs_predict_texts, padding='max_length', truncation=True, max_length=1024, return_tensors='tf')
        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask
        logging.info(f"Input IDs: {input_ids}")
        logging.info(f"Attention Mask: {attention_mask}")
        print(f"Input IDs: {input_ids}")
        print(f"Attention Mask: {attention_mask}")
        # 예측 실시
        predictions = best_model.predict({'input_1': input_ids, 'attention_mask': attention_mask})
        logging.info(f"예측결과: {predictions}")
        print(f"예측결과: {predictions}")
        
        # 결과 출력
        prediction_1_values = []
        prediction_0_values = []

        for i, prediction in enumerate(predictions):
            probability_1 = prediction[0]
            probability_0 = 1 - prediction[0]
            prediction_1_values.append(probability_1)
            prediction_0_values.append(probability_0)
            print(f"Sample {i+1}: Probability of being 1: {prediction[0]:.4f}, Probability of being 0: {1 - prediction[0]:.4f}")

        average_probability_1 = sum(prediction_1_values) / len(prediction_1_values)
        average_probability_0 = sum(prediction_0_values) / len(prediction_0_values)

        print(f"Average Probability of being 1: {average_probability_1:.4f}")
        print(f"Average Probability of being 0: {average_probability_0:.4f}")
        
        return average_probability_1, average_probability_0
    
    except Exception as e:
        print(f"함수 5 - Error in prediction_operating: {e}")
        logging.error(f"함수 5 - Error in prediction_operating: {e}")
        raise e

# 메인함수
def stock_news_predictor(ticker, search_word, ticker_SOX, conn):
    try:
        # 함수 1.
        recent_opening_date = get_recent_opening_date(ticker)
        logging.info('함수 1 리턴값 - 문자열')
        logging.info(f"한국주식 최근 개장일: {recent_opening_date}")
        
        # 함수 2.
        df, GSTC_CODE, INVEST_CODE = get_recent_newsdata_dataframe(search_word, recent_opening_date, conn)
        logging.info('함수 2 리턴값 - 데이터프레임')
        logging.info(f"한국주식 마지막 폐장 이후의 뉴스: {df}")

        # 함수 3.
        SOX_result = get_recent_SOX_result(ticker_SOX)
        logging.info('함수 3 리턴값 - 문자열')
        logging.info(f"필라델피아반도체지수 마지막 결과: {SOX_result}(함수 3 리턴값 - 문자열)")

        # 함수 4.
        df_krnews_series = df_preprocessing(df, SOX_result)
        logging.info('함수 4 리턴값 - 시리즈')
        logging.info(f'뉴스 데이터프레임 전처리 결과: {df_krnews_series}')

        # 함수 5.
        average_probability_1, average_probability_0 = prediction_operating(df_krnews_series)
        
        # 실행시간 기록
        execution_time = datetime.now()
        logging.info(f'메인함수가 실행된 일시: {execution_time}')
        
        print(GSTC_CODE, type(GSTC_CODE))
        print(INVEST_CODE, type(INVEST_CODE))
        print(average_probability_1, type(average_probability_1))
        print(average_probability_0, type(average_probability_0))
        print(execution_time, type(execution_time))
        
        return GSTC_CODE, INVEST_CODE, average_probability_1, average_probability_0, execution_time
        # DB 연결해서 최종값을 insert 하도록 하면 됨.
        # DB 테이블 구조 확인
    
    except Exception as e:
        print(f"메인함수 - Error in stock_news_predictor: {e}")
        logging.error(f"메인함수 - Error in stock_news_predictor: {e}")
        raise e


# DB삽입함수
    # 메인함수의 결과를 DB에 삽입
def prediction_result_db_insert(GSTC_CODE, INVEST_CODE, average_probability_1, average_probability_0, execution_time, conn):
    insert_query = """
    INSERT INTO TB_STOCK_KRNEWS_PREDICT
    VALUES (%s, %s, %s, %s, %s);
    """
    try:
        # numpy.float64 타입을 str로 변환
        predict_rise_rate = str(average_probability_1)
        predict_fall_rate = str(average_probability_0)
        
        # 쿼리 실행
        conn.global_cursor.execute(insert_query, (GSTC_CODE, INVEST_CODE, predict_rise_rate, predict_fall_rate, execution_time))
        logging.info("insert 쿼리가 성공적으로 실행되었습니다.")
        print("insert 쿼리가 성공적으로 실행되었습니다.")
        
        # 변경사항 커밋
        conn.commit_changes()
        
    except Exception as e:
        print(f"Error in prediction_result_db_insert: {e}")
        logging.error(f"Error in prediction_result_db_insert: {e}")
        conn.rollback_changes()  # 오류 발생 시 롤백
        raise e
    
