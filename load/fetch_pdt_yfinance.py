# from connection import conn
import conn
import yfinance as yf
import pandas as pd
import json
import logging

def fetch_select_pdt_tb(): # CLASSIFY 테이블 조회
    try:
        logging.info('### fetch_select_pdt_tb 시작 ###')
        conn.connect_to_database()
        query = 'SELECT tp.PDT_CODE , tp.INVEST_CODE , tp.PDT_TICKER FROM TB_PRODUCTCLASSIFY tp'
        conn.global_cursor.execute(query)
        db_data = [list(row) for row in conn.global_cursor.fetchall()]
        logging.info('########## 데이터 확인 ##########')
        logging.info(db_data)
        logging.info('########## 데이터 확인 ##########')
        
        return db_data
    except Exception as e:
        print(f"Error occurred while fetching data from database: {e}")
        return None
    finally:
        conn.close_database_connection()

def fetch_daily_pdt_yfinance(db_data): # 일봉 데이터
    data = []
    for row in db_data:
        PDT_CODE, INVEST_CODE, PDT_TICKER = row
        try:
            df = yf.download(PDT_TICKER, period='5d')
            if df.empty:
                print(f"No data available for {PDT_TICKER}")
                continue
            df = df.reset_index()
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df.insert(0, 'PDT_CODE', PDT_CODE)
            df.insert(1, 'INVEST_CODE', INVEST_CODE)
            df = df.rename(columns={
                'Date': 'PDT_DATE',
                'Open': 'PDT_OPEN',
                'High': 'PDT_HIGH',
                'Low': 'PDT_LOW',
                'Close': 'PDT_CLOSE',
                'Adj Close': 'PDT_ADJCLOSE',
                'Volume': 'PDT_VOLUME'
            })
            data.append(df)
        except Exception as e:
            print(f"Error occurred while fetching data for {PDT_CODE}: {e}")
    if data:
            df = pd.concat(data, ignore_index=True)
            
            # DataFrame을 JSON 문자열로 변환
            idx_json_data = df.to_json(orient='records', date_format='iso')
            
            return idx_json_data
    else:
        print("No data was successfully fetched.")
        return None

def fetch_min_pdt_yfinance(db_data): # 분봉 데이터
    data = []
    for row in db_data:
        PDT_CODE, INVEST_CODE, PDT_TICKER = row
        try:
            df = yf.download(PDT_TICKER, period='1d', interval='1m', ignore_tz=True)
            if df.empty:
                print(f"No data available for {PDT_TICKER}")
                continue
            df = df.reset_index()
            df['Datetime'] = df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df.insert(0, 'PDT_CODE', PDT_CODE)
            df.insert(1, 'INVEST_CODE', INVEST_CODE)
            df = df.rename(columns={
                'Datetime': 'PDT_M_DATE',
                'Open': 'PDT_OPEN',
                'High': 'PDT_HIGH',
                'Low': 'PDT_LOW',
                'Close': 'PDT_CLOSE',
                'Adj Close': 'PDT_ADJCLOSE',
                'Volume': 'PDT_VOLUME'
            })
            data.append(df)
        except Exception as e:
            print(f"Error occurred while fetching data for {PDT_CODE}: {e}")
    if data:
            df = pd.concat(data, ignore_index=True)
            
            # DataFrame을 JSON 문자열로 변환
            idx_json_data = df.to_json(orient='records', date_format='iso')
            
            return idx_json_data
    else:
        print("No data was successfully fetched.")
        return None
