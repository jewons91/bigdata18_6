# from connection import conn
import conn
import yfinance as yf
import pandas as pd
import json
import logging

def fetch_select_glb_tb():
    try:
        logging.info('### fetch_select_glb_tb 시작 ###')
        conn.connect_to_database()
        query = 'SELECT tg.GLBSTC_CODE, tg.INVEST_CODE, tg.GLBSTC_TICKER  FROM TB_GLOBALSTOCKCLASSIFY tg'
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

def fetch_daily_glbstock_yfinance(db_data):
    data = []
    for row in db_data:
        GLBSTC_CODE, INVEST_CODE, GLBSTC_TICKER = row
        try:
            df = yf.download(GLBSTC_TICKER, period='5d')
            if df.empty:
                print(f"No data available for {GLBSTC_TICKER}")
                continue
            df = df.reset_index()
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df.insert(0, 'GLBSTC_CODE', GLBSTC_CODE)
            df.insert(1, 'INVEST_CODE', INVEST_CODE)
            df = df.rename(columns={
                'Date': 'GLBSTC_DATE',
                'Open': 'GLBSTC_OPEN',
                'High': 'GLBSTC_HIGH',
                'Low': 'GLBSTC_LOW',
                'Close': 'GLBSTC_CLOSE',
                'Adj Close': 'GLBSTC_ADJCLOSE',
                'Volume': 'GLBSTC_VOLUME'
            })
            data.append(df)
        except Exception as e:
            print(f"Error occurred while fetching data for {GLBSTC_CODE}: {e}")
    if data:
            df = pd.concat(data, ignore_index=True)
            
            # DataFrame을 JSON 문자열로 변환
            glbstock_json_data = df.to_json(orient='records', date_format='iso')
            
            return glbstock_json_data
    else:
        print("No data was successfully fetched.")
        return None

def fetch_min_glbstock_yfinance(db_data):
    data = []
    for row in db_data:
        GLBSTC_CODE, INVEST_CODE, GLBSTC_TICKER = row
        try:
            df = yf.download(GLBSTC_TICKER, period='1d', interval='1m', ignore_tz=True)
            if df.empty:
                print(f"No data available for {GLBSTC_TICKER}")
                continue
            df = df.reset_index()
            df['Datetime'] = df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df.insert(0, 'GLBSTC_CODE', GLBSTC_CODE)
            df.insert(1, 'INVEST_CODE', INVEST_CODE)
            df = df.rename(columns={
                'Datetime': 'GLBSTC_M_DATE',
                'Open': 'GLBSTC_OPEN',
                'High': 'GLBSTC_HIGH',
                'Low': 'GLBSTC_LOW',
                'Close': 'GLBSTC_CLOSE',
                'Adj Close': 'GLBSTC_ADJCLOSE',
                'Volume': 'GLBSTC_VOLUME'
            })
            data.append(df)
        except Exception as e:
            print(f"Error occurred while fetching data for {GLBSTC_CODE}: {e}")
    if data:
            df = pd.concat(data, ignore_index=True)
            
            # DataFrame을 JSON 문자열로 변환
            glbstock_json_data = df.to_json(orient='records', date_format='iso')
            
            return glbstock_json_data
    else:
        print("No data was successfully fetched.")
        return None
