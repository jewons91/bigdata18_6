# from connection import conn
import conn
import yfinance as yf
import pandas as pd
import json
import logging

def fetch_select_usstock_tb():
    try:
        logging.info('### fetch_select_pdt_tb 시작 ###')
        conn.connect_to_database()
        query = 'SELECT tu.USSTC_CODE, tu.INVEST_CODE, tu.USSTC_TICKER FROM TB_USSTOCKCLASSIFY tu'
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

def fetch_daily_usstock_yfinance(db_data):
    data = []
    for row in db_data:
        USSTC_CODE, INVEST_CODE, USSTC_TICKER = row
        try:
            df = yf.download(USSTC_TICKER, period='5d')
            if df.empty:
                print(f"No data available for {USSTC_TICKER}")
                continue
            df = df.reset_index()
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df.insert(0, 'USSTC_CODE', USSTC_CODE)
            df.insert(1, 'INVEST_CODE', INVEST_CODE)
            df = df.rename(columns={
                'Date': 'USSTC_DATE',
                'Open': 'USSTC_OPEN',
                'High': 'USSTC_HIGH',
                'Low': 'USSTC_LOW',
                'Close': 'USSTC_CLOSE',
                'Adj Close': 'USSTC_ADJCLOSE',
                'Volume': 'USSTC_VOLUME'
            })
            data.append(df)
        except Exception as e:
            print(f"Error occurred while fetching data for {USSTC_CODE}: {e}")
    if data:
            df = pd.concat(data, ignore_index=True)
            
            # DataFrame을 JSON 문자열로 변환
            usstock_json_data = df.to_json(orient='records', date_format='iso')
            
            return usstock_json_data
    else:
        print("No data was successfully fetched.")
        return None

def fetch_min_usstock_yfinance(db_data):
    data = []
    for row in db_data:
        USSTC_CODE, INVEST_CODE, USSTC_TICKER = row
        try:
            df = yf.download(USSTC_TICKER, period='1d', interval='1m', ignore_tz=True)
            if df.empty:
                print(f"No data available for {USSTC_TICKER}")
                continue
            df = df.reset_index()
            df['Datetime'] = df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df.insert(0, 'USSTC_CODE', USSTC_CODE)
            df.insert(1, 'INVEST_CODE', INVEST_CODE)
            df = df.rename(columns={
                'Datetime': 'USSTC_M_DATE',
                'Open': 'USSTC_OPEN',
                'High': 'USSTC_HIGH',
                'Low': 'USSTC_LOW',
                'Close': 'USSTC_CLOSE',
                'Adj Close': 'USSTC_ADJCLOSE',
                'Volume': 'USSTC_VOLUME'
            })
            data.append(df)
        except Exception as e:
            print(f"Error occurred while fetching data for {USSTC_CODE}: {e}")
    if data:
            df = pd.concat(data, ignore_index=True)
            
            # DataFrame을 JSON 문자열로 변환
            usstock_json_data = df.to_json(orient='records', date_format='iso')
            
            return usstock_json_data
    else:
        print("No data was successfully fetched.")
        return None
