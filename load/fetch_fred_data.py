import pandas as pd
from fredapi import Fred
from datetime import datetime, timedelta
import sqlite3
import conn
import json

def fetch_economic_indicators():
    api_key = 'ae57cb85d7fd94daddf358a159572ca9'
    
    # 데이터베이스에서 지표 정보 가져오기
    try:
        conn.connect_to_database()
        query = 'SELECT USINDICATOR_CODE, INVEST_CODE, USINDICATOR_TICKER FROM TB_INDICATORCLASSIFY'
        conn.global_cursor.execute(query)
        indicators = list(conn.global_cursor.fetchall())
    except Exception as e:
        print(f"데이터베이스에서 데이터를 가져오는 중 오류가 발생했습니다: {e}")
        return None
    finally:
        conn.close_database_connection()

    # FRED API 초기화
    fred = Fred(api_key=api_key)
    
    # 날짜 범위 설정
    end = datetime.now()
    start = end - timedelta(days=3)

    # 결과를 저장할 빈 DataFrame 생성
    results = pd.DataFrame()

    # 각 지표에 대해 데이터 가져오기
    for indicator in indicators:
        USINDICATOR_CODE, INVEST_CODE, USINDICATOR_TICKER = indicator
        try:
            series = fred.get_series(USINDICATOR_TICKER, start, end)
            if not series.empty:
                temp_df = pd.DataFrame({
                    'USINDICATOR_CODE': USINDICATOR_CODE,
                    'INVEST_CODE': INVEST_CODE,
                    'USINDICATOR_DATE': series.index,
                    'USINDICATOR_VALUE': series.values,
                    'INSERTDATE': datetime.now()
                })
                results = pd.concat([results, temp_df], ignore_index=True)
        except Exception as e:
            print(f"지표 {USINDICATOR_CODE}의 데이터를 가져오는 중 오류가 발생했습니다: {e}")

    # 모든 데이터를 포함한 DataFrame을 JSON 문자열로 변환
    fred_json_data = results.to_json(orient='records', date_format='iso')
            
    return fred_json_data

data = fetch_economic_indicators()
print(data)