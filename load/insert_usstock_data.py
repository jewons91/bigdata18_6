# from connection import conn
import conn
import json

def insert_usstock_daily_data_to_db(json_data: str) -> None:
    try:
        conn.connect_to_database()
        # JSON 문자열을 파이썬 객체로 파싱
        data = json.loads(json_data)
        
        query = """
            INSERT IGNORE INTO TB_DAILYUSSTOCK
            (USSTC_CODE, INVEST_CODE, USSTC_DATE, USSTC_OPEN, USSTC_HIGH, USSTC_LOW, USSTC_CLOSE, USSTC_ADJClOSE, USSTC_VOLUME)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # JSON 데이터를 리스트로 변환
        values = [
            (
                item['USSTC_CODE'],
                item['INVEST_CODE'],
                item['USSTC_DATE'],
                item['USSTC_OPEN'],
                item['USSTC_HIGH'],
                item['USSTC_LOW'],
                item['USSTC_CLOSE'],
                item['USSTC_ADJCLOSE'],
                item['USSTC_VOLUME']
            )
            for item in data
        ]
        
        conn.global_cursor.executemany(query, values)
        conn.commit_changes()
        print(f"Successfully inserted {len(values)} records.")
    except json.JSONDecodeError as je:
        print(f"Error decoding JSON: {je}")
    except Exception as e:
        print(f"Error occurred while inserting data: {e}")
        conn.rollback_changes()
    finally:
        conn.close_database_connection()


def insert_usstock_min_data_to_db(json_data: str) -> None:
    try:
        conn.connect_to_database()
        # JSON 문자열을 파이썬 객체로 파싱
        data = json.loads(json_data)
        
        query = """
            INSERT IGNORE INTO TB_MINUSSTOCK
            (USSTC_CODE, INVEST_CODE, USSTC_M_DATE, USSTC_OPEN, USSTC_HIGH, USSTC_LOW, USSTC_CLOSE, USSTC_ADJClOSE, USSTC_VOLUME)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # JSON 데이터를 리스트로 변환
        values = [
            (
                item['USSTC_CODE'],
                item['INVEST_CODE'],
                item['USSTC_M_DATE'],
                item['USSTC_OPEN'],
                item['USSTC_HIGH'],
                item['USSTC_LOW'],
                item['USSTC_CLOSE'],
                item['USSTC_ADJCLOSE'],
                item['USSTC_VOLUME']
            )
            for item in data
        ]
        
        conn.global_cursor.executemany(query, values)
        conn.commit_changes()
        print(f"Successfully inserted {len(values)} records.")
    except json.JSONDecodeError as je:
        print(f"Error decoding JSON: {je}")
    except Exception as e:
        print(f"Error occurred while inserting data: {e}")
        conn.rollback_changes()
    finally:
        conn.close_database_connection()
