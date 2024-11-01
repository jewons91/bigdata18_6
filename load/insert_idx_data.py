import conn
import json

def insert_idx_daily_data_to_db(json_data: str) -> None:
    try:
        conn.connect_to_database()
        
        # JSON 문자열을 파이썬 객체로 파싱
        data = json.loads(json_data)
        
        query = """
            INSERT IGNORE INTO TB_DAILYINDEX
            (GINDEX_CODE, INVEST_CODE, USINDEX_DATE, USINDEX_OPEN, USINDEX_HIGH, USINDEX_LOW, USINDEX_CLOSE, USINDEX_ADJCLOSE, USINDEX_VOLUME)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # JSON 데이터를 리스트로 변환
        values = [
            (
                item['GINDEX_CODE'],
                item['INVEST_CODE'],
                item['USINDEX_DATE'],
                item['USINDEX_OPEN'],
                item['USINDEX_HIGH'],
                item['USINDEX_LOW'],
                item['USINDEX_CLOSE'],
                item['USINDEX_ADJCLOSE'],
                item['USINDEX_VOLUME']
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

def insert_idx_min_data_to_db(json_data: str) -> None:
    try:
        conn.connect_to_database()
        
        # JSON 문자열을 파이썬 객체로 파싱
        data = json.loads(json_data)
        
        query = """
            INSERT IGNORE INTO TB_MININDEX
            (GINDEX_CODE, INVEST_CODE, USINDEX_M_DATE, USINDEX_OPEN, USINDEX_HIGH, USINDEX_LOW, USINDEX_CLOSE, USINDEX_ADJCLOSE, USINDEX_VOLUME)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # JSON 데이터를 리스트로 변환
        values = [
            (
                item['GINDEX_CODE'],
                item['INVEST_CODE'],
                item['USINDEX_M_DATE'],
                item['USINDEX_OPEN'],
                item['USINDEX_HIGH'],
                item['USINDEX_LOW'],
                item['USINDEX_CLOSE'],
                item['USINDEX_ADJCLOSE'],
                item['USINDEX_VOLUME']
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
