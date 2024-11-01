# from connection import conn
import conn
import json

def insert_pdt_daily_data_to_db(json_data: str) -> None: # 일봉 데이터
    try:
        conn.connect_to_database()
        # JSON 문자열을 파이썬 객체로 파싱
        data = json.loads(json_data)
        
        query = """
            INSERT IGNORE INTO TB_DAILYPRODUCT
            (PDT_CODE, INVEST_CODE, PDT_DATE, PDT_OPEN, PDT_HIGH, PDT_LOW, PDT_CLOSE, PDT_ADJClOSE, PDT_VOLUME)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # JSON 데이터를 리스트로 변환
        values = [
            (
                item['PDT_CODE'],
                item['INVEST_CODE'],
                item['PDT_DATE'],
                item['PDT_OPEN'],
                item['PDT_HIGH'],
                item['PDT_LOW'],
                item['PDT_CLOSE'],
                item['PDT_ADJCLOSE'],
                item['PDT_VOLUME']
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


def insert_pdt_min_data_to_db(json_data: str) -> None: # 분봉 데이터
    try:
        conn.connect_to_database()
        # JSON 문자열을 파이썬 객체로 파싱
        data = json.loads(json_data)
        
        query = """
            INSERT IGNORE INTO TB_MINPRODUCT
            (PDT_CODE, INVEST_CODE, PDT_M_DATE, PDT_OPEN, PDT_HIGH, PDT_LOW, PDT_CLOSE, PDT_ADJClOSE, PDT_VOLUME)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # JSON 데이터를 리스트로 변환
        values = [
            (
                item['PDT_CODE'],
                item['INVEST_CODE'],
                item['PDT_M_DATE'],
                item['PDT_OPEN'],
                item['PDT_HIGH'],
                item['PDT_LOW'],
                item['PDT_CLOSE'],
                item['PDT_ADJCLOSE'],
                item['PDT_VOLUME']
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
