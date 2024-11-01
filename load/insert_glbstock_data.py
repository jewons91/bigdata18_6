# from connection import conn
import conn
import json

def insert_glbstock_daily_data_to_db(json_data: str) -> None:
    try:
        conn.connect_to_database()
        # JSON 문자열을 파이썬 객체로 파싱
        data = json.loads(json_data)
        
        query = """
            INSERT IGNORE INTO TB_DAILYGLBSTOCK
            (GLBSTC_CODE, INVEST_CODE, GLBSTC_DATE, GLBSTC_OPEN, GLBSTC_HIGH, GLBSTC_LOW, GLBSTC_CLOSE, GLBSTC_ADJClOSE, GLBSTC_VOLUME)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # JSON 데이터를 리스트로 변환
        values = [
            (
                item['GLBSTC_CODE'],
                item['INVEST_CODE'],
                item['GLBSTC_DATE'],
                item['GLBSTC_OPEN'],
                item['GLBSTC_HIGH'],
                item['GLBSTC_LOW'],
                item['GLBSTC_CLOSE'],
                item['GLBSTC_ADJCLOSE'],
                item['GLBSTC_VOLUME']
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


def insert_glbstock_min_data_to_db(json_data: str) -> None:
    try:
        conn.connect_to_database()
        # JSON 문자열을 파이썬 객체로 파싱
        data = json.loads(json_data)
        
        query = """
            INSERT IGNORE INTO TB_MINGLBSTOCK
            (GLBSTC_CODE, INVEST_CODE, GLBSTC_M_DATE, GLBSTC_OPEN, GLBSTC_HIGH, GLBSTC_LOW, GLBSTC_CLOSE, GLBSTC_ADJClOSE, GLBSTC_VOLUME)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # JSON 데이터를 리스트로 변환
        values = [
            (
                item['GLBSTC_CODE'],
                item['INVEST_CODE'],
                item['GLBSTC_M_DATE'],
                item['GLBSTC_OPEN'],
                item['GLBSTC_HIGH'],
                item['GLBSTC_LOW'],
                item['GLBSTC_CLOSE'],
                item['GLBSTC_ADJCLOSE'],
                item['GLBSTC_VOLUME']
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
