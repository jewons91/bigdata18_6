# from connection import conn
import conn
import json

def insert_fred_data_to_db(json_data: str) -> None:
    try:
        conn.connect_to_database()
        # JSON 문자열을 파이썬 객체로 파싱
        data = json.loads(json_data)
        
        query = """
            INSERT IGNORE INTO TB_INDICATOR
            (USINDICATOR_CODE, INVEST_CODE, USINDICATOR_DATE, USINDICATOR_VALUE, INSERTDATE)
            VALUES (%s, %s, %s, %s, %s)
        """
        
        # JSON 데이터를 리스트로 변환
        values = [
            (
                item['USINDICATOR_CODE'],
                item['INVEST_CODE'],
                item['USINDICATOR_DATE'],
                item['USINDICATOR_VALUE'],
                item['INSERTDATE']
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