import pymysql # pip install pymysql == 1.1.1
import sys
from jproperties import Properties

global_conn = None
global_cursor = None

def load_db_config(file_path=r'C:\big18\final\DB\config.properties'):
    configs = Properties()
    with open(file_path, 'rb') as config_file:
        configs.load(config_file)
    
    return {
        'host': configs.get('database.host').data,
        'port': int(configs.get('database.port').data),
        'database': configs.get('database.database').data,
        'user': configs.get('database.user').data,
        'password': configs.get('database.password').data
    }

def connect_to_database():
    global global_conn, global_cursor
    try:
        db_config = load_db_config()
        global_conn = pymysql.connect(**db_config)
        global_cursor = global_conn.cursor()
        
        global_cursor.execute("SELECT VERSION()")
        db_version = global_cursor.fetchone()
        print(f"MariaDB 서버에 성공적으로 연결되었습니다. 서버 버전: {db_version[0]}")

        global_cursor.execute("SELECT DATABASE()")
        db_name = global_cursor.fetchone()[0]
        print(f"현재 사용 중인 데이터베이스: {db_name}")

    except pymysql.Error as e:
        print(f"MariaDB에 연결하는 데 실패했습니다: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")
        sys.exit(1)

def close_database_connection():
    global global_conn, global_cursor
    if global_cursor:
        global_cursor.close()
    if global_conn:
        global_conn.close()
        print("MariaDB 연결이 종료되었습니다.")
    
    global_conn = None
    global_cursor = None

def commit_changes():
    global global_conn
    if global_conn:
        global_conn.commit()
        print("변경사항이 커밋되었습니다.")
    else:
        print("데이터베이스 연결이 없습니다.")

def rollback_changes():
    global global_conn
    if global_conn:
        global_conn.rollback()
        print("변경사항이 롤백되었습니다.")
    else:
        print("데이터베이스 연결이 없습니다.")
