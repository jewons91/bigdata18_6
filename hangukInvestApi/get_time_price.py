from datetime import datetime
import sys
import os
import pandas as pd
import hangukInvestAPI.inquire_time_itemchartprice as iti
from dateutil.relativedelta import relativedelta
import time
from connection import conn
import logging

# 마지막 시간 기준 + 30분해서 데이터 가져오기
def get_last_time(stock_no, last_time):
    # 마지막 시간에서 30분 더하기
    last_time_dt = datetime.strptime(last_time, '%H%M%S')
    last_time_dt += relativedelta(minutes=30)
    last_time_plus30 = last_time_dt.strftime('%H%M%S')
    
    result = iti.get_time_price(stock_no, last_time_plus30)
    time.sleep(0.01)

    if result['output2'][0]:
        # output2는 분봉 관련, output1은 기본 정보
        output2_df = pd.DataFrame(result['output2'])
        output1 = result['output1']
        # output2_df에 output1 데이터 넣기
        output2_df[list(output1.keys())[:5]] = list(output1.values())[:5]
        # output2_df에 insertdate 넣기
        output2_df['insertdate'] = datetime.now()
        return output2_df, last_time_plus30
    else:
        logging.info('Json에 데이터가 없습니다.')
        return None

# 현재 시간 기준 데이터 가져오기
def get_current_time(stock_no, last_time):
    result = iti.get_time_price(stock_no)
    time.sleep(0.01)

    if result['output2'][0]:
        # output2는 분봉 관련, output1은 기본 정보
        output2_df = pd.DataFrame(result['output2'])
        output1 = result['output1']
        # output2_df에 output1 데이터 넣기
        output2_df[list(output1.keys())[:5]] = list(output1.values())[:5]
        # output2_df에 insertdate 넣기
        output2_df['insertdate'] = datetime.now()
        # ouput2_df에서 마지막시간 보다 늦은 시간만 남기기
        output2_select_df = output2_df[output2_df['stck_cntg_hour'] > last_time]
        return output2_select_df
    else:
        logging.info('Json에 데이터가 없습니다.')
        return None

def fetch_time_price(stock_no):
    # 필요 정보 초기화 및 선언
    last_time = '' # 적재된 마지막 시간
    time_now = datetime.now() # 현재 시간&날짜
    current_date = time_now.strftime('%Y%m%d') # 현재 날짜
    current_time = time_now.strftime('%H%M%S') # 현재 시간
    current_time_m30 = time_now - relativedelta(minutes=30)
    current_time_m30_str = current_time_m30.strftime('%H%M%S')
    
    # GSTC_CODE 및 INVEST_CODE 가져오기
    code_select_query = f'''
        SELECT a.GSTC_CODE, a.INVEST_CODE
        FROM TB_STOCKCLASSIFY a
        WHERE a.KSTC_CODE = '{stock_no}'
        '''

    try:
        conn.global_cursor.execute(code_select_query)
        # 결과를 데이터프레임으로 반환
        columns = [column[0] for column in conn.global_cursor.description]
        data = conn.global_cursor.fetchall()
        select_df = pd.DataFrame(data, columns=columns)
    except Exception as e:
        logging.error(f'SELECT 중 오류 발생: {e}')

    # DB에 들어있는 데이터 시간 및 날짜 확인
    time_select_query = f'''
        SELECT MAX(a.STCK_CNTG_HOUR) AS STCK_CNTG_HOUR
        FROM TB_MINSSTOCK a
        WHERE a.GSTC_CODE = '{select_df['GSTC_CODE'][0]}'
        AND a.STCK_BSOP_DATE = '{current_date}'
        '''

    try:
        conn.global_cursor.execute(time_select_query)
        # 결과를 데이터프레임으로 반환
        columns = [column[0] for column in conn.global_cursor.description]
        data = conn.global_cursor.fetchall()
        time_select_df = pd.DataFrame(data, columns=columns)
        last_time = time_select_df.iloc[0, 0]
    except Exception as e:
        logging.error(f'시간 체크 중 오류 발생: {e}')

    # 데이터 프레임 만들기
    final_df = pd.DataFrame()
    if last_time:
        if last_time < '153000':
            if last_time < '150100':
                while True:
                    if last_time >= '150000':
                        result_df = get_current_time(stock_no, last_time)
                        final_df = pd.concat([final_df, result_df], ignore_index=True)
                        break
                    elif last_time >= current_time_m30_str:
                        result_df = get_current_time(stock_no, last_time)
                        final_df = pd.concat([final_df, result_df], ignore_index=True)
                        break
                    else:
                        result_df, last_time = get_last_time(stock_no, last_time)
                        final_df = pd.concat([final_df, result_df], ignore_index=True)
            else:
                result_df = get_current_time(stock_no, last_time)
                final_df = pd.concat([final_df, result_df], ignore_index=True)
    else:
        last_time = '085900'
        while True:
            if last_time >= '150000':
                result_df = get_current_time(stock_no, last_time)
                final_df = pd.concat([final_df, result_df], ignore_index=True)
                break
            elif last_time >= current_time_m30_str:
                result_df = get_current_time(stock_no, last_time)
                final_df = pd.concat([final_df, result_df], ignore_index=True)
                break
            else:
                result_df, last_time = get_last_time(stock_no, last_time)
                final_df = pd.concat([final_df, result_df], ignore_index=True)

    if not final_df.empty:
        # 최종적으로 GSTC_CODE랑 INVEST_CODE 추가
        for column in select_df.columns:
            final_df[column] = select_df[column].iloc[0]
        
        json_data = final_df.to_json(orient='records', date_format='iso')
        return final_df, json_data
    else:
        logging.info('No Data')
        return None, None

def main():
    # DB 연결
    # conn.connect_to_database()

    # 주식 종목 번호 리스트로 가져오기
    all_code_select_query = '''
        SELECT a.KSTC_CODE
        FROM TB_STOCKCLASSIFY a
        '''

    # 가져오기
    try:
        conn.global_cursor.execute(all_code_select_query)
        # 결과를 데이터프레임으로 반환
        columns = [column[0] for column in conn.global_cursor.description]
        data = conn.global_cursor.fetchall()
        all_select_df = pd.DataFrame(data, columns=columns)
    except Exception as e:
        logging.error(f'SELECT 중 오류 발생: {e}')
    stock_no_list = all_select_df['KSTC_CODE'].values.tolist()

    for stock_no in stock_no_list:
        
        final_df, _ = fetch_time_price(stock_no)
        logging.info('### INSERT DATA ###')
        logging.info(final_df)
        logging.info('### INSERT DATA ###')
        if final_df is not None and not final_df.empty:
            # 데이터프레임에서 튜플 리스트 생|성
            data_tuples = [tuple(x) for x in final_df.to_numpy()]

            # TB_MINSSTOCK 테이블에 넣을 쿼리문 작성
            insert_query = '''
                INSERT INTO TB_MINSSTOCK
                (stck_bsop_date, stck_cntg_hour, stck_prpr, stck_oprc, stck_hgpr, stck_lwpr, cntg_vol, acml_tr_pbmn, prdy_vrss, prdy_vrss_sign, prdy_ctrt, stck_prdy_clpr, acml_vol, insertdate, GSTC_CODE, INVEST_CODE)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                '''

            # DB에 데이터 삽입
            try:
                conn.global_cursor.executemany(insert_query, data_tuples)
                conn.commit_changes()
                logging.info('데이터 적재 완료')
            except Exception as e:
                logging.error(f'데이터 적재 중 오류 발셍: {e}')
                conn.rollback_changes()
        else:
            logging.info(f'{stock_no}관련 데이터 없음. 다음 종목 시작')

    # DB 연결 닫기
    conn.close_database_connection()
