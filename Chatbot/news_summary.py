from preprocess_article import preprocess_articles_large, preprocess_article
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import pandas as pd
import numpy as np
from datetime import datetime
import re
import sys
import os
import json

from connection import conn

def select_news(start_date, end_date):
    try:
        # conn.connect_to_database()
        query = f'''
        SELECT tk.KRNEWS_CODE, tk.GSTC_CODE, tk.KRNEWS_DATE, tk.KRNEWS_CONTENT
        FROM TB_KRNEWS tk
        WHERE KRNEWS_DATE BETWEEN '{start_date}' AND '{end_date}'
        '''
        conn.global_cursor.execute(query)
        df = pd.read_sql(query, conn.global_conn)
        
        return df
    except Exception as e:
        print(f"Error occurred while fetching data from database: {e}")
        return None
    finally:
        conn.close_database_connection()

def get_first_sentence(text):
    sentences = text.split('다.')
    for sentence in sentences:
        cleaned = sentence.strip()
        if cleaned and len(cleaned) > 10:
            return cleaned + '다.'
    return "적절한 첫 문장을 찾을 수 없습니다."

def summarize_text(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = inputs.to(device)
    
    summary_ids = model.generate(
        inputs['input_ids'], 
        max_length=200,
        min_length=30,
        length_penalty=1.0,
        num_beams=5,
        early_stopping=True,
        temperature=1.2,
        top_k=50,  
        top_p=0.9,
        no_repeat_ngram_size=3,
        do_sample=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def news_summary_json(start_date, end_date):
    # 데이터 가져오기
    result_df = select_news(start_date, end_date)

    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 경로 지정
    model_path = r"C:\big18\final\CHATBOT\kobart_summarization_model_v3"

    # 모델 로드
    model = BartForConditionalGeneration.from_pretrained(model_path)
    model.to(device)

    # 토크나이저 로드
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

    batch_size = 128
    results_list = []

    preprocess_articles = preprocess_articles_large(result_df['KRNEWS_CONTENT'])

    for i in range(0, len(preprocess_articles), batch_size):
        batch = preprocess_articles[i:i+batch_size]
        for j, text in enumerate(batch):
            preprocessed_summary = summarize_text(text, model, tokenizer, device)
            preprocessed_summary_first_sentence = get_first_sentence(preprocessed_summary)
            results_list.append({
                'KRNEWS_CODE': str(result_df['KRNEWS_CODE'].iloc[i+j]),
                'GSTC_CODE': str(result_df['GSTC_CODE'].iloc[i+j]),
                'KRNEWS_DATE': result_df['KRNEWS_DATE'].iloc[i+j].strftime('%Y-%m-%d %H:%M:%S'),
                'KRNEWS_CONTENT_SUMMARY': preprocessed_summary_first_sentence,
                'INSERTDATE': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            })

    # JSON 형식으로 변환
    json_data = json.dumps(results_list, ensure_ascii=False, indent=2)
    
    return json_data

def insert_json_summary_to_db(json_data):
    try:
        # JSON 문자열을 Python 객체로 변환
        data = json.loads(json_data)
        
        # 데이터베이스 연결
        conn.connect_to_database()
        
        # 삽입된 행 수를 추적
        inserted_rows = 0
        
        for item in data:
            query = """
            INSERT IGNORE INTO TB_KRNEWS_SUMMARY 
            (KRNEWS_CODE, GSTC_CODE, KRNEWS_DATE, KRNEWS_CONTENT_SUMMARY, INSERTDATE)
            VALUES (%s, %s, %s, %s, %s)
            """
            values = (
                item['KRNEWS_CODE'],
                item['GSTC_CODE'],
                item['KRNEWS_DATE'],
                item['KRNEWS_CONTENT_SUMMARY'],
                item['INSERTDATE']
            )
            conn.global_cursor.execute(query, values)
            inserted_rows += 1
        
        # 변경사항 커밋
        conn.commit_changes()
        print(f"{inserted_rows} rows inserted successfully.")
    except Exception as e:
        print(f"Error occurred while inserting data into database: {e}")
        conn.rollback_changes()
    finally:
        conn.close_database_connection()
