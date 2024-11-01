import torch
import pandas as pd
import numpy as np
import re
from hanja import translate # pip install hanja 0.15.1
from tqdm import tqdm


def process_hanja(text):
    # 한자-한글 병기 패턴 찾기 (예: 韓國(한국))
    pattern = r'([\u4e00-\u9fff]+)\(([\uac00-\ud7a3]+)\)'
    
    def replace_hanja(match):
        hanja = match.group(1)
        hangul_text = match.group(2)
        if hangul_text == translate(hanja, 'substitution'):
            return hangul_text
        else:
            return f"{hangul_text}({hanja})"
    
    # 패턴에 맞는 부분 대체
    text = re.sub(pattern, replace_hanja, text)
    
    # 남은 한자를 한글로 변환
    return translate(text, 'substitution')

def improve_text_normalization(text):
    # 괄호 안의 부가 설명 제거
    text = re.sub(r'\([^)]*\)', '', text)
    
    # 큰따옴표, 작은따옴표 정리
    text = re.sub(r'["""]', '"', text)  # 큰따옴표 대체
    text = re.sub(r"[''']", "'", text)  # 작은따옴표 대체
    
    return text

def clean_korean_news_text(text):
    patterns = [
        # HTML 태그 제거
        r'<[^>]+>',
        
        # URL 제거
        r'http\S+|www\.\S+',
        
        # 이메일 주소 제거 (개선된 패턴)
        r'\S+\s*@\s*\S+\.\S+',
        
        # 기자 정보 제거 패턴
        r'\(.*?=.*?기자\)',  # (서울=뉴스) 홍길동 기자
        r'\[.*?=.*?기자\]',  # [서울=뉴스]홍길동 기자
        r'\[.*?=.*?기자\].*',  # [뉴스 = 홍길동 기자]
        r'\(.*?=.*?\).*?기자',  # (서울=뉴스1) 홍길동 기자
        r'\[.*?=.*?\].*?기자',  # [서울=뉴스1] 홍길동 기자
        r'\[.*?기자\]',  # [홍길동 기자]
        r'\(.*?기자\)',  # (홍길동 기자)
        # r'【.*?기자】',  # 【홍길동 기자】
        # r'『.*?기자』',  # 『홍길동 기자』
        # r'▶.*?기자',    # ▶홍길동 기자
        # r'■.*?기자',    # ■홍길동 기자
        # r'●.*?기자',    # ●홍길동 기자
        r'\S+\s+기자\s+[\w.]+\s*@\s*\S+',  # 홍길동 기자 email@example.com (공백 허용)
        r'\S+\s+기자$',  # 문장 끝에 있는 '홍길동 기자'
        r'^[가-힣]+\s+기자\s*=*',  # 문장 시작에 있는 '홍길동 기자 ='
        r'^\([가-힣]+\s+기자\)\s*=*',  # 문장 시작에 있는 '(홍길동 기자) ='
        r'[가-힣]{2,4}\s+기자',  # 2-4글자 한글 이름 뒤에 공백과 '기자'가 오는 경우
        r'[가-힣]{2,4}\s+기자[^\w]',  # 이름 기자 뒤에 비단어 문자가 오는 경우 (문장 중간)
        r'[가-힣]{2,4}\s+기자\s',  # 이름 기자 뒤에 공백이 오는 경우
        r'기자\s+[가-힣]{2,4}',  # '기자' 뒤에 공백과 2-4글자 한글 이름이 오는 경우
        r'[^\w][가-힣]{2,4}\s+기자',  # 비단어 문자 + 이름 + 공백 + 기자
        r'[가-힣]{2,4}기자$',  # 문장 끝에 있는 '홍길동기자' (공백 없음)
        r'[가-힣]{2,4}기자[^\w]',  # '홍길동기자,' 와 같이 이름과 '기자'가 붙어있고 뒤에 비단어 문자가 오는 경우
        
        # 저작권 및 웹마스터 정보 제거 패턴
        r'Copyright.*?All Rights Reserved',
        r'Contact Webmaster for more Information',
        r'Copyright\s*\(c\)\s*The Korea Economic Daily',
        r'All Rights Reserved',
        r'Contact.*?for more information',
        r'Copyright.*?([\.\n]|$)',
        r'All rights reserved.*?([\.\n]|$)',
    ]
    
    # 모든 패턴 적용
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 특수 문자 제거 (일부 구두점 유지)
    # text = re.sub(r'[^\w\s\.\,\!\?\"\'\-\(\)]', '', text)
    
    # 연속된 공백을 하나의 공백으로 대체
    # text = re.sub(r'\s+', ' ', text)
    
    # 연속된 줄바꿈을 하나의 공백으로 대체
    text = re.sub(r'\n+', ' ', text)
    
    # 앞뒤 공백 제거
    return text.strip()

def preprocess_article(article):
    if not isinstance(article, str):
        try:
            article = str(article)
        except:
            return ""
    
    article = process_hanja(article)
    article = improve_text_normalization(article)
    article = clean_korean_news_text(article)
    
    return article.strip()

def preprocess_articles_large(articles, batch_size=100):
    if isinstance(articles, pd.Series):
        total_articles = len(articles)
        processed_articles = pd.Series(index=articles.index, dtype='object')

        for i in tqdm(range(0, total_articles, batch_size), desc="Processing articles"):
            batch = articles.iloc[i:i+batch_size]
            processed_batch = batch.apply(preprocess_article)
            processed_articles.iloc[i:i+batch_size] = processed_batch

        return processed_articles
    else:
        raise ValueError("Input must be a pandas Series")
