
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta


# df = pd.read_csv('./000660.csv',encoding='cp949')
df = pd.read_csv('./SK하이닉스.csv')


def divide_volumes(df): # 거래량 / 상장 주식수
    try:
        divisor = 728002365 # 상장 주식수
        df['CNTG_VOL'] = df['CNTG_VOL'] * 1000000 / divisor
        df['CNTG_VOL'] = round(df['CNTG_VOL'], 2)
    except KeyError:
        print(f"코드 {df['code']}를 찾지 못했습니다.")
        input()
    return df

df = divide_volumes(df)


# Unique 코드 및 카운트 계산
counts = df['GSTC_CODE'].value_counts()
counts_dict = counts.to_dict()


# 결과를 저장할 리스트
results = []

for key in tqdm(counts_dict.keys()):
    value = counts_dict[key]
    for i in range(value - 12):   # 예측 일 -1
        start_idx = counts_dict[key] - value + i
        end_idx = start_idx + 9  # 묶을 일 -1

        # 벡터화하여 연산
        vectorList = []
        
        for col in ['STCK_PRPR','STCK_HGPR','STCK_LWPR']:
            diff = (df[col].values[start_idx:end_idx] - df[col].values[start_idx + 1:end_idx + 1]) / df[col].values[start_idx:end_idx] * 10000
            vectorList.extend(round(d, 4) for d in diff)
        diff = (df['CNTG_VOL'].values[start_idx:end_idx] - df['CNTG_VOL'].values[start_idx + 1:end_idx + 1])
        vectorList.extend(round(d, 2) for d in diff)

        # for col in ['STCK_CLPR']:
        #     diff = (df[col].values[start_idx:end_idx] - df[col].values[start_idx + 1:end_idx + 1]) / df[col].values[start_idx:end_idx] * 100
        #     vectorList.extend(round(d, 4) for d in diff)
        # diff = (df['ACML_VOL'].values[start_idx:end_idx] - df['ACML_VOL'].values[start_idx + 1:end_idx + 1]) 
        # vectorList.extend(round(d, 4) for d in diff)
        
        openValue = df['STCK_PRPR'][start_idx + 12] - df['STCK_PRPR'][start_idx + 9]
        if openValue > 0:
            vectorList.append(2.0)
        elif openValue == 0:
            vectorList.append(1.0)
        else:
            vectorList.append(0.0)

        results.append(vectorList)

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(results)

# CSV 파일에 저장
results_df.to_csv('000660Vector4.csv', index=False)
