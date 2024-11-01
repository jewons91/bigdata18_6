import pandas as pd
import numpy as np
import conn
import logging
from sklearn.preprocessing import StandardScaler
import pickle
import torch
import torch.nn as nn
import json
from datetime import datetime

def get_code_list():
    try:
        conn.connect_to_database()
        query = f'''
            SELECT ts.KSTC_CODE
              FROM TB_STOCKCLASSIFY ts
        '''
        conn.global_cursor.execute(query)
        df = pd.read_sql(query, conn.global_conn)
        codes = df['KSTC_CODE'].values.tolist()
        return codes
    except Exception as e:
        logging.error(f'Error occurred while fetching data from database: {e}')
        return None
    finally:
        conn.close_database_connection()

def get_gstc_code(code):
    try:
        conn.connect_to_database()
        query = f'''
            SELECT ts.GSTC_CODE, ts.INVEST_CODE
              FROM TB_STOCKCLASSIFY ts
             WHERE ts.KSTC_CODE = '{code}'
        '''
        
        conn.global_cursor.execute(query)
        df = pd.read_sql(query, conn.global_conn)
        
        return df
    except Exception as e:
        logging.error(f'Error occurred while fetching data from database: {e}')
        return None
    finally:
        conn.close_database_connection()

def select_data(gstc_code):
    try:
        conn.connect_to_database()
        query = f'''
            SELECT td.STCK_BSOP_DATE, td.STCK_CLPR, td.STCK_OPRC, td.STCK_HGPR, td.STCK_LWPR, td.ACML_VOL, td.ACML_TR_PBMN
              FROM TB_DAILYSTOCK td
             WHERE td.GSTC_CODE = '{gstc_code}'
             ORDER BY td.STCK_BSOP_DATE DESC
             LIMIT 100
        '''
        
        conn.global_cursor.execute(query)
        df = pd.read_sql(query, conn.global_conn)
        
        return df
    except Exception as e:
        logging.error(f'Error occurred while fetching data from database: {e}')
        return None
    finally:
        conn.close_database_connection()

def data_preprocess(code):
    gstc_df = get_gstc_code(code)
    if gstc_df is None or gstc_df.empty:
        raise ValueError("GSTC_CODE를 가져올 수 없습니다.")
    gstc_code = gstc_df.iloc[0, 0]
    data = select_data(gstc_code)
    data = data.sort_values(by='STCK_BSOP_DATE', ascending=True).reset_index(drop=True)
    
    if data is None or data.empty:
        raise ValueError("데이터를 가져올 수 없습니다.")
    
    data = data.drop_duplicates(subset='STCK_BSOP_DATE')
    
    data['STCK_BSOP_DATE'] = pd.to_datetime(data['STCK_BSOP_DATE'], format='%Y%m%d')
    data = data.sort_values('STCK_BSOP_DATE').reset_index(drop=True)

    columns_to_convert = ['STCK_CLPR', 'STCK_OPRC', 'STCK_HGPR', 'STCK_LWPR', 'ACML_VOL', 'ACML_TR_PBMN']
    data[columns_to_convert] = data[columns_to_convert].astype(float)
    
    # 이동 평균
    data['MA5'] = data['STCK_CLPR'].rolling(window=5).mean()
    data['MA10'] = data['STCK_CLPR'].rolling(window=10).mean()
    data['MA20'] = data['STCK_CLPR'].rolling(window=20).mean()
    data['MA50'] = data['STCK_CLPR'].rolling(window=50).mean()

    # 지수 이동 평균
    data['EMA5'] = data['STCK_CLPR'].ewm(span=5, adjust=False).mean()
    data['EMA10'] = data['STCK_CLPR'].ewm(span=10, adjust=False).mean()
    data['EMA20'] = data['STCK_CLPR'].ewm(span=20, adjust=False).mean()

    # 상대 강도 지수 (RSI)
    delta = data['STCK_CLPR'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    data['RSI'] = 100 - (100 / (1 + rs))

    # 이동 평균 수렴 발산 (MACD)
    exp1 = data['STCK_CLPR'].ewm(span=12, adjust=False).mean()
    exp2 = data['STCK_CLPR'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # 볼린저 밴드
    data['20_day_MA'] = data['STCK_CLPR'].rolling(window=20).mean()
    data['20_day_STD'] = data['STCK_CLPR'].rolling(window=20).std()
    data['Bollinger_High'] = data['20_day_MA'] + (data['20_day_STD'] * 2)
    data['Bollinger_Low'] = data['20_day_MA'] - (data['20_day_STD'] * 2)

    # 스토캐스틱 오실레이터
    low14 = data['STCK_LWPR'].rolling(window=14).min()
    high14 = data['STCK_HGPR'].rolling(window=14).max()
    data['%K'] = 100 * ((data['STCK_CLPR'] - low14) / (high14 - low14))

    data['%D'] = data['%K'].rolling(window=3).mean()

    # 기술 지표 계산으로 인해 발생하는 NaN 값 제거
    data = data.dropna().reset_index(drop=True)
    
    return data

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, num_timesteps, num_features, feature_size=128, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.feature_size = feature_size

        self.pos_encoder = PositionalEncoding(feature_size, dropout, num_timesteps)
        encoder_layers = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.embedding = nn.Linear(num_features, feature_size)
        self.decoder = nn.Linear(feature_size, 3)  # 출력 클래스 수를 3으로 변경

    def forward(self, src):
        # src shape: [batch_size, seq_len, num_features]
        src = self.embedding(src) * np.sqrt(self.feature_size)
        src = self.pos_encoder(src)
        # Transformer expects input of shape (seq_len, batch_size, feature_size)
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        # 마지막 시점의 출력 사용
        output = output[-1, :, :]
        output = self.decoder(output)
        return output

def load_scaler(scaler_path='./scaler.pkl'):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def load_model(model_path='./model.pth', num_timesteps=30, num_features=22, feature_size=64, num_layers=2, dropout=0.1, device='cuda'):
    model = TransformerModel(
        num_timesteps=num_timesteps, 
        num_features=num_features, 
        feature_size=feature_size, 
        num_layers=num_layers, 
        dropout=dropout
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def make_prediction(code, scaler, model, device='cuda'):
    # 데이터 전처리
    processed_data = data_preprocess(code)
    
    features = processed_data.drop(['STCK_BSOP_DATE'], axis=1)
    
    # numpy 배열로 변환
    features = features.values
    
    sequence_length = 30
    
    # 시퀀스 생성
    X = []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
    
    X = np.array(X)
    X = X[-1]
    X = np.expand_dims(X, axis=0)
    
    if len(X) == 0:
        raise ValueError("입력 데이터가 시퀀스 생성에 충분하지 않습니다.")
    
    # 스케일링
    num_samples, num_timesteps, num_features = X.shape
    X_reshaped = X.reshape(-1, num_features)
    X_scaled = scaler.transform(X_reshaped)
    X_scaled = X_scaled.reshape(num_samples, num_timesteps, num_features)
    
    # 텐서로 변환
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    # 예측
    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)
    
    return preds.cpu().numpy().tolist(), torch.softmax(outputs, 1).cpu().numpy().tolist()

def prediction_to_json(code, model_path='./model.pth', scaler_path='./scaler.pkl'):
    answer, predictions = make_prediction(code, load_scaler(scaler_path=scaler_path), load_model(model_path=model_path))
    df = get_gstc_code(code)
    gstc_code = df.iloc[0, 0]
    invest_code = df.iloc[0, 1]
    
    result_list = []
    result_list.append({
        'GSTC_CODE': gstc_code,
        'INVEST_CODE': invest_code,
        'PREDICT_RISE_RATE': predictions[0][2],
        'PREDICT_NO_CHANGE_RATE': predictions[0][1],
        'PREDICT_FALL_RATE': predictions[0][0],
        'PREDICT_TIME': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    })
    
    json_data = json.dumps(result_list, ensure_ascii=False, indent=2)
    return json_data

def insert_json_to_db(json_data):
    try:
        data = json.loads(json_data)
        
        conn.connect_to_database()
        
        inserted_rows = 0
        
        for item in data:
            query = f'''
                INSERT IGNORE INTO TB_STOCK_PREDICT
                (GSTC_CODE, INVEST_CODE, PREDICT_RISE_RATE, PREDICT_NO_CHANGE_RATE, PREDICT_FALL_RATE, PREDICT_TIME)
                VALUES (%s, %s, %s, %s, %s, %s)
            '''
            values = (
                item['GSTC_CODE'],
                item['INVEST_CODE'],
                item['PREDICT_RISE_RATE'],
                item['PREDICT_NO_CHANGE_RATE'],
                item['PREDICT_FALL_RATE'],
                item['PREDICT_TIME']
            )
            conn.global_cursor.execute(query, values)
            inserted_rows += 1
        
        conn.commit_changes()
        logging.info(f'{inserted_rows} rows inserted successfully')
    except Exception as e:
        logging.error(f'Error occurred while inserting data into database: {e}')
        conn.rollback_changes()
    finally:
        conn.close_database_connection()
