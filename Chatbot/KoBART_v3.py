import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['documents']

class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = ' '.join([sent['sentence'] for sent in item['text'][0]])
        target_text = item['abstractive'][0]

        inputs = self.tokenizer(input_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        targets = self.tokenizer(target_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

# 기존 모델 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained('./kobart_summarization_model_v2')
model = BartForConditionalGeneration.from_pretrained('./kobart_summarization_model_v2')

# 데이터 로드 및 데이터셋 생성
train_data = load_data('./train_original (3).json')
valid_data = load_data('./valid_original (3).json')

train_dataset = SummarizationDataset(train_data, tokenizer)
valid_dataset = SummarizationDataset(valid_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)



def train(model, train_loader, valid_loader, epochs=3, lr=2e-5, max_grad_norm=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    
    # 총 학습 스텝 수 계산
    total_steps = len(train_loader) * epochs
    
    # Learning rate scheduler 설정
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Loss가 nan이 아닌지 확인
            if not torch.isnan(loss):
                total_loss += loss.item()
                loss.backward()
                
                # Gradient clipping 적용
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                scheduler.step()
            else:
                print("Warning: NaN loss encountered. Skipping this batch.")

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, 평균 학습 손실: {avg_train_loss:.4f}')

        # 검증
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc='검증'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                if not torch.isnan(loss):
                    total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(valid_loader)
        print(f'Epoch {epoch+1}/{epochs}, 평균 검증 손실: {avg_val_loss:.4f}')

    return model

# 모델 학습 실행
trained_model = train(model, train_loader, valid_loader)

# 학습된 모델 저장
trained_model.save_pretrained("./kobart_summarization_model_v3")
tokenizer.save_pretrained("./kobart_summarization_model_v3")
