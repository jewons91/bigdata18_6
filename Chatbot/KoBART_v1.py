import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration, AdamW
from tqdm import tqdm

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['documents']

train_data = load_data('문서요약 텍스트/train/train_original.json')
valid_data = load_data('문서요약 텍스트/valid/valid_original.json')

# 데이터 구조 확인
print("Train data sample:", train_data[0])

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
        
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')

train_dataset = SummarizationDataset(train_data, tokenizer)
valid_dataset = SummarizationDataset(valid_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

def train(model, train_loader, valid_loader, epochs=3, lr=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Average train loss: {avg_train_loss:.4f}')

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(valid_loader)
        print(f'Epoch {epoch+1}/{epochs}, Average validation loss: {avg_val_loss:.4f}')

    return model

trained_model = train(model, train_loader, valid_loader)

trained_model.save_pretrained('./kobart_summarization_model')
tokenizer.save_pretrained('./kobart_summarization_model')
