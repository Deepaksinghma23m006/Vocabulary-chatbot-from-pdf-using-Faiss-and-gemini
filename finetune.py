import torch
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Define a custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text': text
        }

# Load your dataset
data = pd.read_csv('your_dataset.csv')  # Make sure your CSV has a column with texts
texts = data['text_column'].tolist()  # Replace 'text_column' with the name of your text column

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')

# Create DataLoader
dataset = TextDataset(texts, tokenizer, max_len=128)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Fine-tune the model
optimizer = AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(3):  # Adjust the number of epochs as needed
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Save the fine-tuned model
model.save_pretrained('./fine-tuned-bert')
tokenizer.save_pretrained('./fine-tuned-bert')

