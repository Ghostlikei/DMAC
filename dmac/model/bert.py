import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

from ..model.model import Model
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BERT(Model):
    def __init__(self, params):
        self.params = params
        self.tokenizer = BertTokenizer.from_pretrained(params.bert_model_name)
        self.clf = BertForSequenceClassification.from_pretrained(params.bert_model_name, num_labels=params.num_labels)
        self.clf.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.clf.parameters(), lr=params.learning_rate)
        self.batch_size = params.batch_size
        self.num_epochs = params.num_epochs

        training_args = TrainingArguments(
            output_dir='./results',
            per_device_train_batch_size=params.batch_size,
            num_train_epochs=params.num_epochs,
            logging_dir='./logs',
            fp16=True,  # Enable mixed-precision training
            # Add other arguments as needed
        )

        

    def train(self, train_data, train_label):
        self.clf.train()

        for epoch in range(self.num_epochs):
            cnt = 0
            for i in range(0, len(train_data), self.batch_size):
                cnt += 1
                batch_data = train_data[i:i + self.batch_size]
                batch_label = train_label[i:i + self.batch_size]

                input_ids = []
                attention_mask = []

                for text in batch_data:
                    encoding = self.tokenizer(text, padding='max_length', max_length=self.params.max_seq_length, truncation=True, return_tensors='pt')
                    input_ids.append(encoding['input_ids'])
                    attention_mask.append(encoding['attention_mask'])

                input_ids = torch.cat(input_ids, dim=0)
                attention_mask = torch.cat(attention_mask, dim=0)

                # Move the input data and labels to the GPU
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                batch_label = torch.tensor(batch_label).to(device)  # Ensure labels are on the GPU

                outputs = self.clf(input_ids, attention_mask=attention_mask)

                # Ensure outputs contain logits, not probabilities
                logits = outputs.logits

                # Calculate the loss
                loss = self.criterion(logits, batch_label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if cnt % 10 == 0:
                    print(f"Epoch {epoch + 1}/{self.num_epochs}, Round: {cnt}/{len(train_data)//self.batch_size}, Loss: {loss.item():.4f}")
                    print("Predicted label: ", torch.argmax(logits[0]),"Label: ", batch_label[0])

    def predict(self, predict_data, label=None, batch_size=16):
        self.clf.eval()
        predicted_labels = []

        for i in range(0, len(predict_data), batch_size):
            batch_data = predict_data[i:i + batch_size]

            input_ids = []
            attention_mask = []

            for text in batch_data:
                encoding = self.tokenizer(text, padding='max_length', max_length=self.params.max_seq_length, truncation=True, return_tensors='pt')
                input_ids.append(encoding['input_ids'])
                attention_mask.append(encoding['attention_mask'])

            input_ids = torch.cat(input_ids, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)

            # Move the input data and labels to the GPU
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            
            outputs = self.clf(input_ids, attention_mask=attention_mask)

            # Ensure outputs contain logits, not probabilities
            predicted = torch.argmax(outputs.logits, dim=1).flatten()

            predicted_labels.extend(predicted.cpu().numpy())

            # Clear GPU cache after each batch prediction to release memory
            torch.cuda.empty_cache()

        return predicted_labels
