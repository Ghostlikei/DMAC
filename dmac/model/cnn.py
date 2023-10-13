import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from .model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(Model):
    def __init__(self, params):
        self.params = params
        self.clf = CNNModel(params)
        self.clf.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.clf.parameters(), lr=params.learning_rate, weight_decay=1e-4)
        self.batch_size = params.batch_size
        self.num_epochs = params.num_epochs

    def train(self, train_data, train_label, logging = False):
        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_label = torch.tensor(train_label, dtype=torch.int64)

        self.clf.train()

        for epoch in range(self.num_epochs):
            for i in range(0, len(train_data), self.batch_size):
                batch_data = train_data[i:i + self.batch_size].to(device)
                batch_label = train_label[i:i + self.batch_size].to(device)

                outputs = self.clf(batch_data)

                loss = self.criterion(outputs, batch_label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if logging:
                if (epoch + 1) % 1 == 0:
                    print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.4f}")

    def predict(self, predict_data, label=None):
        predict_data = torch.tensor(predict_data, dtype=torch.float32).to(device)
        self.clf.eval()

        outputs = self.clf(predict_data)

        _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

class CNNModel(nn.Module):
    def __init__(self, params):
        super(CNNModel, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, params.num_filters, (fs, params.embedding_dim))
            for fs in params.filter_sizes
        ])
        self.dropout = nn.Dropout(p=params.dropout_prob)
        self.fc = nn.Linear(len(params.filter_sizes) * params.num_filters, params.output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
