import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .model import Model


class MLP(Model):
    def __init__(self, params):
        self.params = params
        self.clf = _MLP(params)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.clf.parameters(), lr=params.learning_rate)
        self.batch_size = params.batch_size
        self.num_epochs = params.num_epochs

    def train(self, train_data, train_label, logging = False):
        train_data = train_data.toarray()

        assert self.params.input_size == train_data.shape[1]

        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_label = torch.tensor(train_label, dtype=torch.int64)

        self.clf.train()

        for epoch in range(self.num_epochs):
            for i in range(0, len(train_data), self.batch_size):
                batch_data = train_data[i:i + self.batch_size]
                batch_label = train_label[i:i + self.batch_size]

                outputs = self.clf(batch_data)
                
                # Convert batch_label to one-hot encoding
                loss = self.criterion(outputs, batch_label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if logging:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.4f}")

    def predict(self, predict_data):
        predict_data = predict_data.toarray()
        predict_tensor = torch.tensor(predict_data, dtype=torch.float32)
        self.clf.eval()

        outputs = self.clf(predict_tensor)
        
        _, predicted = torch.max(outputs, 1)
        
        return predicted.numpy()

class _MLP(nn.Module):
    def __init__(self, params):
        super(_MLP, self).__init__()
        self.fc1 = nn.Linear(params.input_size, params.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=params.dropout_prob)
        self.fc2 = nn.Linear(params.hidden_size, params.output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
