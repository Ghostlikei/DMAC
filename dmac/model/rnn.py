import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import torch.nn.functional as F
from .model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(Model):
    def __init__(self, params):
        self.params = params
        self.clf = _RNN(params)
        self.clf.to(device)
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.clf.parameters(), lr=params.learning_rate)
        self.batch_size = params.batch_size
        self.num_epochs = params.num_epochs

    def train(self, train_data, train_label, logging=False):
        train_data = [torch.tensor(seq, dtype=torch.float32).to(device) for seq in train_data]
        train_label = torch.tensor(train_label, dtype=torch.int64).to(device)

        self.clf.train()

        for epoch in range(self.num_epochs):
            cnt = 0
            for i in range(0, len(train_data), self.batch_size):
                cnt += 1
                batch_data = train_data[i:i + self.batch_size]
                batch_label = train_label[i:i + self.batch_size]

                # Pad sequences to the same length within the batch
                batch_data = nn.utils.rnn.pad_sequence(batch_data, batch_first=True, padding_value=0)
                # print(batch_data[0])

                outputs = self.clf(batch_data)

                loss = self.criterion(outputs, batch_label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if logging:
                    if cnt % 10 == 0:
                        print(f"Epoch {epoch + 1}/{self.num_epochs}, Round: {cnt}/{len(train_data)//self.batch_size}, Loss: {loss.item():.4f}")
                        print("Predicted label: ", torch.argmax(outputs[0]),"Label: ", batch_label[0])

    def predict(self, predict_data, label=None):
        predict_data = [torch.tensor(seq, dtype=torch.float32).to(device) for seq in predict_data]
        batch_data = nn.utils.rnn.pad_sequence(predict_data, batch_first=True, padding_value=0)
        self.clf.eval()

        outputs = self.clf(batch_data)  # Add a batch dimension

        predicted = torch.argmax(outputs, dim=1)
        return predicted.cpu().numpy()


class _RNN(nn.Module):
    def __init__(self, params):
        super(_RNN, self).__init__()

        if params.type == "Vanilla":
            self.rnn = nn.RNN(params.input_size, 
                              params.hidden_size, 
                              num_layers=params.num_layers, 
                              batch_first=True, 
                              dropout=params.dropout_prob,
                              bidirectional=params.bidirectional)
        elif params.type == "LSTM":
            self.rnn = nn.LSTM(params.input_size, 
                               params.hidden_size, 
                               num_layers=params.num_layers, 
                               batch_first=True, 
                               dropout=params.dropout_prob,
                               bidirectional=params.bidirectional)
        elif params.type == "GRU":
            self.rnn = nn.GRU(params.input_size, 
                              params.hidden_size, 
                              num_layers=params.num_layers, 
                              batch_first=True, 
                              dropout=params.dropout_prob,
                              bidirectional=params.bidirectional)

        bidir_op = 1
        if params.bidirectional == True:
            bidir_op = 2
        self.fc = nn.Linear(params.hidden_size * bidir_op, params.output_size)

        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
        # self.batch_norm = nn.BatchNorm1d(params.hidden_size)

        self.attention = nn.Linear(params.hidden_size * bidir_op, 1)
        self.fc = nn.Linear(params.hidden_size * bidir_op, params.output_size)
        # self.batch_norm = nn.BatchNorm1d(params.hidden_size * bidir_op)
        

    def forward(self, x):
        out, _ = self.rnn(x)

        # Calculate attention weights
        attention_weights = torch.softmax(self.attention(out), dim=1)
        
        # Apply attention to RNN outputs
        attention_output = torch.sum(attention_weights * out, dim=1)
        
        output = self.fc(attention_output)
        # return output
        # output = self.fc(out[:, -1])  # Use the last output of the sequence
        return output
