import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers' , 'quotes'))

X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.25, random_state=42)

print(y_train)

# TF-IDF

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000) 
X_train_tfidf = vectorizer.fit_transform(X_train).toarray() 
X_test_tfidf = vectorizer.transform(X_test).toarray()
# torch tensor

X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32) 
X_test_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32) 
y_train_tensor = torch.tensor(y_train, dtype=torch.int64) 
y_test_tensor = torch.tensor(y_test, dtype=torch.int64)
# MLP

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

input_dim = X_train_tfidf.shape[1] 
hidden_dim = 100
output_dim = len(set(y_train))
learning_rate = 0.01
epochs = 2


print("Input dim: ", input_dim)
print("Output dim: ", output_dim)

model = MLP(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):

    outputs = model(X_train_tensor)
    optimizer.zero_grad()

    print("Outputs: ", outputs)
    print("y_train_tensor: ", y_train_tensor)

    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

with torch.no_grad():
    predictions = model(X_test_tensor)
    _, predicted = torch.max(predictions, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    predictions = model(X_test_tensor)
    print(f"Test Accuracy: {accuracy*100:.2f}%")