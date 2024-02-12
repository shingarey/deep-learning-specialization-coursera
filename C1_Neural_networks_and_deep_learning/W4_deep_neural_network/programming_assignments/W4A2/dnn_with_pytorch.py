import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import tqdm.notebook as tqdm
from tqdm import tqdm

class MNIST_Logistic_Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(64*64*3, 100)
        torch.nn.init.xavier_uniform(self.layer1.weight)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)  # Add dropout
        self.layer2 = nn.Linear(100, 20)
        torch.nn.init.xavier_uniform(self.layer2.weight)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(20, 5)
        torch.nn.init.xavier_uniform(self.layer3.weight)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(5, 1)
        torch.nn.init.xavier_uniform(self.output.weight)

        self.sigmoid = nn.Sigmoid()
        self.threshold = torch.tensor([0.5])

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

class CustomDataset(Dataset):
    def __init__(self, file_path, dataset_type):
        self.file_path = file_path
        self.data = None
        self.labels = None
        self.classes = None
        self.load_data(dataset_type)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

    def load_data(self, dataset_type):
        dataset = h5py.File(self.file_path, "r")
        if dataset_type == 'train':
            self.data = torch.tensor(dataset["train_set_x"][:])
            self.labels = torch.tensor(dataset["train_set_y"][:])
        elif dataset_type == 'test':
            self.data = torch.tensor(dataset["test_set_x"][:])
            self.labels = torch.tensor(dataset["test_set_y"][:])
        self.classes = dataset["list_classes"][:]

    def normalize_data(self):
        self.data = self.data.float() / 255.0

train_dataset = CustomDataset("datasets/train_catvnoncat.h5", 'train')
test_dataset = CustomDataset("datasets/test_catvnoncat.h5", 'test')
train_dataset.normalize_data()
test_dataset.normalize_data()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=209, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=False)


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Training
# Instantiate model
model = MNIST_Logistic_Regression().to(device)

# Loss and Optimizer
#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.00075)

model.train()
# Iterate through train set minibatchs
for _ in tqdm(range(10000)):
    for data, labels in train_loader:
           # Zero out the gradients
           optimizer.zero_grad()
           # Move data to device
           data = data.to(device)
           labels = labels.to(device)
           # Forward pass
           x = data.view(-1, 64 * 64 * 3)
           y = model(x.float())
           loss = criterion(y, labels.float().view(-1, 1))
           # Backward pass
           loss.backward()
           optimizer.step()
    if _ % 100 == 0:     
        model.eval()
        print(loss.item())
        yp = (y > 0.5).float()
        correct = torch.sum((yp == labels.float().unsqueeze(1)).float())
        print("Test accuracy: {}".format(correct/len(labels)))
        model.train()

## Testing
correct = 0
total = len(test_dataset)

model.eval()

with torch.no_grad():
    # Iterate through test set minibatchs
    for data, labels in test_loader:
        # Move data to device
        data = data.to(device)
        labels = labels.to(device)

        # Forward pass
        x = data.view(-1, 64 * 64 *3)
        y = model(x.float())
        y = (y > 0.5).float()

        #predictions = torch.argmax(y, dim=1)
        correct = torch.sum((y == labels.float().unsqueeze(1)).float())

print("Test accuracy: {}".format(correct / total))