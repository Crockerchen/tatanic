import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch


class MyDataset(Dataset):

    def __init__(self, filepath):
        features = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
        data = pd.read_csv(filepath)
        self.len = data.shape[0]

        self.x_data = torch.from_numpy(np.array(pd.get_dummies(data[features])))
        self.y_data = torch.from_numpy(np.array(data["Survived"]))

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = MyDataset('./train.csv')
train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(6, 3)
        self.linear2 = torch.nn.Linear(3, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.sigmoid(self.linear1(x))
            x = self.sigmoid(self.linear2(x))
            y = []
            for i in x:
                if i > 0.5:
                    y.append(1)
                else:
                    y.append(0)
            return y


model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()

            y_pred = model(inputs)
            y_pred = y_pred.squeeze(-1)
            loss = criterion(y_pred, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
test_data = pd.read_csv("./test.csv")
features = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
test = torch.from_numpy(np.array(pd.get_dummies(test_data[features])))

y = model.predict(test.float())

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y})
output.to_csv('Submission.csv', index=False)
