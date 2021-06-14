import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchsummary import summary


class Data(Dataset):
    # Constructor
    def __init__(self, X, y):
        self.x = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.len = len(X)

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len


class TestData(Dataset):
    # Constructor
    def __init__(self, X):
        self.x = torch.from_numpy(X).float()
        self.len = len(X)

    # Getter
    def __getitem__(self, index):
        return self.x[index]

    # Get Length
    def __len__(self):
        return self.len


class NN(nn.Module):
    def __init__(self, in_features, out_features):
        super(NN, self).__init__()

        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):

        x = self.fc(self.dropout(x))

        return x.squeeze()


def training(
    model, num_epochs, train_loader, val_loader, scheduler, optimiser, loss_fn, device
):

    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        # train================================
        model.train()
        training_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            y_pred = model(x)
            loss = torch.sqrt(loss_fn(y_pred, y))
            training_loss += loss.data.item()
            loss.backward()
            optimiser.step()
        train_loss = training_loss / len(train_loader)
        train_loss_list.append(train_loss)

        # valid================================
        model.eval()
        training_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = torch.sqrt(loss_fn(y_pred, y))
                training_loss += loss.data.item()
        val_loss = training_loss / len(val_loader)
        val_loss_list.append(val_loss)
        scheduler.step(val_loss)

        print(
            "Epoch: {:2}".format(epoch),
            "Train Loss: {:6.4f}".format(train_loss),
            "Val Loss: {:6.4f}".format(val_loss),
        )

    return train_loss_list, val_loss_list


if __name__ == "__main__":

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("GPU is avalible")
        print("Working on ", torch.cuda.get_device_name())
    else:
        device = torch.device("cpu")
        print("GPU is not avalible")

    model = NN(in_features=5, out_features=1)
    model.to(device)
    print(model)
