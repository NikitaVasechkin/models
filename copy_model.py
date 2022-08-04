
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(51136, 50)
        self.fc2 = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


model = CNNet()

cost = torch.nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



def train(dataloader, model, loss, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, Y) in enumerate(dataloader):

        optimizer.zero_grad()
        pred = model(X)
        loss = cost(pred, Y)
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


# Create the validation/test function

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            pred = model(X)

            test_loss += cost(pred, Y).item()
            correct += (pred.argmax(1) == Y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size

    print(
        f'\nTest Error:\nacc: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f}\n')


if __name__ == '__main__':

    data_path = './boxes'

    num_set = datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([transforms.Resize((201, 81)),
                                      transforms.ToTensor()
                                      ])
    )

    class_map = num_set.class_to_idx

    print("\nClass category and index of the images: {}\n".format(class_map))

    train_size = int(0.8 * len(num_set))
    test_size = len(num_set) - train_size
    num_train_set, num_test_set = torch.utils.data.random_split(
        num_set, [train_size, test_size])

    print("Training size:", len(num_train_set))
    print("Testing size:", len(num_test_set))

    train_dataloader = torch.utils.data.DataLoader(
        num_train_set,
        batch_size=8,
        num_workers=2,
        shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        num_test_set,
        batch_size=8,
        num_workers=2,
        shuffle=True
    )

    epochs = 15

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train(train_dataloader, model, cost, optimizer)
        test(test_dataloader, model)
    print('Done!')

    torch.save(model.state_dict(), "./model_num")
