from torch.optim import lr_scheduler
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models


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

    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    IN_FEATURES = model.fc.in_features
    model.fc = nn.Linear(IN_FEATURES, 2)

    data_path = './data/f_m'

    IMAGE_SIZE = 224
    NUM_CLASSES = 2
    BATCH_SIZE = 32

    cost = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001)
    lr_sch = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    num_set = datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        num_test_set,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=True
    )

    for t in range(15):
        print(f'Epoch {t+1}\n-------------------------------')
        train(train_dataloader, model, cost, optimizer_ft)
        test(test_dataloader, model)
    print('Done!')

    torch.save(model.state_dict(), "f_m_model.bin")
