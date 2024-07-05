import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Assuming resnet20 and se_resnet20 are defined elsewhere or imported properly
from senet.baseline import resnet20
from senet.se_resnet import se_resnet20

class AccuracyCallback:
    def __init__(self):
        self.accuracies = []

    def __call__(self, model, data_loader, device):
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(data_loader.dataset)
        self.accuracies.append(accuracy)
        #doan nay sua code de in ra mo hinh dang dung
        print(f'Test set {args.baseline}: Accuracy: {accuracy:.2f}%')

def main():
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_loader = DataLoader(
        datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    test_loader = DataLoader(
        datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Model selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #neu baseline duoc chi dinh khi chay python cifaphuong.py --baseline thi la true neu true thi resnes20 nguoc lai la se_resnet20
    if args.baseline:
        model = resnet20().to(device)
    else:
        model = se_resnet20(num_classes=10, reduction=args.reduction).to(device)

    # Optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=80, gamma=0.1)

    # Callbacks and training setup
    accuracy_callback = AccuracyCallback()
    for epoch in range(1, args.epochs + 1):
        model.train()
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}") as pbar:
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        # Scheduler step
        scheduler.step()

        # Testing
        accuracy_callback(model, test_loader, device)

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--reduction", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--baseline", action="store_true")
    args = p.parse_args()
    main()
