import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
    
    SIZE = 96  
    BATCH = 128
    EPOCHS = 15

    train_tf = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])
    test_tf = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = datasets.FashionMNIST("./data", True,  download=True, transform=train_tf)
    test_ds  = datasets.FashionMNIST("./data", False, download=True, transform=test_tf)
    train_ld = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=0)
    test_ld  = DataLoader(test_ds,  BATCH, shuffle=False, num_workers=0)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 10),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    tr_l, te_l, tr_a, te_a = [], [], [], []
    best_acc = 0.0

    for ep in range(1, EPOCHS+1):
        model.train(); rl=c=t=0
        for x, y in tqdm(train_ld, desc=f"E{ep} train"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            out = model(x)
            loss = criterion(out, y)
            
            loss.backward()
            optimizer.step()
            
            rl += loss.item()*x.size(0)
            c  += (out.argmax(1)==y).sum().item()
            t  += y.size(0)
        tr_l.append(rl/t); tr_a.append(c/t)

        model.eval(); rl=c=t=0
        with torch.no_grad():
            for x, y in tqdm(test_ld, desc=f"E{ep} test"):
                x, y = x.to(device), y.to(device)
                out = model(x); loss = criterion(out, y)
                rl += loss.item()*x.size(0)
                c  += (out.argmax(1)==y).sum().item()
                t  += y.size(0)
        te_l.append(rl/t); te_a.append(c/t)
        scheduler.step()

        if te_a[-1] > best_acc:
            best_acc = te_a[-1]
            torch.save(model.state_dict(), "best_resnet18_fashion.pth")

        print(f"E{ep}: train_acc={tr_a[-1]:.4f} test_acc={te_a[-1]:.4f} "
              f"train_loss={tr_l[-1]:.4f} test_loss={te_l[-1]:.4f}")

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(tr_l, label="train"); plt.plot(te_l, label="test")
    plt.title("Loss"); plt.legend(); plt.grid()
    plt.subplot(1,2,2)
    plt.plot(tr_a, label="train"); plt.plot(te_a, label="test")
    plt.title("Accuracy"); plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()

    print(f"Best test accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main()