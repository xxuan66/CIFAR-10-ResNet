# @Time    : 8/1/2025 上午 9:19
# @Author  : Xuan
# @File    : ResNet_CIFAR10.py
# @Software: PyCharm

"""
项目背景：
CIFAR-10是一个包含60000张32x32彩色图像的数据集，共分为10个类别，每个类别6000张图像。
在本项目中，我们设计并实现了一个基于ResNet的神经网络模型，以提升模型的特征提取能力和识别效果。

核心技术：
1. 残差连接：通过短路连接将输入直接加到输出上，缓解了梯度消失问题，提升了模型的训练效果。
2. 批标准化：通过对每一层的输出进行标准化处理，加速模型收敛，提升模型的泛化能力。
3. 池化层：通过最大池化操作降低特征图的维度，减少模型参数量，提升模型的计算效率。


项目实现：
下面的代码展示了整个流程，从数据加载与预处理到模型定义、训练和测试，最后对ResNet与标准卷积的参数量进行对比分析。
数据集地址：https://www.cs.toronto.edu/~kriz/cifar.html
本地地址：./dataset/CIFAR10/
"""

# 1、导入库
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 2、加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./dataset/CIFAR10', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./dataset/CIFAR10', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 3、检查数据集大小
print(f'Train dataset size: {len(train_dataset)}')  # Train dataset size: 50000
print(f'Test dataset size: {len(test_dataset)}')  # Test dataset size: 10000

# 4、数据可视化--展示每个类别1个样本
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class_samples = {class_name: None for class_name in classes}

# 遍历数据集，直到每个类别都有一个样本
dataiter = iter(train_loader)
while None in class_samples.values():
    images, labels = next(dataiter)
    for i in range(len(labels)):
        class_name = classes[labels[i]]
        if class_samples[class_name] is None:
            class_samples[class_name] = images[i]

# 可视化每个类别的样本
plt.figure(figsize=(8, 8))
for i, class_name in enumerate(classes):
    plt.subplot(2, 5, i + 1)
    plt.imshow(np.transpose(class_samples[class_name] / 2 + 0.5, (1, 2, 0)))
    plt.title(class_name)
    plt.axis('off')
plt.show()

# 5、定义ResNet模型
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 6、定义ResNet18模型
def ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2, 2])

# 7、定义训练函数
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

# 8、定义测试函数
def test(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {correct / total * 100}%')

# 9、训练ResNet18模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train(model, train_loader, criterion, optimizer, num_epochs=10)
test(model, test_loader)

