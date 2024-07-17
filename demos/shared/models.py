import torch
import torch.nn as nn
class myVGG(nn.Module):
    def __init__(self):
        super(myVGG, self).__init__()  # Call the parent class's initializer
        block1 = self.create_block(3, 64)  # Create the first block
        block2 = self.create_block(64, 64)  # Create the second block
        block3 = self.create_block(64, 128)
        block4 = self.create_block(128, 128)
        self.net = nn.Sequential(
            block1,
            block2,
            block3,
            block4,
            nn.Sequential(
            nn.Flatten(),
            nn.Linear(73728, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
            )
        )


    def forward(self, x):
        return self.net(x)

    def create_block(self, in_channels, filters, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size, padding),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        
        



class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()  # Call the parent class's initializer
        block1 = self.create_block(3072, 100)  # Create the first block
        block2 = self.create_block(100, 100)  # Create the second block
        block3 = self.create_block(100, 100)
        self.net = nn.Sequential(
            block1,
            block2,
            block3,
            nn.Sequential(
            nn.Linear(100,10)
            )
        )

    def forward(self, x):
        return self.net(x)
    
    
    def create_block(self, input_shape, output_shape):
        return nn.Sequential(
            nn.Linear(input_shape, output_shape),
            nn.ReLU()
        )
    
    
    
class smallVGG(nn.Module):
    def __init__(self):
        super(smallVGG, self).__init__()  # Call the parent class's initializer
        block1 = self.create_block(3, 64)  # Create the first block
        block2 = self.create_block(64, 128)  # Create the second block
        self.feature_extractor = nn.Sequential(
            block1,
            block2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(100352, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)
        
    def create_block(self, in_channels, filters, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size, padding),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        


class TwoStepVgg(nn.Module):
    def __init__(self):
        super(TwoStepVgg, self).__init__()  # Call the parent class's initializer
        block1 = self.create_block(3, 64)  # Create the first block
        block2 = self.create_block(64, 64)  # Create the second block
        block3 = self.create_block(64, 128)
        block4 = self.create_block(128, 128)
        self.net = nn.Sequential(
                nn.Sequential(
                block1,
                block2,
                block3
            ),
            nn.Sequential(            
            block4,
            nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 24 * 24, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
            )
        )
        )
    def forward(self, x):
        return self.net(x)

    def create_block(self, in_channels, filters, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size, padding),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        
        

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

    def create_block(self, in_channels, filters, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size, padding),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        
        
def test_model_accuracy(model, testloader, device='cpu'):
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy}%')
    return accuracy


def train_model(model, 
                n_epochs,
                trainloader,
                optimizer=None,
                criterion=None,
                device= 'cpu'):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(trainloader.dataset)
        print(f'Training Loss: {epoch_loss:.4f}')
        
    
    
    
class ThreeStepVgg(nn.Module):
    def __init__(self):
        super(ThreeStepVgg, self).__init__()  # Call the parent class's initializer
        block1 = self.create_block(3, 64)  # Create the first block
        block2 = self.create_block(64, 64)  # Create the second block
        block3 = self.create_block(64, 128)
        block4 = self.create_block(128, 128)
        block5 = self.create_block(128, 256)
        block6 = self.create_block(256, 256)

        self.device_side_net = nn.Sequential(
            block1,
            block2,
        )
        
        self.server_side_net= nn.Sequential(            
            block3,
            block4
        )
        
        self.cloud_net = nn.Sequential(
            block5,
            block6,
            nn.Flatten(),
            nn.Linear(16384,512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )
        self.net = nn.Sequential(
            self.device_side_net,
            self.server_side_net,
            self.cloud_net
        )
    def forward(self, x):
        return self.net(x)
    
    
    def create_block(self, in_channels, filters, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size, padding),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size, padding),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        