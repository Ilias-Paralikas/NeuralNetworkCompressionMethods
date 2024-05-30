import torch
import torch.nn.functional as F

def knowledge_distillation(teacher_model, 
                        student_model, 
                        n_epochs,
                        trainloader,
                        temperature= 5,
                        optimizer=None,
                        criterion=None,
                        device= 'cpu'):
    if optimizer is None:
        optimizer = torch.optim.Adam(student_model.parameters(),lr=0.001)
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()
    student_model.train()  # Set the model to training mode
    i = 0
    running_loss = 0.0
    for epoch in range(n_epochs):
        for inputs, _ in trainloader:
            inputs = inputs.to(device)
            with torch.no_grad():
                targets = teacher_model(inputs).to(device)
                targets = F.softmax(targets / temperature, dim=1)
                targets=  targets.to(device)
            optimizer.zero_grad()  # Zero the gradients
            outputs = student_model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item() * inputs.size(0)
            if i %100 ==0:
                epoch_loss = running_loss / len(trainloader.dataset)
                print(f'Training Loss: {epoch_loss:.4f}')
            i = i+1