import torch
import torch.nn.functional as F

def knowledge_distillation(teacher_model, 
                        student_model, 
                        n_epochs,
                        trainloader,
                        teacher_percentage = 0.5 ,
                        temperature= 2,
                        optimizer=None,
                        criterion=None,
                        device= 'cpu'):     
    if optimizer is None:
        optimizer = torch.optim.Adam(student_model.parameters(),lr=0.001)
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    student_model.train()  
    for epoch in range(n_epochs):
        running_loss = 0.0

        for inputs, targets in trainloader:
            inputs = inputs.to(device)
            targets=  targets.to(device)
            with torch.no_grad():
                teacher_targets = teacher_model(inputs).to(device)
                teacher_targets = F.softmax(teacher_targets / temperature, dim=1)
                teacher_targets=  teacher_targets.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs = student_model(inputs)  # Forward pass
            teacher_loss = criterion(outputs, teacher_targets)  # Compute loss
            absolute_loss = criterion(outputs, targets) 
            total_loss = teacher_loss * teacher_percentage  + (1-teacher_percentage) *absolute_loss
            total_loss.backward()  
            optimizer.step() 
            running_loss += total_loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(trainloader.dataset)
        print(f'Training Loss: {epoch_loss:.4f}')
        