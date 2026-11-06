# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="942" height="970" alt="image" src="https://github.com/user-attachments/assets/6a2e81b2-844f-4b5c-a861-cf5f02360b77" />

## DESIGN STEPS
### STEP 1: 

Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### STEP 2: 

Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

### STEP 3: 

Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

### STEP 4: 

Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### STEP 5: 

Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

### STEP 6: 

Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.


## PROGRAM

### Name:JOSHITHA SHREE BS

### Register Number:212224230107

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)


    def forward(self, x):
        #Include your code here
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x) 
        return x
# Initialize the Model, Loss Function, and Optimizer
input_size = X_train.shape[1]
model = PeopleClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
def train_model(model, train_loader, criterion, optimizer, epochs):
    #Include your code here
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
 
        if (epoch + 1) % 10 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
```

### Dataset Information

<img width="743" height="250" alt="image" src="https://github.com/user-attachments/assets/04c83480-5798-4547-a347-92e47d1f308a" />


### OUTPUT

<img width="289" height="34" alt="image" src="https://github.com/user-attachments/assets/8c41bfcd-0ad4-4b1a-895f-c8ad5755c4c3" />

## Confusion Matrix

<img width="540" height="479" alt="image" src="https://github.com/user-attachments/assets/4dd7db34-2e5c-4861-9b6d-6415ab6bf7aa" />

## Classification Report

<img width="532" height="336" alt="image" src="https://github.com/user-attachments/assets/f84087cc-1916-4354-a5ef-7a10173cb7aa" />


### New Sample Data Prediction

<img width="318" height="93" alt="image" src="https://github.com/user-attachments/assets/7969bbb6-24a0-4364-b501-980d9ad36cf0" />


## RESULT
Include your result here
