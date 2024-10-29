import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

# Step 1: Read and preprocess the data

def preprocess(df):
    df["Name"] = df["Name"].apply(lambda x: " ".join([v.strip(",()[].\"'") for v in x.split(" ")]))
    df["Ticket_number"] = df["Ticket"].apply(lambda x: x.split(" ")[-1])
    df["Ticket_item"] = df["Ticket"].apply(lambda x: "_".join(x.split(" ")[:-1]) if len(x.split(" ")) > 1 else "NONE")
    return df

train_df = pd.read_csv("train.csv")
serving_df = pd.read_csv("test.csv")

preprocessed_train_df = preprocess(train_df)
preprocessed_serving_df = preprocess(serving_df)

input_features = ["Pclass", "Sex"]  # Adjust based on your dataset

# Encode categorical variables
label_encoder = LabelEncoder()
preprocessed_train_df["Sex"] = label_encoder.fit_transform(preprocessed_train_df["Sex"])
preprocessed_serving_df["Sex"] = label_encoder.transform(preprocessed_serving_df["Sex"])

# Fill missing values
preprocessed_train_df.fillna(0, inplace=True)
preprocessed_serving_df.fillna(0, inplace=True)

x_train = torch.tensor(np.array(preprocessed_train_df[input_features])).float()
y_train = torch.tensor(np.array(preprocessed_train_df["Survived"])).float()
x_test = torch.tensor(np.array(preprocessed_serving_df[input_features])).float()

# Step 2: Define and train the LSTM model

class TRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.05):
        super(TRNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.Dropout(x)
        out, _ = self.lstm(out)
        out = self.fc(out)
        return out

# Initialize model
model = TRNN(input_dim=len(input_features), hidden_dim=256, output_dim=1)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs.squeeze(), y_train)  # Squeeze to remove unnecessary dimension
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Step 3: Make predictions on serving data

model.eval()
with torch.no_grad():
    predicted_outputs = model(x_test)
    predicted_labels = (torch.sigmoid(predicted_outputs) > 0.5).int()  # Apply threshold for binary classification

predicted_labels_flat = predicted_labels.squeeze().tolist()

# Create the output DataFrame using a dictionary of columns
output_data = {
    "PassengerId": serving_df["PassengerId"],  # Use PassengerId column from serving_df
    "Survived": predicted_labels_flat  # Use predicted_labels as the Survived column
}

output = pd.DataFrame(output_data)
output.to_csv("submission.csv", index=False)

print(output.head(10))