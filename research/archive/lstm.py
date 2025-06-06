import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd


torch.manual_seed(42)
np.random.seed(42)


def load_sequences_from_csv(path, seq_length=100, look_ahead=50):
    data = pd.read_csv(path)


class VariableLengthLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(VariableLengthLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        packed_output, (hidden, _) = self.lstm(packed_input, (h0, c0))

        out = hidden[-1, :, :]
        out = self.fc(out)
        out = self.leaky_relu(out)
        out = self.softmax(out)
        return out


# Step 2: Create a custom dataset class for variable-length sequences
class VariableLengthSequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences  # List of variable-length sequences
        self.targets = targets  # List of targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# Step 3: Custom collate function for DataLoader
def collate_fn(batch):
    # Separate sequences and targets
    sequences = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Get sequence lengths
    lengths = torch.tensor([len(seq) for seq in sequences])

    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True)

    # Convert targets to tensor
    targets = torch.stack(targets)

    return padded_sequences, lengths, targets


# Step 4: Generate dummy multi-feature variable-length data
def generate_variable_length_data(
    num_samples=100, max_length=50, min_length=10, num_features=3
):
    sequences = []
    targets = []

    for _ in range(num_samples):
        # Random sequence length between min_length and max_length
        seq_length = np.random.randint(min_length, max_length + 1)

        # Generate multiple feature sequence
        # Each feature follows a different pattern (sine, cosine, linear)
        x = np.linspace(0, 5 * np.pi, seq_length)
        feature1 = np.sin(x)
        feature2 = np.cos(x)
        feature3 = np.linspace(0, 1, seq_length)

        # Combine features
        seq_array = np.column_stack((feature1, feature2, feature3))[:, :num_features]

        # Create a target (for demo, we'll use the sum of the last values of each feature)
        target = np.sum(seq_array[-1, :])

        # Convert to tensors
        sequence_tensor = torch.FloatTensor(seq_array)
        target_tensor = torch.FloatTensor([target])

        sequences.append(sequence_tensor)
        targets.append(target_tensor)

    return sequences, targets


# Generate data
train_sequences, train_targets = generate_variable_length_data(
    num_samples=500, num_features=3
)
val_sequences, val_targets = generate_variable_length_data(
    num_samples=100, num_features=3
)

for i in range(len(train_sequences)):
    print(train_sequences[i])
    print(train_targets[i])
    print(train_sequences[i].shape)
    print(train_targets[i].shape)
    print()

# Create datasets
train_dataset = VariableLengthSequenceDataset(train_sequences, train_targets)
val_dataset = VariableLengthSequenceDataset(val_sequences, val_targets)

# Create dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
)

# Step 5: Model parameters
input_size = 61  # Number of features
hidden_size = 128
num_layers = 2
output_size = 1
learning_rate = 0.001
num_epochs = 50

# Initialize model, loss function, and optimizer
model = VariableLengthLSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step 6: Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for sequences, lengths, targets in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(sequences, lengths)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for sequences, lengths, targets in val_loader:
            outputs = model(sequences, lengths)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    # Print statistics
    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {train_loss/len(train_loader):.4f}, "
            f"Val Loss: {val_loss/len(val_loader):.4f}"
        )


# Step 7: Test with a new variable-length sequence with multiple features
def predict_with_variable_length_sequence(model, sequence):
    model.eval()
    with torch.no_grad():
        # Prepare data
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(
            0
        )  # Add batch dimension
        length = torch.tensor([len(sequence)])

        # Get prediction
        output = model(sequence_tensor, length)

        return output.item()


# Create a test sequence
test_length = 35
x = np.linspace(0, 5 * np.pi, test_length)
test_sequence = np.column_stack((np.sin(x), np.cos(x), np.linspace(0, 1, test_length)))

# Make prediction
prediction = predict_with_variable_length_sequence(model, test_sequence)
print(f"Prediction for test sequence: {prediction:.4f}")
print(f"Expected value (sum of last features): {np.sum(test_sequence[-1, :]):.4f}")

# Save the model
# torch.save(model.state_dict(), 'variable_length_lstm_model.pth')
# print("Model saved successfully!")
