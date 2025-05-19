import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.load import DataLoader as Loader
from tqdm import tqdm


class OnlineModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.05, max_memory_size=100):
        super(OnlineModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(dropout),
            # nn.LayerNorm(hidden_dim),
        )

        self.memory = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.key_transform = nn.Linear(hidden_dim, hidden_dim)
        self.value_transform = nn.Linear(hidden_dim, hidden_dim)

        self.output_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(dropout),
            # nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            # nn.Softmax(dim=-1),
        )

        self.hidden = None
        self.trade_memory = None
        self.max_memory_size = max_memory_size

    def _init_hidden(self):
        return torch.zeros(self.num_layers, self.hidden_dim)
    
    def _init_trade_memory(self):
        return torch.zeros(0, self.hidden_dim)
    
    def _attention(self, query, keys, values, batch_size: int = 1):
        query, keys, values = (
            query.unsqueeze(0), 
            keys.unsqueeze(0), 
            values.unsqueeze(0)
        )
        query = self.query.repeat(keys.size(0), 1)
        attn_scores = torch.bmm(
            self.key_transform(keys),
            query.unsqueeze(2)
        )
        attn_scores = attn_scores.squeeze(2)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(1)
        context = torch.bmm(attn_weights, self.value_transform(values))
        return context.squeeze(1)

    def _normalize(self, x):
        return torch.abs(x) / torch.sum(torch.abs(x), dim=1, keepdim=True)
    
    def reset_state(self):
        self.hidden = None
        self.trade_memory = None

    def forward(self, x, dist, return_state: bool = False):
        self.hidden = self._init_hidden() if self.hidden is None else self.hidden
        self.trade_memory = self._init_trade_memory() if self.trade_memory is None else self.trade_memory

        trade_encoded = self.encoder(x)
        output, self.hidden = self.memory(trade_encoded, self.hidden)
        self.hidden = self.hidden.detach()

        if self.trade_memory.size(0) >= self.max_memory_size:
            self.trade_memory = self.trade_memory[1:, :]

        self.trade_memory = torch.cat([self.trade_memory, output], dim=0)
        self.trade_memory = self.trade_memory.detach()
        
        context = self._attention(
            self.query,
            self.trade_memory,
            self.trade_memory
        )
        dist = torch.from_numpy(dist).float()
        combined = torch.cat([context, trade_encoded], dim=1)
        # output = torch.abs(self.output_layer(combined))
        # output = output / torch.sum(output, dim=1, keepdim=True)
        output = self.output_layer(combined)
        output = self._normalize(output + dist)
        # output = self.output_layer(combined)
        if return_state:
            return output, self.trade_memory, self.hidden
        return output


def normalize(x):
    return x / np.sum(x)


def train_model(
        model, 
        daily_data, 
        features, 
        look_ahead: int = 50,
        look_ahead_bias: float = 0.25,
        num_epochs: int = 3, 
        learning_rate: float = 0.01,
    ):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()

    for i, day in enumerate(daily_data):
        df = daily_data[day]
        dist_cols = [f for f in df.columns.tolist() if f.startswith("strike_")]
        result = np.argmax(df.iloc[-1][dist_cols].values)

        final_dist = np.zeros(len(dist_cols))
        final_dist[result] = 1

        if i == 2:

            y_pred, y_true = [], []
            for i in range(len(df) - look_ahead):
                look_ahead_dist = normalize(np.array([
                    float(x) for x in df.iloc[i + look_ahead][dist_cols].values
                ]))

                y = look_ahead_dist * (1 - look_ahead_bias) + final_dist * look_ahead_bias
                x = np.array(df.iloc[i][features].values, dtype=np.float32).reshape(1, -1)
                x = x / np.sum(x)
                dist = np.array([
                    float(x) for x in df.iloc[i][dist_cols].values
                ])
                x = torch.from_numpy(x).float()
                y = torch.from_numpy(y).float().unsqueeze(0)

                output = model(x, dist)
                y_pred.append([float(x) for x in output.detach().numpy()[0]])
                y_true.append(y.detach().numpy()[0])
            idx = np.argmax(y_true[-1])
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)
            plt.plot(list(range(len(y_pred))), y_pred[:, idx], label="pred")
            plt.plot(list(range(len(y_true))), y_true[:, idx], label="true")
            plt.legend()
            plt.show()
            break
        

        for _ in range(num_epochs):
            loss_sum = 0
            for j in range(len(df) - look_ahead):  
                look_ahead_dist = normalize(np.array([
                    float(x) for x in df.iloc[j + look_ahead][dist_cols].values
                ]))
                

                y = look_ahead_dist * (1 - look_ahead_bias) + final_dist * look_ahead_bias
                x = np.array(df.iloc[j][features].values, dtype=np.float32).reshape(1, -1)
                dist = np.array([
                    float(x) for x in df.iloc[j][dist_cols].values
                ])
                x = x / np.sum(x)
                # print(x)

                x = torch.from_numpy(x).float()
                y = torch.from_numpy(y).float().unsqueeze(0)

                optimizer.zero_grad()
                output = model(x, dist)
                loss = criterion(output, y)
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()
            
            print(f'Epoch {_ + 1} loss: {loss_sum / len(df)}')
            model.reset_state()
        
        

if __name__ == "__main__":

    loader = Loader(
        data_dir="../data",
    )
    daily_data = loader.load_daily_data(
        ticker="kxhighny",
        max_days=100,
        type_="polysignal",
    )

    features = [
        f
        for f in daily_data[list(daily_data.keys())[0]].columns.tolist()
        if f not in ["time", "ticker", "trade_id"]

    ]
    output_dim = 6
    look_ahead = 50
    look_ahead_bias = 0
    num_epochs = 50
    learning_rate = 1e-4

    model = OnlineModel(
        input_dim=len(features),
        hidden_dim=len(features),
        num_layers=1,
        output_dim=output_dim,
        max_memory_size=10,
    )
    train_model(
        model, 
        daily_data, 
        features, 
        look_ahead, 
        look_ahead_bias, 
        num_epochs,
        learning_rate,
    )