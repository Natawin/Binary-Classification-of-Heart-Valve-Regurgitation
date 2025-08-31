import torch.nn as nn
import torch


class MultiValveCNN(nn.Module):
    def __init__(self, num_valves=4):
        super().__init__()
        self.embedding = nn.Embedding(num_valves, 16)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x, valve_idx):
        x_feat = self.cnn(x)
        v_feat = self.embedding(valve_idx)
        x_feat = torch.flatten(x_feat, 1)
        all_feat = torch.cat((x_feat, v_feat), dim=1)
        out = self.fc(all_feat)
        return out

