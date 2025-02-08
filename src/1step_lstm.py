import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# ===== 1. データ前処理 =====
class StockDataset(Dataset):
    def __init__(self, df, seq_length=30):
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(df[["終値"]])
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]  # 直近30日分
        y = self.data[idx + self.seq_length]  # 未来1日の[終値, 出来高, 変化率]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ===== 2. LSTMモデル =====
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])  # 最後の時刻の出力を取得
        return self.fc(lstm_out)

# ===== 3. ハイパーパラメータとデータセット作成 =====
seq_length = 30 #入力時系列の長さ
input_dim = 1  # [終値, 出来高, 変化率 %]
hidden_dim = 64
output_dim = input_dim  # 未来1日の [終値, 出来高, 変化率]
num_layers = 4
batch_size = 16
epochs = 3
learning_rate = 0.001

df = pd.read_csv("./resources/stock_price.csv", encoding="utf-8", parse_dates=["日付け"], index_col="日付け")

# 日付を古い順に並べ替える
df = df.sort_index(ascending=True)

df["変化率 %"] = df["変化率 %"].str.replace("%", "").astype(float)
df["出来高"] = df["出来高"].replace({"M": "*1e6", "B": "*1e9"}, regex=True).map(pd.eval).astype(float)

dataset = StockDataset(df, seq_length=seq_length)
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size - seq_length:]

train_dataset = StockDataset(train_df, seq_length=seq_length)
test_dataset = StockDataset(test_df, seq_length=seq_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ===== 4. モデル学習 =====
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # テストデータでの評価
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            test_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}")

# ===== 5. 予測（テストデータ全体） =====
model.eval()

actual_values = []
future_preds = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_pred = model(x_batch).cpu().numpy()  # 予測値
        actual_values.append(y_batch.numpy())  # 正解値
        future_preds.append(y_pred)

actual_values = np.array(actual_values).squeeze().reshape(-1, output_dim)
future_preds = np.array(future_preds).squeeze().reshape(-1, output_dim)

# スケール変換を元に戻す
actual_values_unscaled = test_dataset.scaler.inverse_transform(actual_values)[:, 0]
future_preds_unscaled = test_dataset.scaler.inverse_transform(future_preds)[:, 0]

# ===== 6. プロット（終値のみ） =====
plt.figure(figsize=(12, 6))
plt.plot(actual_values_unscaled, label="Actual Close Price", color="blue")
plt.plot(future_preds_unscaled, label="Predicted Close Price", color="red", linestyle="dashed")
plt.legend()
plt.title("Stock Price Prediction (Close Price)")
plt.xlabel("Time (days)")
plt.ylabel("Stock Price")
plt.savefig("result_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# ===== 7. MSE 計算（終値のみ） =====
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(actual_values_unscaled, future_preds_unscaled)
print(f"MSE (Close Price): {mse:.4f}")


# plot_step日前の30日移動平均を取得
df["終値_shifted"] = df["終値"].shift(1)

# NaNを含む行を削除（rollingとshiftの影響で生じる）
df = df.dropna()

# MSEの計算
mse_30ma = mean_squared_error(df["終値_shifted"], df["終値"])

# RMSEの計算
rmse_30ma = np.sqrt(mse_30ma)

print(f"MSE (前日の終値との差、前日をそのまま次の日のMSEとするモデルとの比較): {mse_30ma:.4f}")
