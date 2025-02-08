import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# ===== 1. データ前処理 =====
class StockDataset(Dataset):
    def __init__(self, df, seq_length=30, future_steps=5):
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(df[["30日移動平均"]])
        self.seq_length = seq_length
        self.future_steps = future_steps

    def __len__(self):
        return len(self.data) - self.seq_length - self.future_steps

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]  # 過去データ
        y = self.data[idx + self.seq_length : idx + self.seq_length + self.future_steps]  # 未来データ
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, future_steps=5):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.future_steps = future_steps
        self.num_layers = num_layers
        self.hidden_size = hidden_dim

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # LSTM の初期状態
        h_n = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_n = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM の処理 (過去の時系列を入力)
        lstm_out, (h_n, c_n) = self.lstm(x, (h_n, c_n))
        
        # 過去のデータの最後の出力を取得し、FC を通して次の入力として使用
        last_output = self.fc(lstm_out[:, -1, :])  # (batch_size, output_dim)
        
        # 入力のスライドウィンドウを用意
        input_step = torch.cat([x[:, 1:, :], last_output.unsqueeze(1)], dim=1)  # (batch_size, seq_len, input_dim)
        
        predictions = []
        
        for _ in range(self.future_steps):
            # LSTM に渡すための適切な次元に変換
            lstm_out, (h_n, c_n) = self.lstm(input_step, (h_n, c_n))
            lstm_out = self.dropout(lstm_out)

            # 予測値を取得
            pred = self.fc(lstm_out[:, -1, :])  # (batch_size, output_dim)
            predictions.append(pred.unsqueeze(1))  # (batch_size, 1, output_dim)

            # 次のステップの入力を作成
            input_step = torch.cat([input_step[:, 1:, :], pred.unsqueeze(1)], dim=1)  # (batch_size, seq_len, input_dim)

        return torch.cat(predictions, dim=1)  # (batch_size, future_steps, output_dim)

# ===== 3. ハイパーパラメータとデータセット作成 =====
seq_length = 300  # 過去のデータ長
future_steps = 30  # 未来の予測ステップ
plot_step = 29 #プロット、比較、mseの計算をするステップ（何ステップ先の予測性能を見るか）
input_dim = 1
hidden_dim = 64
output_dim = 1
num_layers = 4
batch_size = 16
epochs = 3
learning_rate = 0.001

df = pd.read_csv(".resources/stock_price.csv", encoding="utf-8", parse_dates=["日付け"], index_col="日付け")
#昇順、降順の並び替え
df = df.sort_index(ascending=True)
#変化率、出来高の数値化
df["変化率 %"] = df["変化率 %"].str.replace("%", "").astype(float)
df["出来高"] = df["出来高"].replace({"M": "*1e6", "B": "*1e9"}, regex=True).map(pd.eval).astype(float)
# 30日移動平均を計算して新しい列を追加
df["30日移動平均"] = df["終値"].rolling(window=30).mean()
# NaN を含む行を削除（移動平均を計算できない最初の 29 行）
df = df.dropna()

dataset = StockDataset(df, seq_length=seq_length, future_steps=future_steps)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
# 時系列の順番を保ったまま分割
train_dataset = StockDataset(df.iloc[:train_size], seq_length=seq_length, future_steps=future_steps)
test_dataset = StockDataset(df.iloc[train_size+seq_length+future_steps:], seq_length=seq_length, future_steps=future_steps)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM(input_dim, hidden_dim, output_dim, num_layers, future_steps).to(device)
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

    #テストデータでの評価
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            test_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}")
# ===== 5. 予測 =====
model.eval()
actual_values, future_preds = [], []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_pred = model(x_batch).cpu().numpy()
        actual_values.append(y_batch.numpy())
        future_preds.append(y_pred)

actual_values = np.array(actual_values).squeeze()
future_preds = np.array(future_preds).squeeze()

# スケール変換を元に戻す
actual_values_unscaled = dataset.scaler.inverse_transform(actual_values)
future_preds_unscaled = dataset.scaler.inverse_transform(future_preds)
# 右に plot_step だけシフト
shifted_values = np.roll(actual_values_unscaled[:, plot_step], shift=plot_step)

# シフトした分の先頭に NaN を埋める（または他の値）
shifted_values[:plot_step] = np.nan  # 例: NaN埋め
# ===== 6. プロット =====
plt.figure(figsize=(12, 6))
plt.plot(actual_values_unscaled[:, plot_step], label="Actual Close Price", color="blue")
plt.plot(shifted_values, label="Delayed Actual Close Price", color="yellow")
plt.plot(future_preds_unscaled[:, plot_step], label="Predicted Close Price", color="red", linestyle="dashed")
plt.legend()
plt.title("Stock Price Prediction (Close Price)")
plt.xlabel("Time (days)")
plt.ylabel("Stock Price")
file_name = f"forecasted_stock_price_seq{seq_length}_fut{future_steps}_step{plot_step}.png"
plt.savefig(file_name, dpi=300, bbox_inches="tight")
plt.show()

# ===== 7. MSE 計算 =====
from sklearn.metrics import mean_squared_error
for i in range(future_steps):
    mse = mean_squared_error(actual_values_unscaled[:, i], future_preds_unscaled[:, i])
    # RMSE（平方根を取る）
    rmse = np.sqrt(mse)
    print(f"RMSE (Close Price, step={i}): {rmse:.4f}")
    # plot_step日前の30日移動平均を取得
    df[f"{i}日移動平均"] = df["30日移動平均"].shift(i)
    # NaNを含む行を削除（rollingとshiftの影響で生じる）
    df = df.dropna()
    # MSEの計算、plot_step日後の株価をだす最も単純なモデルとの比較
    mse_30ma = mean_squared_error(df[f"{i}日移動平均"], df["30日移動平均"])
    # RMSEの計算
    rmse_30ma = np.sqrt(mse_30ma)
    print(f"RMSE (30日移動平均, {i}日前との誤差): {rmse_30ma:.4f}")
