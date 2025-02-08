import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.font_manager as fm

# 日本語フォントの設定（MSゴシック）
font_path = 'C:/Windows/Fonts/msgothic.ttc'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# CSVの読み込み（エンコーディング調整）
df = pd.read_csv("./resources/stock_price.csv", encoding="utf-8", parse_dates=["日付け"], index_col="日付け")

# データの前処理
df["変化率 %"] = df["変化率 %"].str.replace("%", "").astype(float)  # 変化率を数値化
# 出来高の前処理（M を100万倍, B を10億倍）
df["出来高"] = df["出来高"].replace({"M": "*1e6", "B": "*1e9"}, regex=True).map(pd.eval).astype(float)
# データの確認
print(df.info())  # データ型や欠損値の確認
print(df.describe())  # 基本統計量の確認

# ① 終値の推移
plt.figure(figsize=(12, 6))
plt.plot(df["終値"], label="Close Price", color="blue")
plt.title("NTT 株価の推移")
plt.legend()
plt.show()

# ② 出来高の推移
plt.figure(figsize=(12, 6))
plt.plot(df["出来高"], label="Trading Volume", color="purple")
plt.title("NTT 出来高の推移")
plt.legend()
plt.show()

# ③ 移動平均の計算（30日、90日）
df["MA30"] = df["終値"].rolling(window=30).mean()
df["MA90"] = df["終値"].rolling(window=90).mean()

plt.figure(figsize=(12, 6))
plt.plot(df["終値"], label="Close Price", color="blue", alpha=0.5)
plt.plot(df["MA30"], label="30-day MA", color="red")
plt.plot(df["MA90"], label="90-day MA", color="green")
plt.title("移動平均の比較")
plt.legend()
plt.show()

# ④ 変化率のヒストグラム
plt.figure(figsize=(8, 5))
sns.histplot(df["変化率 %"], bins=50, kde=True)
plt.title("日次変化率の分布")
plt.show()

# ⑤ 自己相関（ACF/PACF）
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_acf(df["終値"].dropna(), lags=50, ax=axes[0])
plot_pacf(df["終値"].dropna(), lags=50, ax=axes[1])
plt.show()
