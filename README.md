# time-series-prediction

## 概要
「日付」「終値」カラムを持つCSVファイルを読み込み、LSTMを用いて株価を予測するモデルです。  

## 実行方法
### 予測の実行  
それぞれの`.py` ファイルを直接実行することで、LSTMによる株価予測が可能です。  

### 各スクリプトの役割  
- `src/1step_lstm.py`：1ステップ先の株価を予測し、評価・プロットを行う。  
- `src/somesteps_lstm.py`：30ステップ先の株価を予測し、評価・プロットを行う。  
- `src/data_check.py`：データの特徴や欠損をチェックする。  

## データ
- `resources/stock_price.csv`：読み込む株価データ。  

## その他
- `presentation.pdf`：本プロジェクトのプレゼン資料。
