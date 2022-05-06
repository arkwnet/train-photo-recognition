# train-photo-recognition

TensorFlow の機械学習で鉄道写真から形式を特定するサンプル

## 開発環境

~~Google Colaboratory~~  
2022 年 4 月 22 日更新分より、ローカル環境の Python 3.10.4 + TensorFlow 2.8.0 に変更

## 動作手順

1. `python train.py`を実行し、学習モデルを生成
2. コンソール出力を確認し、検証データの損失（`loss`）が最小の学習モデルを選ぶ
3. `python predict.py [trainxx.hdf5] [画像ファイル.jpg]`を実行
4. 以下のような形式で識別結果が出力されます

```
keikyu-1000        14.5047%
meitetsu-3500      34.7103%
meitetsu-6000      50.7850%
```

`predict_api.py`は Flask を用いた API サーバです。  
標準出力の代わりに JSON 形式で結果を返します。

## 参考文献

everylittle. "[TensorFlow] AI を車両鉄に入門させてみた". Qiita. https://qiita.com/everylittle/items/954207b1ae917c25ff96, (参照 2022-02-08)

## ライセンス

### 京急 1000 形

MaedaAkihiko - 投稿者自身による作品, CC 表示-継承 4.0, https://commons.wikimedia.org/w/index.php?curid=107656447 による  
Chabata_k(Japan) - Picture taken by Chabata_k(Japan)., CC 表示-継承 3.0, https://commons.wikimedia.org/w/index.php?curid=2679807 による  
MaedaAkihiko This photo was taken with Panasonic Lumix DC-FZ1000 II - 投稿者自身による作品, CC 表示-継承 4.0, https://commons.wikimedia.org/w/index.php?curid=95729775 による  
MaedaAkihiko This photo was taken with Panasonic Lumix DC-FZ1000 II - 投稿者自身による作品, CC 表示-継承 4.0, https://commons.wikimedia.org/w/index.php?curid=95729782 による  
MaedaAkihiko - 投稿者自身による作品, CC 表示-継承 4.0, https://commons.wikimedia.org/w/index.php?curid=107656444 による  
Tennen-Gas This photo was taken with Canon EOS 5D - 投稿者自身による作品, CC 表示-継承 3.0, https://commons.wikimedia.org/w/index.php?curid=4131084 による  
MaedaAkihiko - 投稿者自身による作品, CC 表示-継承 4.0, https://commons.wikimedia.org/w/index.php?curid=107656441 による  
Inatewi - 投稿者自身による作品, CC 表示-継承 4.0, https://commons.wikimedia.org/w/index.php?curid=72161670 による  
MaedaAkihiko This photo was taken with Panasonic Lumix DC-FZ1000 II - 投稿者自身による作品, CC 表示-継承 4.0, https://commons.wikimedia.org/w/index.php?curid=95729781 による  
MaedaAkihiko - 投稿者自身による作品, CC 表示-継承 4.0, https://commons.wikimedia.org/w/index.php?curid=95729789 による  
MaedaAkihiko - 投稿者自身による作品, CC 表示-継承 4.0, https://commons.wikimedia.org/w/index.php?curid=107656435 による  
MaedaAkihiko - 投稿者自身による作品, CC 表示-継承 4.0, https://commons.wikimedia.org/w/index.php?curid=95729777 による

### 名鉄 3500 系

ButuCC - 投稿者自身による作品, CC 表示-継承 4.0, https://commons.wikimedia.org/w/index.php?curid=39168074 による  
Tennen-Gas - 投稿者自身による作品, CC 表示-継承 3.0, https://commons.wikimedia.org/w/index.php?curid=4124958 による  
Tennen-Gas - 投稿者自身による作品, CC 表示-継承 3.0, https://commons.wikimedia.org/w/index.php?curid=7918008 による  
ButuCC - 投稿者自身による作品, CC 表示-継承 4.0, https://commons.wikimedia.org/w/index.php?curid=35836115 による  
Tennen-Gas - 投稿者自身による作品, CC 表示-継承 3.0, https://commons.wikimedia.org/w/index.php?curid=7918150 による  
Tennen-Gas - 投稿者自身による作品, CC 表示-継承 3.0, https://commons.wikimedia.org/w/index.php?curid=7918038 による  
ButuCC - 投稿者自身による作品, CC 表示-継承 4.0, https://commons.wikimedia.org/w/index.php?curid=96129140 による  
ButuCC - 投稿者自身による作品, CC 表示-継承 4.0, https://commons.wikimedia.org/w/index.php?curid=71024589 による

### 名鉄 6000 系

Tennen-Gas - 投稿者自身による作品, CC 表示-継承 3.0, https://commons.wikimedia.org/w/index.php?curid=6491179 による  
Tennen-Gas - 投稿者自身による作品, CC 表示-継承 3.0, https://commons.wikimedia.org/w/index.php?curid=4361012 による  
Tennen-Gas - 投稿者自身による作品, CC 表示-継承 3.0, https://commons.wikimedia.org/w/index.php?curid=8266547 による  
Tennen-Gas - 投稿者自身による作品, CC 表示-継承 3.0, https://commons.wikimedia.org/w/index.php?curid=4361201 による  
CC 表示-継承 3.0, https://commons.wikimedia.org/w/index.php?curid=770208  
ButuCC - 投稿者自身による作品, CC 表示-継承 4.0, https://commons.wikimedia.org/w/index.php?curid=73839366 による
