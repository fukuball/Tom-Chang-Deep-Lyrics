# Tom-Chang-Deep-Lyrics

基於 LSTM 深度學習方法研發而成的張雨生歌詞產生模型，致敬張雨生

# 環境安裝

```
$ git clone https://github.com/fukuball/Tom-Chang-Deep-Lyrics.git
$ cd Tom-Chang-Deep-Lyrics
$ pip install -r requirements.txt
$ mkdir model
```

# 訓練

```
$ python train.py path_to_your_corpus.txt

$ python train.py lyrics/all_training_lyrics.txt
```

# 產生歌詞

```
$ python generate.py "start sentence"

$ python generate.py "漂向北方 別問我家鄉"
```

