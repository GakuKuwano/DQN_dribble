# DQN_dribble
Robot Design 4 : Generation of Dribble Trajectories Using Reinforcement Learning.

ロボットの制御指令を出力とする深層強化学習を行い、目的地へドリブルする動作を学習させる。

## プログラムの実行

以下のコマンドで学習させる。

```sh
cd dribble_dqn
python train.py
```

以下のコマンドで学習済みのデータを用いて動かす。

```sh
cd dribble_dqn
python predict.py result/-/-.h5
```

## 実行結果

学習途中

https://drive.google.com/file/d/1kR76qv_U072IAZ35c7qeURwYtEEJEV9k/view?usp=sharing

1000回学習後

https://drive.google.com/file/d/1CeGiz-8_oei5AqAxq-B9WBY2rcGCgni0/view?usp=sharing
