# AlphaZero_Gomoku_PaddlePaddle


## AlphaZero_Gomoku_PaddlePaddle
- 这是AlphaZero算法的一个实现（使用PaddlePaddle框架），用于玩简单的棋盘游戏Gomoku（也称为五子棋），使用纯粹的自我博弈的方式开始训练。Gomoku游戏比围棋或象棋简单得多，因此我们可以专注于AlphaZero的训练方案，在一台PC机上几个小时内就可以获得一个相当好的AI模型。 
- 因为和围棋相比，五子棋的规则较为简单，落子空间也比较小，因此没有用到AlphaGo Zero中大量使用的残差网络，只使用了卷积层和全连接层。

References:  
1. AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
2. AlphaGo Zero: Mastering the game of Go without human knowledge
3. 郭宪，宋俊潇. 深入浅出强化学习编程实战 —— 电子工业出版社2020
4. 邹伟. 强化学习 ——清华大学出版社 2020
4. 高随祥，文新，马艳军，李轩涯等. 深度学习导论与应用实践 ——清华大学出版社 2019




### Requirements
- Python >= 3.7
- Numpy >= 1.11
- PaddlePaddle >= 1.8.0 


### Getting Started
开始人机对战或者看AI对战, 请运行文件"human_play.py":  
```
python human_play.py  
```

要训练AI模型的话，运行文件"train.py":   
```
python train.py
```


模型 (best_policy.model and current_policy.model) 每隔一定的步数会被保存 (默认50)。


**训练技巧:**
1. 最好从6 * 6的棋盘、4子连成直线获胜开始训练。这样的话，我们可以在大约2个小时内，以500~1000局的自我博弈获得一个让人不可大意的模型，因为一不小心，AI就可能战胜了你。
2. 对于 8 * 8 的棋盘、 5子连成直线获胜的情况, 大约需要2000~3000局的自我博弈得到一个好的模型，在单台电脑上面训练可能需要2天时间。
3. 对于 15 * 15的棋盘、 5子连成直线获胜的情况，为了获得一个优秀的AI，预计需要进行百万次的对局，因此对设备的要求较高。



