import sys
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import random



#ゲームボードの位置を表すために使用される定数のリストとタプル
gVec = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
gCol = ('A','B','C','D','E','F','G','H')
gRow = ('1','2','3','4','5','6','7','8')

# (1)
#MLP（多層パーセプトロン）モデルの定義
#3つの全結合層からなる単純なニューラルネットワークモデルを定義
#モデルはReLU活性化関数を使用

#MLPクラスでMLPの構造を定義します。前述のとおり、入力=8x8=64、出力=65、隠れ層2層(ニューロン数100)で定義
#活性化関数と層構造は _call_ 内で定義しています。
#ClassifierクラスでSoftmaxでのクラス分けを定義しています。



class MLP(Chain):#中間層2層
    def __init__(self):
        super(MLP, self).__init__(
                l1=L.Linear(64, 100),
                l2=L.Linear(100, 100),
                l3=L.Linear(100, 64)
        )
 
    def __call__(self, x):
        h1 = F.relu(self.l1(x))#　x 入力　y 出力
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

class Classifier(Chain):#ネットワークのトレーニングおよび評価
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x, t):# t 正解ラベル
        y = self.predictor(x)# xをpredictorに渡し出力を得る 
        loss = F.softmax_cross_entropy(y, t)#交差エントロピー誤差（softmax_cross_entropy）損失
        accuracy = F.accuracy(y, t)#精度
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss
        

def print_board(board):
    for i in range(8):
        print(board[i])

    print("")

def update_board(board, pos_str, clr):#() オセロのルールが書いてある関数 
    assert clr!=0, "stone color is not black or white."
    updated_board = [[0 for col in range(8)] for row in range(8)]#更新後のゲームボードを保持するための2次元リスト
    rev_list = []#反転される石の位置を追跡するためのリスト
    pos = pos_str2pos_index(pos_str)#pos_str をゲームボード上の座標に変換
    for v in gVec:
        temp_list = []
        for i in range(1, 8):
            # out of board
            if pos[0]+v[0]*(i+1) > 7 or pos[1]+v[1]*(i+1) > 7 or\
               pos[0]+v[0]*(i+1) < 0 or pos[1]+v[1]*(i+1) < 0:
                continue

            if board[pos[0]+v[0]*i][pos[1]+v[1]*i] == (clr % 2 + 1):
                temp_list.append([pos[0]+v[0]*i, pos[1]+v[1]*i])

                if board[pos[0]+v[0]*(i+1)][pos[1]+v[1]*(i+1)] == clr:
                    for j in temp_list:
                        rev_list.append(j)

                    break
            else:
                break

    rev_list.append(pos)  # put stone at pos
    assert board[pos[0]][pos[1]] == 0, "put position is not empty."
    print("rev_list = " + str(rev_list))
    for i in range(0, 8):
        for j in range(0, 8):
            if [i, j] in rev_list:
                updated_board[i][j] = clr
            else:
                updated_board[i][j] = board[i][j]

    return updated_board

def who_is_winner(board):#ボードから勝敗を決める関数
    # ret : 0  draw
    #       1  black win
    #       2  white win
    ret = 0
    score_b = 0
    score_w = 0
    for i in range(0, 8):
        for j in range(0, 8):
            if board[i][j] == 1:
                score_b += 1
            elif board[i][j] == 2:
                score_w += 1

    if score_b > score_w:
        ret = 1
    elif score_b < score_w:
        ret = 2

    print("Black vs White : " + str(score_b) + " vs " + str(score_w))
    return ret

def pos_str2pos_index(pos_str):#pos_str をゲームボード上の座標に変換
    pos_index = []
    for i, c in enumerate(gRow):
        if pos_str[1] == c:
            pos_index.append(i)

    for i, c in enumerate(gCol):
        if pos_str[0] == c:
            pos_index.append(i)


    return pos_index

def pos_str2pos_index_flat(pos_str):#オセロの棋譜をモデルに入力する際に、棋譜の位置を一意の整数値に変換
    pos_index = pos_str2pos_index(pos_str)
    index = pos_index[0] * 8 + pos_index[1]
    return index


#==== Main ====#
record_X = []    # 1万試合分の棋譜
record_y = []    # 棋譜に対する出力値
temp_X = []
temp_y = []
temp2_X = []
temp2_y = []
board = []
row = []

argv = sys.argv
argc = len(argv)

if argc != 3:
    print('Usage') 
    print('    python ' + str(argv[0]) + ' <record_filename> <type>') 
    print('        type : black') 
    print('               black_win') 
    print('               white') 
    print('               white_win') 
    quit()

# check type
build_type = ''
for t in ['black', 'black_win', 'white', 'white_win']:
    if argv[2] == t:
        build_type = t

if build_type == '':
    print('record type is illegal.') 
    quit()

#(2)-- load record --#
#(2) 棋譜の読み込みと変換
#長いループですが、棋譜の読み込みと、それを入力(8x8)と出力(0～64)に変換しています。
#プログラミングの何が大変ってここが一番大変でした...
f = open(argv[1], "r")
line_cnt = 1
for line in f:
    print('Line Count = ' + str(line_cnt)) 
    idx = line.find("BO[8")
    if idx == -1:
        continue

    idx += 5
    # make board initial state
    for i in range(idx, idx+9*8):
        if line[i] == '-':
            row.append(0)
        elif line[i] == 'O':
            row.append(2)
        elif line[i] == '*':
            row.append(1)

        if (i-idx)%9 == 8:
            board.append(row)
            row = []
            if len(board) == 8:
                break

    row = []
    print_board(board)
    # record progress of game
    i = idx+9*8+2
    while line[i] != ';':
        if (line[i] == 'B' or line[i] == 'W') and line[i+1] == '[':
            temp_X.append(board)
            pos_str = line[i+2] + line[i+3]
            if pos_str == "pa":    # pass
                temp_y.append(64)
                # board state is not change
                print_board(board)
            else:
                if line[i] == 'B':
                    clr = 1
                elif line[i] == 'W':
                    clr = 2
                else:
                    clr = 0
                    assert False, "Stone Color is illegal."

                pos_index_flat = pos_str2pos_index_flat(pos_str)
                temp_y.append(pos_index_flat)
                board = update_board(board, pos_str, clr)

            if (line[i] == 'B' and (build_type == 'black' or build_type == 'black_win')) or \
               (line[i] == 'W' and (build_type == 'white' or build_type == 'white_win')):
                temp2_X.append(temp_X[0])
                temp2_y.append(temp_y[0])
                print('X = ') 
                print_board(temp_X[0])
                print('y = ' + str(temp_y[0]) + ' (' + \
                               str(pos_str2pos_index(pos_str)) + ') ' + \
                               '(' + pos_str + ')') 
                print('')

            temp_X = []
            temp_y = []

        i += 1

    print("End of game") 
    print_board(board)

    winner = who_is_winner(board)
    if (winner == 1 and build_type == 'black_win') or \
       (winner == 2 and build_type == 'white_win') or \
       build_type == 'black' or build_type == 'white':
        record_X.extend(temp2_X)
        record_y.extend(temp2_y)

    board = []
    temp2_X = []
    temp2_y = []
    line_cnt += 1






#(3)-- MLP model and Training --#

#(3) MLPモデルのトレーニング
#変換した棋譜をモデルに入力してトレーニングします。
#入力、出力ともにNumpyのarray形式にして、dataset.TupleDataset で設定できます。

##あとはソースコードのように

#batch_sizeの設定
#最適化方法(今回はSGD)の設定
#実行回数(epoch)の設定



#record_X と record_y のデータをNumPy配列に変換し、データ型を指定


# passの手を除去
indices_to_remove = [i for i, value in enumerate(record_y) if value == 64]
indices_to_remove.reverse()  # 逆順にしないと順番ずれる

for index in indices_to_remove:
    del record_X[index]
    del record_y[index]


for i in record_y:
    if i == 64:
        print("w")

'''

#データ数調整
record_X = record_X[:118579]     
record_y = record_y[:118579]      
'''

print(record_y)
print(len(record_X))


'''
# 打ち手の分布を配列で表現
y_num = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

for i in record_y:
    y_num[i] += 1

print(y_num)
'''





'''
###データをシャッフル

zipped_lists = list(zip(record_X, record_y))  # 対応する要素をペアにする
random.shuffle(zipped_lists)  # ペアをランダムにシャッフル

shuffled_list_1, shuffled_list_2 = zip(*zipped_lists)

'''



#データの準備
X = np.array(record_X, dtype=np.float32)
y = np.array(record_y, dtype=np.int32)
train = datasets.TupleDataset(X, y)



train_iter = iterators.SerialIterator(train, batch_size=100)

model = Classifier(MLP())

#確率的勾配降下法
optimizer = optimizers.SGD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)

trainer = training.Trainer(updater, (1000, 'epoch'), out='result')




trainer.extend(extensions.ProgressBar())

trainer.run()

#保存
serializers.save_npz('model_ggs_black_1layer_100n.npz', model)



#実行コマンド　python3 cpu.py Othello.01e4.ggf black_win



