# 一天搞懂深度學習 心得筆記

[一天搞懂深度學習](http://datasci.tw/event/deep_learning_one_day/)由李宏毅助理教授濃縮在台大上一個學期 [Machine Learning and having it deep and structured](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLSD15_2.html) 的課程精華，授課淺顯易懂，[投影片](http://www.slideshare.net/tw_dsconf/ss-62245351)內容精彩豐富。以下是重點整理與學習心得：

## 第一講 介紹深入學習

### 深度學習三步驟

- 建立網路結構
- 學習目標
- 開始學習

人類腦中有許多神經元與突觸，經過學習認知，有些神經元的連結被強化，有些則退化或消失。神經網路也與人腦類似，透過學習已知事物建立認知模型，然後面對未知做出判斷。

#### 步驟一 建立網路結構

神經元 (Neuron) 是神經網路的基本元件，用以下公式表示：

z = a<sub>1</sub>w<sub>1</sub> + ... + a<sub>k</sub>w<sub>k</sub> + ... + a<sub>K</sub>w<sub>K</sub> + b

a = σ(z)

- a<sub>1</sub> ~ a<sub>K</sub>: 輸入
- a: 輸出
- w: weight (權重)
- b: bias (偏移量)
- σ(z): activation function (活化函數)

這是一個簡單的多元一次方程式，一組輸入 (a<sub>1</sub>, ..., a<sub>k</sub>, ..., a<sub>K</sub>)，經過一組權重 (w<sub>1</sub>, ..., w<sub>k</sub>, ..., w<sub>K</sub>) 調整後，加上偏移量 b，得到中間值 z，再通過活化函式 σ(z)，得到輸出 a

σ(z) 最初採用 Sigmoid function (= 1 / 1 + e<sup>-z</sup>)，當輸入值很小得到 0，當輸入值很大得到 1，輸入值為零得到 0.5。使用 Sigmoid function 把輸出值限定在一個可控制的範圍，才不會經過多層網路放大差異。

多個神經元組成神經網路 (Neuron Network)，每個神經元有不同的權重與偏移量，權重與偏移量將透過第三步驟求得。

所有層與層間的神經元都相互連結的神經網路，稱作完全連結前饋網路 (Fully Connected Feedforward Network)。如果上層有 N 個神經元，下層有 M 個神經元，那麼就有 NxM 個連結，表示權重有 NxM 個。

每層輸入以向量 x 表示，權重以矩陣 W 表示，偏移量以向量 b 表示，經過活化函數 σ() 得到輸出。

- 第一層: σ(W<sup>1</sup>x + b<sup>1</sup>)
- 第二層: σ(W<sup>2</sup>x + b<sup>2</sup>)
- ...
- 第Ｌ層: σ(W<sup>L</sup>x + b<sup>L</sup>)

輸入層輸入 (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>N</sub>) 經過多個隱藏層作用，每個隱藏層中包含多個神經元，最後透過輸出層得到結果 (y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>N</sub>)。

y = f(x) = σ(W<sup>L</sup> ... σ(W<sup>2</sup>σ(W<sup>1</sup>x + b<sup>1</sup>) + b<sup>2</sup>) ... + b<sup>L</sup>)

最後，神經網路使用 softmax layer 作為輸出層 (Output Layer)，讓輸出結果呈現機率分佈，所有可能加總為 1。

#### 步驟二 學習目標

輸入一張 16x16 的圖片，有墨水的地方為1，沒有墨水的地方為0，形成一個一為的陣列(x<sub>1</sub>,x<sub>2</sub>, ..., x<sub>256</sub>)。

輸出一個 10 種元素的陣列，分別對應 (0, 1, 2, ..., 9) 的機率。

創造一個 full connected feedforward network 模型，裡面的 hidden layers 形成一個非常複雜的函數。接下來，第三步驟的目標就是透過機器學習就是找出適合的函數參數。

我們希望

- 當圖片是 0, 找出 0 的機率最高
- 當圖片是 1, 找出 1 的機率最高
- ...
- 當圖片是 9, 找出 9 的機率最高

使用 loss 評估準確度，當圖片為 1，得到一個機率的矩陣，除了結果為 1 之外，其餘都是錯誤答案，使用 square error 或 cross entropy  評估錯誤率，計算輸出結果與目標之間的差距。

加總所有錯誤，得到 L = Σ l<sub>r</sub>，然後經過學習找出一組讓 L 最小的參數。

#### 步驟三 開始學習

- 目標：找出讓 total loss 最小化的網路參數
- 學習：解出方程式最佳解!?

這個硬幹的做法不切實際，以兩層 1000 個神經元的神經網路為例，就有一百萬個 (1000x1000) 個未知參數，不可能用傳統數學方式找出答案。

以一層 hidden layer 為例，網路參數 θ = {w<sub>1</sub>, w<sub>2</sub>, ..., b<sub>1</sub>, b<sub>2</sub>, ...}。學習目標：找出讓 total loss 最小的網路參數 θ

- 隨機挑選一個初始值
- 計算切線斜率 (∂L / ∂w)
- 往左、往右看，找到斜率為零的點 w ← w - η∂L/∂w (η: learning rate，越大表示移動越快)
- 計算 gradient descent ∇L

> [梯度下降法](https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95)（英語：Gradient descent）是一個最佳化算法，通常也稱為最速下降法。基於這樣的觀察：如果實值函數F(x)在點a處可微且有定義，那麼函數F(x)在a點沿著梯度相反的方向-∇F(a) 下降最快。

面對多層網路考慮的問題更複雜。以電玩為例，計算 gradient descent 就像玩世紀帝國一樣，你只能看到鄰近的點，看不到完整的地圖，所以有可能找到局部最佳解，卻達不到全域最佳解。

雖然找到全域最佳解需要運氣跟技巧，不過計算 gradient descent 有一堆工具代勞。

- 使用 backpropagation 的方式計算 ∂L/∂w

### 為何要*深度*學習

基本上，神經網路層數越多，錯誤率越低，效果越好。但有另一個論點，為什麼要用多層，而不是將多層攤平成一層？

從實驗數據來看，瘦高 (Thin + Tall) 網路的效果比肥短 (Fat + Short) 好。為什麼？

|      | 男性 | 女性 |
|------|------|------|
| 短髮 |  多  |  多  |
| 長髮 |  少  |  多  |

以辨識男女長短髮為例，男性長髮的樣本遠少於其他族群，對於**肥短**型網路得到的訓練資料不夠，判斷效果不佳。對於**瘦高**型網路來說，有對男性、女性分類，有對長髮、短髮分類，雖然男性長髮樣本不多，但長髮特徵是由男女長髮樣本共同訓練，所以分辨長短髮的效果不會因為男性長髮樣本少而變差。


### 踏出深度學習第一步

- [TensorFlow](https://www.tensorflow.org/) is an open source software library for numerical computation using data flow graphs.
- [Theano](http://deeplearning.net/software/theano/) is a Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently.

以上兩個工具提供深度學習的功能，功能強大但不好駕馭。

[Keras](http://keras.io/) 包裝 Theano 與 TensorFlow，成為一個簡單易用的 Deep Learning library。

- 輸入：28x28 影像每個像素
- 輸出：0~9，十種可能

以手寫辨識過程為例，Keras 程式碼：

```python
model = Sequential()
```
- 產生一個模型

```python
model.add( Dense( input_dim = 28*28, output_dim = 500) )
model.add( Activation('Sigmoid') )
```
- 接收 28x28 的輸入向量
- 輸出結果到 500 個神經元 (500隨意定)
- 使用 sigmoid 活化函數

```python
model.add( Dense( output_dim = 500) )
model.add( Activation('Sigmoid') )
```
- 下一層不需要宣告輸入數量 (因為 Keras 推測上層的輸出)
- 輸出結果到 500 個神經元 (500隨意定)
- 使用 sigmoid 活化函數

```python
model.add( Dense( output_dim = 10) )
model.add( Activation('Softmax') )
```
- 下一層不需要宣告輸入數量 (因為 Keras 推測上層的輸出)
- 輸出可能有 10 個結果
- 使用 softmax 活化函數

```python
model.compile( loss='mse', optimizer=SGD(lr=0.1), metrics=['accurace'] )
```
- 使用 mean square error 計算錯誤
- 設定 optimizer, metrics

```python
model.fit(x_train, y_train, batch_size=100, nb_epoch=20)
```
- 設定訓練資料
- batch_size: 將數據集分批次做訓練，每個批次大小
- nb_epoch: 全數據集測試次數

```python
score = model.evalute(x_test, y_test)
print('Total loss on Testing Set:', score[0])
print('Accuracy of Testing Set:', score[1])
```
- 輸出評估結果

```python
result = model.predict(x_test)
```
- 對測試資料做預測

更進一步，使用 GPU 加速計算過程
```python
THEANO_FLAGS=device=gpu0 python YourCode.py
```




## 第二講 深度神經網路訓練訣竅

深度學習效率不佳可能有兩個原因，但不能把所有問題都歸因到 overfitting 上：

- 在訓練資料上沒有好的結果 (training data) 
- 在測試資料上沒有好的結果 (testing data) 

 > [wiki] 在統計學中，過適（英語：overfitting，或稱過度擬合）現象是指在調適一個統計模型時，使用過多參數。對比於可取得的資料總量來說，一個荒謬的模型只要足夠複雜，是可以完美地適應資料。過適一般可以識為違反奧卡姆剃刀原則。當可選擇的參數的自由度超過資料所包含資訊內容時，這會導致最後（調適後）模型使用任意的參數，這會減少或破壞模型一般化的能力更甚於適應資料。過適的可能性不只取決於參數個數和資料，也跟模型架構與資料的一致性有關。此外對比於資料中預期的雜訊或錯誤數量，跟模型錯誤的數量也有關。

提高效能的訣竅：

| 訓練資料上 | 測試資料上 |
|----------|----------|
| Choosing proper loss  | Early Stop |
| Mini-batch | Regularization |
| New activation function | Dropout |
| Adaptive learning rate | Newwork Structure |
| Momentum | |

### 選擇合適的損失計算方式 (Choosing proper loss)

| 方式 | 計算公式   |
|----------|---|
| Square Error | Σ (y<sub>i</sub> - ŷ<sub>i</sub>)<sup>2</sup> |
| Cross Entropy | Σ ŷ<sub>i</sub>ln(y<sub>i</sub>) |

[Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)提到，使用 Cross Entropy 計算 total loss 得到的 softmax output 結果，比使用 Square Error 計算 total loss 有更劇烈的反應。也就是說使用 Cross Entropy 比使用 Square Error 在計算 total loss 上有更快的收斂速度。

### 小批量 (Mini-batch)

將 epoch 分成小批量，有助於提高訓練效果

- 隨機初始化網路參數
- 選取第 1 個批次，L' = l<sup>1</sup> + l<sup>31</sup> + ...，更新網路參數
- 選取第 2 個批次，L" = l<sup>2</sup> + l<sup>16</sup> + ...，更新網路參數
- ...
- 直到所有數據都批次處理完成
- 一個 epoch 完成，接著下一個 epoch

> keras 隨機 shuffle 訓練的樣本

```python
model.fit(x.train, y_train, batch_size=100, nb_epoch=20)
```
- 每個小批量 100 個樣本
- 重複 20 次全數據集訓練

問題：每次更新網路參數的 L 都不一樣，這樣不是隨機更換訓練目標？

- 原本的 gradient descent 收斂緩慢
- 使用 mini-batch 的 gradient descent 收斂快速，但呈現跳躍不穩定的狀態

當訓練資料量不大時，使用 mini-batch 與不使用效率差不多。但資料量大，使用 mini-batch 有較好的效率。

### 新的活化函數 (New activation function)

網路架構越深，梯度問題 (Gradient Probleam) 越嚴重，結果是越深的網路效率越差

- 靠近 input 的網路：較小的梯度，學習很慢，結果幾乎隨機
- 遠離 input 的網路：較大的梯度，學習很快，結果幾乎收斂 (based on random?)

問題出在 Sigmoid 這個 Activation Function 會影響輸出的斜率 ∂l/∂w 的計算結果

- 當 output 小：斜率陡峭
- 當 output 大：斜率平坦

> 在 2006 人們使用 RBM pre-training, 在 2015 大家使用 ReLU

#### ReLU (Rectified Linear Unit)
- when z < 0, σ(z) = 0
- when z > 0, σ(z) = z
- when z = 0, σ(z) = 0 or z

理由：

- 計算快速
- 生物理由 (神經元特性?)
- 不同偏移量的無窮 Sigmoid
- **消除梯度問題**

使用 ReLU，在 z < 0 的時候 activation funciton 讓網路架構看起來像移除某些神經元的連結(σ(z) = 0)，變成一個較瘦的線性網路。

其他變形：

- Leaky ReLU
 - when z < 0, σ(z) = 0.01z
 - when z > 0, σ(z) = z
- Parametric ReLU
 - when z < 0, σ(z) = αz
 - when z > 0, σ(z) = z 


### 自動調整的學習速度 (Adaptive learning rate)

要小心設定學習速度，學習速度太快不容易穩定收斂；學習速度太慢 total loss 收斂太慢。

普遍簡單的想法，每次 epoch 逐漸減低學習速度。但無法找到一個全部適用的學習速度，不同的網路參數、不同的學習速度。

#### Adagrad

原來：w ← w - η∂L/∂w
Adagrad：w ← w - η<sub>w</sub>∂L/∂w

η<sub>w</sub> (由w決定的學習速度) = η / √Σ(g<sup>i</sup>)<sup>2</sup>

觀察到的現象：

1. 斜率陡的地方，學習速度慢 (才不會一下跳太遠)
2. 斜率緩的地方，學習速度快 (才不會移動太龜速)


### 動量 (Momentum)

在數學的世界：∂L/∂w = 0 的位置，可能只是區域最佳解 (saddle, local minima)，一旦跑到這裡就很難脫離，繼續往全域最佳解前進。

在物理的世界：像球由高處滾落遇到區域最低點，如果動能足夠就能脫離區域最低點，越過波峰找到全域最佳解。

### Overfitting 怎麼發生的

訓練資料與測試資料不同，學習的目標由訓練資料定義，所學習出來的網路參數適合學習目標但不一定在測試資料上得到好結果。

降低overfitting的方法，更多的訓練資料或產生更多的訓練資料，例如手寫的數字圖片可以轉個小角度或加入一些雜訊變成更多的訓練資料。

### 提早停止訓練 (Early Stop)

提早停止訓練：雖然會得到較差的訓練效能，但可能對測試資料效能更好，因為不會過度 overfitting。

如果 validation loss 不再減小，如何提前停止訓練？
```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])
```

### 規則化 (Regularization)

就像人類腦袋的神經連結，沒有使用就會退化甚至消失，weight decay 找到通常為零的輸入，讓這條連結變數萎縮消失，這樣的好處是能抵抗雜訊出現在不該出現的地方。

> Weight Decay 是一種 regularization

- 原來的：w ← w - η∂L/∂w
- weight decay: w ← (1-λ)w - η∂L/∂w, λ=0.01

雖然 w 會越來越小，但只要不斷有 input 進來刺激，w 就不會衰減為零 (跟人類腦袋運作原理類似，不是嗎？)

### 丟棄 (Dropout)

每次更新網路參數前，每個神經元有 P% 的機率被丟棄，得到一個瘦薄的網路架構，使用這個網路架構訓練。

> 每次 mini-batch 都重新取樣要 dropout 的神經元

訓練時，dropout 的機率 P%, 測試時，權重要乘上 (1-P)%。例如 dropout = 50%, 如果在訓練時 w = 1, 真正測試時 w 要設為 0.5。(因為訓練時缺席率=50%，出席的神經元會加倍努力，所得到的權重也就加倍。到了測試時，所有神經元都出席了，每個神經元的力道要減輕，這樣才不過用力過度)

Dropout 是一種樂團合奏的概念，分別訓練不同的樂器群，最後所有人一起上台表演

- set 1 for network 1
- set 2 for network 2
- ...

### 網路結構 (Newwork Structure)
關於網路架構的設計，CNN 是個很棒的範例 (特別適合處理圖片)，下一講會詳述。



## 第三講 神經網絡的變形

| 縮寫 | 全名 |
|------|------|
| DNN | Deep Neural Network |
| CNN | Convolutional Neural Network |
| RNN | Recurrent Neural Network |

### Convolutional Neural Network (CNN)

為什麼要使用 CNN 處理圖像？

- 輸入的資料可能很大量，例如 800x600 RGB 的圖片
- 藉由考慮影像的特性，fully connected network 可能被簡化嗎？

以鳥類為例，鳥喙可能出現在圖片的任何地方，出現在畫面左上，需要左上有群神經元處理這個資訊，出現在畫面中央，需要中央位置有群經元處理類似的資訊。但鳥喙有特定的影像模式，能由一組神經元專門負責處理嗎？

透過 subsampling 的技術讓影像縮小，較少的資訊只需要較少的網路參數

- 特性一：某些模式 (pattern) 比整體影像小很多
- 特性二：相同的模式出現在不同的位置
- 特性三：subsampling 影像像素不會改變物件

CNN 透過三步驟，建構神經網路

1. Convolution: 將特定模式當作過濾器走訪圖片各位置，得到 feature map
2. Max Pooling: 將 feature map 劃分區域，取出最大值，得到另一張 subsampling 的小圖
3. Flatten: 將小圖攤平，放到 DNN 分析

> 步驟1,2 可重複多次

以 36x32 的圖為例，使用傳統 DNN (dim=4x4x2) 分析需要，36x32=1152 個參數。但使用 CNN 只需要 9x2=18 個參數 (怎麼推導的?)。

Keras CNN 範例 (只修改網路架構):

```python
model = Sequential()
```
```python
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
```
```python
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
```
```python
model.add(Flatten())
```
```python
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
```
```python
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
```

### Recurrent Neural Network (RNN)
- 擁有記憶力的神經網路

範例：語音訂票系統

當用戶說出：“I would like to arrive Taipei on November 2nd.” 要填空兩筆資料

- 目的地：Taipei
- 到達時間：November 2nd

但當用戶說出：“I would leave Taipei on November 2nd.” 要填空兩筆資料

- 出發地：Taipei
- 出發時間：November 2nd

雖然用戶說出的時間跟地點都一樣，但如果神經網路有記憶性，就能區別兩者差異。

RNN 的設計：

- hidden layer 的輸出被儲存在記憶體中，這些輸出可以當作輸入
- 如果文字出現“arrive”，這個訊息會被記憶下來，當出現“Taipei”時，解釋成「到達目的地」的機率就高於「離開出發地」

#### RNN Neural Network

RNN 最常使用的神經元：Long Short-term Memory (LSTM)，有四個輸入、一個輸出

- Input 1: 來自其他網路的輸入
- Input 2: input gate 的訊號控制 (控制輸入影響程度)
- Input 3: forget gate 的訊號控制 (控制歷史資料影響程度)
- Input 4: output gate 的訊號控制 (控制輸出影響程度)
- Ouput: 輸出到其他網路

#### RNN Learning Target

|   | word1 | word2 | word3 | word4 |
|---|-------|-------|-------|-------|
| training sentences | arrive | Taipei | on | November 2nd |
| explained as | other | dest | other | time time |

因為 “arrive” 的關係，“Taipei” 會被解釋為 dest

#### RNN Learn!

Backpropagation through time (BPTT) 讓 RNN Learning 在實務上很難實現。

- total loss 不會隨著 epoch 收斂 (呈現跳動)
- error surface 可能很平坦，也可能很陡峭
- w 參數難以設定
  - 當w > 1，記憶會放大 gradient descent，需要較慢的學習速率
  - 當w < 1，記憶會縮小 gradient descent，需要較快的學習速率

| w | y<sup>1000</sup> |
|---|------------------|
| 1 | 1 |
| 1.01 | ≈ 20000 | 
| 0.99 | ≈ 0 |
| 0.01 | ≈ 0 |

有用的技巧：

- advanced momentum method: Nesterov’s Accelerated Gradient (NAG)
- Long Short-term Memory (LSTM) - memory and input are added
- Gated Recurrent Unit (GRU) - input gate + output gate = 1
- Clockwise RNN
- Structurally Convolutioned Recurrent Network (SCRN)

#### RNN 的應用
- Many to one：輸入一串文字，輸出字詞 (positive, neutral, negative)
  - 輸入：看了這部電影覺得很高興...
  - 輸出：正雷
- Many to many：輸入一串文字，得到較短的一段文字
  - 輸入：好好好棒棒棒
  - 輸出：好棒棒
- Many to Many (No Limitation)：輸入、輸出各是一段不同長度的文字
  - 輸入：machine learning
  - 輸出：機器學習
- One to Many：輸入一個圖片，輸出一串文字
  - 輸入：圖(一對母子在草地遊戲)
  - 輸出：a woman is ....




## 第四講 下一波技術

| 跟網路結構相關的 | 跟學習目標有關的 |
|------------------|------------------|
| Ultra Deep Network | Reinforcement Learning |
| Attention Model | Towards Unsupervised Learning |

### 超級深度網路 (Ultra Deep Network)

自古以來大家都喜歡蓋高樓，從埃及金字塔、到台北101、再到杜拜塔，一棟比一棟高。

深度學習也一樣，從 2012 年 AlexNet 8 層網路的 16.4% 錯誤率，到 2014 年 VGG 19 層 7.3% 的錯誤率，到 2014 年 GoogleNet 22 層的 6.7% 錯誤率，最後 2015 年 Residual Net 152 層 3.57% 的錯誤率，一路往超級深的神經網路前進。但這樣的網路模型不會 overfitting 嗎？計算上能做得到嗎？

Ultra Deep Network 採用 Residual Network 或 Hightway Network 架構，透過複製輸出到下下層的輸入，直接忽略某些網路層，能夠自動調整網路架構的深度。

### 注意模型 (Attention Model)

情境：

- 已知：腦子裡有今天學的、今天中午吃什麼、國中二年級的夏天
- 問題：What's deep learning?
- 反應：針對問題，**組織**已知的知識，回答正確的問題


#### 閱讀理解 (Reading Comprehension)

Query → DNN/RNN 
→ Reading Head Control → Each sentence becomes a vector. 
(→ Reading Head Control → Each sentence becomes a vector.) ...
→ Answer

| Story            | Hop 1    | Hpp 2    | Hop 3    |
|------------------|----------|----------|----------|
| Brian is a frog. | 0.00     | **0.98** | 0.00     |
| Lili is gray.    | 0.07     | 0.00     | 0.00     |
| Brian is yellow. | 0.07     | 0.00     | **1.00** |
| Julius is green. | 0.06     | 0.00     | 0.00     |
| Greg is a frog.  | **0.76** | 0.02     | 0.00     |

- Question: What color is Greg?
- Answer: yellow
- Prediction: yellow

#### 視覺問答 (Visual Question Answering)

範例ㄧ：

- 照片：一張女性用香蕉裝扮鬍子
- 問題：鬍子是用什麼做的
- 過程：先找出人臉，在找出鬍子的位置，然後判別出這個位置的東西

範例二：

- 照片：一個籃子裝著一隻貓與一個玩偶，貓下方有個紅色的方塊
- 問題：貓下面有個紅色的方塊嗎？
- 過程：找出貓的位置，判斷貓臉下方的物體是不是紅色方塊


#### 機器翻譯 (Machine Translation)

- 記憶：一種語言 (machine learning)
- 輸出：另一種語言 (機器學習)
- 做法：使用 RNN 關聯字詞

#### 語音識別 (Speech Recognition)

- 記憶：一段語音 (...)
- 輸出：文字稿 (how much wo...)

> 我覺得這個例子講得不是很清楚

#### 頭條產生 (Headline Generation)

- 輸入：一段新聞稿 (台北市今天蔬菜到貨量較昨天略增,市況逢星期假日,需求增加,交易活絡,各類交易行情漲多跌少, 每公斤批發價上漲為約新台幣十四點八元; ......)
- 輸出：台北市蔬菜批發價格行情上揚
- 實際：台北市蔬菜批發行情上揚水果下挫

#### 標題產生 (Caption Generation)

範例ㄧ：

- 圖片：一支金色的獅子雕像，坐在大樓外面的石台上
- 輸出：A cat is sitting on a bad in a room.

範例二：

- 圖片：台北 101 亮燈的夜拍照片
- 輸出：A clock tower with a clock on it.

### 強化學習 (Reinforcement Learning)

學校裡的機器學習是監督式學習 (supervised learning):輸入、輸出範圍固定。但真實世界裡的機器學習，沒有明確的輸入。

例如，訂票系統的語音助理可以使用 RNN 根據客戶的語音輸入，幫客戶訂到飛機機票。但真正的挑戰，不是每個動作都有回饋，或是語音助理可能影響客戶未來的行為。

範例一：機器無從判斷第一句的問候，不是造成客戶生氣的原因

- 機器：Hello
- 客戶：...
- 機器：...
- 客戶：...
- ...
- 客戶：#^$@$#@ (生氣)

範例二：機器請客戶重複描述，客人可能輸出更多資訊，影響機器將觀察到的未來

#### 深度強化學習 for 圍棋

挑戰一：不是每個動作都有回饋

- 初手天元 → ... 下了好幾百手 ... → Win!
- 無法判別初手天元是好是壞

挑戰二：機器可以影響觀察到的未來

- 蒙地卡羅 (Monte Carlo) 樹狀搜尋很適合用來下圍棋，但無法通吃所有問題。

### 朝向非監督式學習 (Towards Unsupervised Learning)

#### 自動編碼器 (Auto-encoder)

- 使用 28x28 的圖片表示數字
- 不是每張 28x28 的圖片都能判斷出 0~9 的數字
- 能用更精簡的方式表達文字的圖片嗎?

計算過程：x → encode → a → decode → y

- 輸入圖片 x
- 輸出圖片 y
- 找出 (x-y)<sup>2</sup> 最小值的網路參數

過程中加入雜訊訓練抗噪能力，也可以加深神經網路的深度。最後得到的模型，雖然沒有告訴機器圖片代表的意義，但比起原先的 DNN 更能精準地將數字分群。

#### 字詞向量 (Word Vector)

藉由機器閱讀，找出文章中字詞間的距離，藉此推測字詞的關係。例如：

- Rome : Italy = Berlin : ?
- 計算 V(Berlin) - V(Rome) + V(Italy)，找出最接近 V(w) 的字詞 w (答案當然是 Germany)
