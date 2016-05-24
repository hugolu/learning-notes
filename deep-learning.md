# 一天搞懂深度學習 心得筆記

[一天搞懂深度學習](http://datasci.tw/event/deep_learning_one_day/)由李宏毅助理教授濃縮在台大上一個學期 [Machine Learning and having it deep and structured](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLSD15_2.html) 的課程精華，授課淺顯易懂，[投影片](http://www.slideshare.net/tw_dsconf/ss-62245351)內容精彩豐富。以下是重點整理與學習心得：

## 第一講 介紹深入學習

### 深度學習三步驟

- 建立網路結構
- 學習目標
- 開始學習

人類腦中有許多神經元與突觸，經過學習認知，有些神經元的連結被強化，有些則退化或消失。神經網路也與人腦類似，透過學習已知事物建立認知模型，然後面對未知做出判斷。

#### 步驟一 建立網路結構

##### Neuron (神經元)

神經元是神經網路的基本元件，用以下公式表示：

z = a<sub>1</sub>w<sub>1</sub> + ... + a<sub>k</sub>w<sub>k</sub> + ... + a<sub>K</sub>w<sub>K</sub> + b

a = σ(z)

- w: weight (權重)
- b: bias (偏移量)
- σ(z): activation function (活化函數)

這是一個簡單的多元一次方程式，一組輸入 (a<sub>1</sub>, ..., a<sub>k</sub>, ..., a<sub>K</sub>)，經過一組權重 (w<sub>1</sub>, ..., w<sub>k</sub>, ..., w<sub>K</sub>) 調整後，加上偏移量 b，得到中間值 z，再通過活化函式 σ(z)，得到輸出 a

σ(z) 最初採用 Sigmoid function (= 1 / 1 + e<sup>-z</sup>)，當輸入值很小得到 0，當輸入值很大得到 1，輸入值為零得到 0.5。使用 Sigmoid function 把輸出值限定在一個可控制的範圍，才不會經過多層網路放大差異。

##### Neuron Network (神經網路)
多個神經元組成神經網路，每個神經元有不同的權重與偏移量，權重與偏移量將透過第三步驟求得。

##### Fully Connected Feedforward Network (完全連結前饋網路)
上層所有元件與下層所有元間都有連結，如果上層有 N 個神經元，下層有 M 個神經元，那麼就有 NxM 個連結，表示權重有 NxM 個。

每層輸入以向量 x 表示，權重以矩陣 W 表示，偏移量以向量 b 表示，經過活化函數 σ() 得到輸出。

- 第一層: σ(W<sup>1</sup>x + b<sup>1</sup>)
- 第二層: σ(W<sup>2</sup>x + b<sup>2</sup>)
- ...
- 第Ｌ層: σ(W<sup>L</sup>x + b<sup>L</sup>)

輸入層輸入 (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>N</sub>) 經過多個隱藏成層作用，每個隱藏層中包含多個神經元，最後透過輸出層得到結果 (y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>N</sub>)。

y = f(x) = σ(W<sup>L</sup> ... σ(W<sup>2</sup>σ(W<sup>1</sup>x + b<sup>1</sup>) + b<sup>2</sup>) ... + b<sup>L</sup>)

##### Output Layer (輸出層)
使用 softmax layer 作為輸出層，讓輸出結果呈現機率分佈，所有可能加總為 1。

#### 步驟二 學習目標

基本上，神經網路層數越多，錯誤率越低，效果越好。但有另一個論點，為什麼要用多層，而不是將多層攤平成一層？

從實驗數據來看，瘦高 (thin+tall) 網路的效果比肥短 (fat-short) 好。為什麼？

|      | 男性 | 女性 |
|------|------|------|
| 短髮 |  多  |  多  |
| 長髮 |  少  |  多  |

以辨識男女長短髮為例，男性長髮的樣本遠少於其他族群，對於**肥短**型網路得到的訓練資料不夠，判斷效果不佳。對於**瘦高**型網路來說，有對男性、女性分類，有對長髮、短髮分類，雖然男性長髮樣本不多，但長髮特徵是由男女長髮樣本共同訓練，所以分辨長短髮的效果不會因為男性長髮樣本少而變差。

#### 步驟三 開始學習


### 為何要*深度*學習

### 踏出深度學習第一步





## 第二講 DNN 訓練訣竅
DNN: Deep Neural Network (深度神經網路)
提高效能的訣竅：

| 訓練資料上 | 測試資料上 |
|----------|----------|
| Choosing proper lose  | Early Stop |
| Mini-batch | Regularization |
| New activation function | Dropout |
| Adaptive learning rate | Newwork Structure |
| Momentum | |

## 第三講 神經網絡的變形

| 縮寫 | 全名 |
|------|------|
| DNN | Deep Neural Network |
| CNN | Convolutional Neural Network |
| RNN | Recurrent Neural Network |

### Convolutional Neural Network (CNN)

### Recurrent Neural Network (RNN)

## 第四講 下一波技術

| 跟網路結構相關的 | 跟學習目標有關的 |
|------------------|------------------|
| Ultra Deep Network | Reinforcement Learning |
| Attention Model | Towards Unsupervised Learning |

### Ultra Deep Network
### Attention Model
### Reinforcement Learning
### Towards Unsupervised Learning
