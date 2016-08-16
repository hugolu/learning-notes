# 從資料到知識：從零開始的資料探勘

- [講座連結](http://datasci.tw/intro-mining.html)
- [簡報連結](http://www.slideshare.net/tw_dsconf/ss-64145214)

## 從資料到目標，從目標到知識 (Data Mining: From Data to Task to Knowledge)

### 簡介資料探勘 (Introduction to Data Mining)
#### 什麼是資料探勘
- 從資料抽取有用的資訊
- 將資料轉換為可理解的結構
- 資料庫知識發掘(knownledge discovery in database, KDD)的分析步驟

#### 真實世界的應用
- Googole Search Engine
- Spam Filtering
- Advertising
- Recommender Systems
- Facebook Newsfeed

#### 但真實世界的資料很雜亂
- 沒有組織化
- 大部份是伺服器的 logs
- 問題可能難以直接解決

#### 資料探勘與其他技術的重疊
- Database - Large-scale Data 
- Machine Learning - Learning Model
- CS Theory - Algorithms

### 資料探勘中的任務與模型 (Tasks and Models in Data Mining)

![](pictures/data-task-knowledge.png)

#### 資料探勘中的模型
建立模型以解決問題，模型的特性
- 有效性 - 有明確的新資訊
- 新穎性 - 得到原來無法明顯觀察到的結果
- 有用處 - 能藉此做出行動 (預期股市會漲：可產生行動，他是個禿子：無法產生行動)
- 可理解 - 人類應該可以解釋模式 (pattern)

#### 資料探勘中的任務
- 解決問題前應該明確的表達問題

兩種任務
- 預測型
  - 處理未知的值，例如分類
  - 分類(Classification)、迴歸(Regression)、排名(Ranking)...
- 描述型
  - 找到模式與描述資料，例如社群分群
  - 分群(Clustering)、摘要(Summarization)、關聯規則學習(Association Rule Learning)

#### 分類(Classification)
- 概括**已知**的結構，應用在**未知**的資料
- 學習 classifier 模型分類新資料
- 例如，給訂目前的社交網路，預測兩個節點是否會在未來連結

#### 迴歸(Regression)
- 找出能用最小誤差描述一群資料的函數
- 輸出可能是數值
- 例如，預測搜尋引擎廣告的點擊率(click-through rate, CRT)

#### 排名(Ranking)
- 產生項目的名次排列
- 位置越高的項目越重要
- 例如，搜尋引擎的結果 (越上面的連結越有相關性)

#### 分群(Clustering)
- 發現資料的群聚(groups)與結構(structures)
- 在不知道資料結構的情況下，學習分群
- 例如，根據標誌位置的照片，找出興趣點

> Wiki: 興趣點（point of interest, POI）乃是電子地圖上的某個地標、景點，用以標示出該地所代表的政府部門、各行各業之商業機構（加油站、百貨公司、超市、餐廳、酒店、便利商店、醫院等）、旅遊景點（公園、公共廁所等）、古蹟名勝、交通設施（各式車站、停車場、超速照相機、速限標示）等處所。

#### 摘要(Summarization)
- 呈現更精簡的資訊 (compact representation)
- 文本類型 - 文章摘要
- 一般資訊 - 資訊視覺化
- 例如，將網頁內容濃縮成精簡但完整的片段 (snippet)

#### 關聯規則學習(Association Rule Learning)
- 發掘變數間的關係
- 發現的關係可能對應用或分析有幫助
- 例如，啤酒與尿布、偵測網路上的隱私權漏洞

#### 合併多個任務
- 分解問題為多個任務
- 例如，使用 Twitter 偵測事件
  - 先分類：將 tweets 分類成各種事件
  - 後迴歸：某 tweets 是在說某事件嗎？ (評估可能性)

### 資料探勘中的機器學習 (Machine Learning in Data Mining)

#### 從機器學習的觀點
- 利用演算法來學習模型
- 資訊編碼為特徵向量 (feature vectors)
- 特徵與模型的選取很重要

![](pictures/machine-learning.png)

#### 機器學習模型的種類
- 不同演算法適用於不同資料
- 依賴於資料本身與應用場景

#### 監督式學習 vs. 非監督式學習
- 監督式學習 (Supervised Learning)
  - with labeled data: 使用有標籤的資料學習，例如分類
- 非監督式學習 (Unsupervised Learning)
  - without labeled data: 使用無標籤的資料學習，例如分群

#### 半監督式學習 (Semi-supervised Learning)
- 使用有標籤與無標簽的資料學習
- 主要想法：類似的資料有類似的標籤
- 例如，自動幫未上標籤的資料上標籤

#### 增強學習 (Reinforcement Learning)
- 沒有明確的標籤，但能從環境中隱約觀察到
- 從環境隱約回饋中學習
- 例如，AlphaGO 每步驟沒有明確的標籤，只有「輸贏」的隱約回饋

> [机器学习算法之旅](http://blog.jobbole.com/60809/) 增强学习：输入数据可以刺激模型并且使模型做出反应。反馈不仅从监督学习的学习过程中得到，还从环境中的奖励或惩罚中得到。问题例子是机器人控制，算法例子包括Q-learning以及Temporal difference learning。

### 創新：從資料到任務到知識 (Innovation: from Data to Task to Knowledge)

#### 兩個關鍵基礎
- 資料：資訊來源
- 任務：要解決的問題

![](pictures/data-task-knowledge.png)

***Then the model can solve the task with data and produce knowledge!***

#### 資料在哪裡？到處都是！
- 社群服務 (Facebook, Twitter, ...)
- 網路 (sockal networks, road networks, ...)
- 感應器 (time-series, ...)
- 影像 (photos, fMRI, ...)
- 文本 (news, documents)
- 網頁 (forums, websites, ...)
- 公開資料 (populations, ubike logs, ...)
- 商業資料 (transactions, customers, ...)
- 更多

#### 資料挖掘如何革新？

#### 資料驅動
- 從特定資料引入任務：可以對資料做什麼？
  - 對於空氣品質：推測目前品質、預測未來品質、監測站選擇 
  - 對於社會事件：找到潛在客戶、推薦事件、用戶影響排名

#### 問題驅動
- 針對特定任務搜集相關資料：什麼資料有助於解決問題？
  - 音樂推薦：聆聽紀錄、音樂標籤、社群網路
  - 交通評估：氣象、過去的交通、社群網路

### 資料探勘工具 (Tools for Data Mining)

#### 工具
- 圖形化介面
  - Weka
- 命行式介面
  - LIBSVM & LIBLINEAR
  - RankLib
  - Mahout
- 函式庫
  - scikit-learn
  - XGBoost

#### Weka
- http://www.cs.waikato.ac.nz/ml/weka/
- 功能：分類、回歸、分群、關聯規則
- 優點：眾多模型、友善介面、容易視覺化
- 缺點：慢、參數調整、令人混亂的格式

#### scikit-learn
- http://scikit-learn.org/
- 功能：分類、迴歸、分群、降維
- 優點：快速、彈性、眾多模式、參數調整
- 缺點：參數調整、自行處理資料、沒有圖形化介面、需要寫程式

#### LIBSVM & LIBLINEAR
- http://www.csie.ntu.edu.tw/~cjlin/libsvm/
- 功能：分類、迴歸
- 優點：高效能、純粹命令式介面、參數調整、固定資料格式
- 缺點：沒有圖形化介面、只有 SVM

#### RankLib
- http://www.lemurproject.org/
- 功能：排名
- 優點：眾多模式、純粹命令式介面、參數調整、固定資料格式
- 缺點：沒有圖形化介面、只有排名功能

## 從資料中發現蛛絲馬跡：特徵抽取與選擇 (Clues in Data: Features Extraction and Selection)

### 資料探勘中的特徵 (Features in Data Mining)

#### 真實世界的資料又髒又亂

#### 產品評估：我喜歡 iPhone 嗎？

“I bought an iPhone a few days ago. It is such a `nice phone`. The touch screen is `really cool`. The voice quality is `clear` too. It is much `better than my old Blackberry`, which was a `terrible phone` and so `difficult to type` with its `tiny keys`. `However`, my mother was `mad` with me as I did not tell her before I bought the phone. She also thought the phone was `too expensive`, ...” — An example from [Feldman, IJCAI ’13]

- 文章中參雜許多正面、負面的詞

#### 類似的照片
- 拉布拉多 還是 炸雞？ (棕色、團狀)
- 牧羊犬 還是 拖把？ (白色、長毛)

#### 人類如何區別照片
- 小狗：有眼睛、鼻子，很可愛
- 炸雞：可口、一塊塊、有翅膀

觀察到得以做出決定的重要特性 (properties)

#### 特徵 (fatures): 特性的描述
![](pictures/features.png)

#### 一般化的描述
- 原始資料無法直接比較
- Features are same-length vectors with meaningful information.

### 特徵抽取 (Feature Extraction)

#### 如何抽取特徵
不同資料需要不同特徵
- 類別特徵 (Categorical features)
- 統計特徵 (Statistical features)
- 文本特徵 (Text features)
- 影像特徵 (Image features)
- 訊號特徵 (Signal features)

#### 類別特徵 (Categorical features)
- 有些資訊屬於種類，而非數值
  - 血型、天氣、出生地
- 從 n 種類別特徵擴充為 n 個二元數值特徵

血型 | A | B | AB | O
-----|---|---|----|---
Type A  | 1 | 0 | 0 | 0
Type B  | 0 | 1 | 0 | 0
Type AB | 0 | 0 | 1 | 0
Type O  | 0 | 0 | 0 | 1

#### 統計特徵 (Statistical features)
- 將數值編碼成特徵
  - 國民所得、一週交通
- 運用統計量測來呈現資料特性
  - 最大值、最小值、平均值、中位數、模數、標準差

#### 平均 vs 中位數
- outlier 嚴重影響統計結果
- 例如，國民所得資料：22k, 22k, 33k, 1000k
  - 平均值 = 269250 (= `(22k + 22k + 33k + 1000k) / 4`)
  - 中位數 = 27500 (= `(22000 + 33000) / 2`)
  - 哪個更有代表性？

#### 統計特徵適合分組資料 (grouped data)
- 不規律資料：1, 1, 1000000, 1000000
- 資料分成 n 組：可以包含 n 個統計特徵

傳統分組標準
- 時間 (年、月、日)
- 地區 (國家、州)
- 人口特徵 (年齡、性別)

#### 文本特徵 (Text features)
文字探勘很困難
- 高度變化與不確定性
  - 文本可以用不同語言、長度、標題編寫
- 字詞分割
  - 字母語言 (英文、法文): data science -> data + science
  - 非字母語言 (中文、日文): 全台大停電 -> 全台 + 大 + 停電 or 全 + 台大 + 停電
- 文法方式
  - 詞幹提取(stemming): cats -> cat; image, imaging -> imag
  - 詞形還原(lemmatization): fly, flies, flied -> fly; am, is, are -> be

#### 文法方式: Part-of-Speech (POS)
不同位置(詞性)呈現不同意義
- exploit: (N) 功績, (V) 利用
- 落漆: (Adj) 遜掉了, (V) 牆壁油漆脫落

![](pictures/NLTK.png)

工具
- [NLTK](http://www.nltk.org/): Toolkit for English Natural Language Processing
- [jieba (結巴)](https://github.com/fxsjy/jieba): Toolkit for Chinese segmentation and tagging

#### 字詞頻率
- Bag-of-Words (BoW): A simplifying representation
  - 越頻繁的字詞越重要？
- Zipf's Law
  - 頻率越高的字可能越普通
- TF-IDF (Term Frequence x Inverse Document Frequence)
  - IDF(w) = log (|D| / df(w)), |D|: 所有文章數, df(w): 出現 w 的文章數
  - 降低經常出現字詞的重要性

#### Stopwords (無意義的字)
- 高頻但無意義
  - 英文：a, the, and, to, be, at, ...
  - 中文：的、了、嘛、吧、但是

#### 連續字詞
- 有些字詞合起來才有意義
  - 例如，Micheal, Jordan => Micheal Jordan
- Character-level n-gram: 得到字的模式
- Word-level n-gram: 得到片語的模式

#### Word2Vec
- 計算文字空間中兩個字的距離
  - 男人之於女人，國王之於王后

#### 影像特徵 (Image features)
影像由 pixel 所構成，特徵處理方式有
- Resizing: 轉換成相同大小
- pixel 位置不重要
- Scale-invariant Feature Transform (SIFT): 抽取區域關鍵 (local keypoints) 作為特徵，不隨大小(scaling)、旋轉(rotation)、轉換(translation) 而改變

#### Bag of Visual Words
- 應用 bag-of-word 的概念
  - TF-IDF 
  - keypoint clustering
  - visual-word vectors

#### CNN (Convolution Neural Network
- 來自多層CNN的特徵
  - 預先使用其他資料訓練
  - hidden layer = 特徵？

工具
- OpenCV
  - 內建 SIFT
  - 可連接 CNN 深度學習函式庫
- 深度學習函式庫
  - Tensorflow, Theano, Caffe

#### 訊號特徵 (Signal features)
來源
- 感應器資料
- 音樂
- 股票
- 動作
- 影像與影片

問題
- 多數為可變長度的時間序列資料 (time series data)

處理方式
- Sliding Window 從感興趣的位置抽取特定長度的資料
- Fourier Transform 從另一個 domain 觀察資料：將訊號分解成頻率

### 特徵與性能 (Features and Performance)
如何決定特徵的好壞？

不好的特徵會導致災難
- 模式會試著符合訓練資料
- 但模式可能無法符合其他(測試)資料

好的特徵可以提高效能
- 選擇較高效能的特徵

#### Evaluation for supervised models
準備資料
- 訓練資料 (training data) - 學習模式
- 評估資料 (evaluation data) - 調整模式
- 測試資料 (testing data) - 判斷效能

#### 分類的評估方式

> 我覺得投影片的表有問題，以下是我自己的想法

        | Predict=1 (Positive) | Predict=0 (Negative)
--------|----------------------|----------------------
Truth=1 | TP                   | FN
Truth=0 | FP                   | TN

縮寫| 全名            | 說明                      | 預測結果
----|-----------------|---------------------------|----------
TP  | True Positive   | 真陽性：預測為真，事實為真| 成功
FP  | False Positive  | 偽陽性：預測為真，事實為假| 失敗
TN  | True Negative   | 真陰性：預測為假，事實為假| 成功
FN  | False Negative  | 偽陰性：預測為假，事實為真| 失敗

![](https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/422px-Precisionrecall.svg.png)

評估方式  | 公式 | 說明
----------|------|------
Accuracy  | `(TP + TN) / (TP + TN + FP + FN)` | The ratio of correct predictions 
Precision | `TP / (TP + FP)` | The ratio of correct predictions among positive prediction
Recall    | `TP / (TP + FN)` | The ratio of correct predictions among all such-class instances
F1-Score  | `2∙P∙R/(P+R)` | Consider precision and recall at the same time 

#### 遞迴的評估方式
- MAE (Mean Absoluate Error): 預測距離真實結果有多接近
- RMSE (Root-mean-square error): 放大嚴重的錯誤

#### 排序的評估方式
- Binary Relevance Ranking: 針對兩個類別 (有關連與無關聯)
  - Mean Average Precision (MAP)
  - Mean Reciprocal Rank (MRR)
  - Area under the Curve (AUC)
  - Precision at k (P@k)
- Graded Relevance Ranking: 相關可有多個級別 (分數1~5)
  - Normalized Discounted Cumulated Gain (NDCG)

#### 人工標籤的評估
- 有些 ground truth (参考标准) 不可得
- 要得到 ground truth 需要人打標籤
- 每個標籤至少需要兩個人確認
- 上標籤的過程需要判斷
 
##### Cohen’s Kappa Coefficient (k)
- 評估兩個人做決定一致性
- P(a) 兩個人有共識


### 特徵的選擇 (Feature Selection)
### 特徵的縮減 (Feature Reduction)

## 發現資料中的小團體：分群與其應用 (Small Circles in Data: Clustering and its Applications)

### 介紹分群 (Introduction to Clustering)
### 階級分群 (Hierarchical Clustering)
### 切割分群 (Partitional Clustering)
### 分群的應用 (Applications of Clustering)

## 沒有特徵該怎麼辦？從推薦系統談起 (No Features? Starting from Recommender Systems)

### 介紹推薦系統 (Introduction to Recommender System)
### 根據內容過濾 (Content-based Filtering)
### 協同式過慮 (Collaborative Filtering)
### 潛在因素模型 (Latent Factor Models)
### 潛在因素模型的變化 (Variations of Latent Factor Models)
