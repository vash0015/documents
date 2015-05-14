
# タスクリスト

[トップページ]

* [序論]
    - [x] 機械学習：問題設定
    - [x] サンプルデータセットの読込
    - [x] 学習と予測
    - [ ] Model persistence
* [科学データ処理のための統計学習チュートリアル]
    - [ ] Statistical learning: the setting and the estimator object in scikit-learn
    - [ ] Supervised learning: predicting an output variable from high-dimensional observations
    - [ ] Model selection: choosing estimators and their parameters
    - [ ] Unsupervised learning: seeking representations of the data
    - [ ] Putting it all together
    - [ ] Finding help
* Working With Text Data
    * Tutorial setup
    * Loading the 20 newsgroups dataset
    * Extracting features from text files
    * Training a classifier
    * Building a pipeline
    * Evaluation of the performance on the test set
    * Parameter tuning using grid search
    * Exercise 1: Language identification
    * Exercise 2: Sentiment Analysis on movie reviews
    * Exercise 3: CLI text classification utility
    * Where to from here


# 序論

## 機械学習：問題設定

機械学習は下記の大きなカテゴリに分類できる。

* 教師あり学習
    * 分類
    * 回帰
* 教師なし学習
    * クラスタリング
    * 密度推定
    * 可視化のための次元削減

機械学習では、データセットをトレーニングセットとテストセットに分け、学習と評価を行う。


## サンプルデータセットの読込

scikit-learnにはいくつかの標準的なデータセットが付属している。

* for classification
    * [iris dataset]( http://en.wikipedia.org/wiki/Iris_flower_data_set)
    * [digits dataset]( http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits)
* for regression
    * [boston house prices dataset]( http://archive.ics.uci.edu/ml/datasets/Housing)

シェルからPythonインタプリタに下記のコマンドを打ち込むことで、データセットの読込を行うことができる。

```py
$ python
>>> from sklearn import datasets
>>> iris = datasets.load_iris()
>>> digits = datasets.load_digits()
```


## 学習と予測

digit データセットの場合「画像を与えると、どの数字を意味するのかを予測する」というタスクに使う。
10クラス(0から9までの数字)の分類に [estimator] を用い、未知のサンプルがどのクラスに属するのか予測することが出来る。

scikit-learn では estimator for classification には `fit(X, y)` と `predict(T)` が実装されている。

estmator の例は `sklearn.svm.SVC` クラスであるが、これは [support vector classification] の実装である。estimator のコンストラクタはモデルのパラメータを引数として持つ、しかし当面 estimator をブラックボックスとして扱う。

```py
>>> from sklearn import svm
>>> clf = svm.SVC(gamma=0.001, C=100.)
```

** モデルパラメータの選択 **

> この例の中で我々は gamma を手動で設定した。これはツールを使って自動で良好な値を見つけることも可能である。その場合は [グリッドサーチ] と [交差検証] を行うツールを使用する。

`clf` は estimator のインスタンスであり、分類器である。モデルにフィットさせたければ、モデルから学習しなければならない。これは `fit()` に我々のトレーニングセットを通すことにより実現できます。
As (～として取り扱う) a training set, let (～すると宣言する) us use all the images of our dataset apart (別々にする) from the last one.

我々はこのトレーニングセットをPython文法で `[:-1]` と指定して使用する、これは `digits.data` の 最後のエントリを除いた全てを指す：

```py
>> clf.fit(digits.data[:-1], digits.target[:-1])  
SVC(
    C=100.0, cache_size=200, class_weight=None, coef0=0.0,
    degree=3, gamma=0.001, kernel='rbf', max_iter=-1,
    probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False
)
```

Now you can predict new values, in particular (特に), we can ask to the classifier (分類器に問い合わせる) what is the digit of our last image in the digits dataset, which we have not used to train (教えることをしていない) the classifier:

```py
>>> clf.predict(digits.data[-1])
array([8])
```
The corresponding (対応する) image is the following (以下に):

<div align="center"><img src=
"http://scikit-learn.org/stable/_images/plot_digits_last_image_0011.png"
width="200"></div>

As (～の立場で) you can see, it is a challenging task: the images are of poor resolution. Do you agree with the classifier?

A complete example of this classification problem is available as an example that you can run and study: [Recognizing hand-written digits.]( http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#example-classification-plot-digits-classification-py)

## Model persistence

It is possible to save a model in the scikit by using Python’s built-in persistence model, namely (～と呼ばれる) [pickle](http://docs.python.org/library/pickle.html):

```py
>>> from sklearn import svm
>>> from sklearn import datasets
>>> clf = svm.SVC()
>>> iris = datasets.load_iris()
>>> X, y = iris.data, iris.target
>>> clf.fit(X, y)  
SVC(
    C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    degree=3, gamma=0.0, kernel='rbf', max_iter=-1,
    probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False
)

>>> import pickle
>>> s = pickle.dumps(clf)
>>> clf2 = pickle.loads(s)
>>> clf2.predict(X[0])
array([0])
>>> y[0]
0
```

In the specific (特定の) case of the scikit, it may be more interesting to use joblib’s replacement of pickle ( `joblib.dump` & `joblib.load` ), which is more efficient on big data, but can only pickle to the disk and not to a string:

```py
>>> from sklearn.externals import joblib
>>> joblib.dump(clf, 'filename.pkl')
```

> ** Note: ** joblib.dump returns a list of filenames. Each individual numpy array contained in the clf object is serialized as a separate file on the filesystem. All files are required in the same folder when reloading the model with joblib.load.

Note that pickle has some security and maintainability issues. Please refer to section Model persistence for more detailed information about model persistence with scikit-learn


# 科学データ処理のための統計学習チュートリアル

** Statistical learning **

Machine learning is a technique with a growing importance (重要性が増している), as (～の視点における) the size of the datasets experimental sciences (実験科学) are facing (向かっている) is rapidly growing (急速に増大している).
Problems it tackles range from (変動する) building a prediction function linking different observations (観点), to classifying observations, or learning the structure in an unlabeled dataset.

This tutorial will explore (調査する) statistical learning (統計学習), the use of machine learning techniques with the goal of statistical inference (統計的推定): drawing conclusions on the data at hand.

Scikit-learn is a Python module integrating classic machine learning algorithms in the tightly-knit (密接かつしっかりと結合された) world of scientific Python packages (NumPy, SciPy, matplotlib).


## Statistical learning: the setting and the estimator object in scikit-learn





## Supervised learning: predicting an output variable from high-dimensional observations
## Model selection: choosing estimators and their parameters
## Unsupervised learning: seeking representations of the data
## Putting it all together
## Finding help


</br>
</br>
</br>
</br>
</br>




[トップページ]: http://scikit-learn.org/stable/tutorial/index.html

[序論]: http://scikit-learn.org/stable/tutorial/basic/tutorial.html

[科学データ処理のための統計学習チュートリアル]: http://scikit-learn.org/stable/tutorial/statistical_inference/index.html

[estimator]: http://en.wikipedia.org/wiki/Estimator

[support vector classification]: http://en.wikipedia.org/wiki/Support_vector_machine

[グリッドサーチ]: http://scikit-learn.org/stable/modules/grid_search.html#grid-search

[交差検証]: http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
