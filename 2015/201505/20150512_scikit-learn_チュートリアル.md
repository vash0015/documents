# タスクリスト

[トップページ]

* [序論]
    - [x] .[機械学習：問題設定]
    - [x] .[サンプルデータセットの読込]
    - [x] .[学習と予測]
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

この例の中で我々は gamma を手動で設定した。これはツールを使って自動で良好な値を見つけることも可能である。その場合は[グリッドサーチ]と[交差検証]を行うツールを使用する。

`clf` は estimator のインスタンスであり、分類器である。モデルにフィットさせたければ、モデルから学習しなければならない。



<div align="center"><img src=
"http://scikit-learn.org/stable/_images/plot_digits_last_image_0011.png"
width="200"></div>















</br>
</br>
</br>
</br>
</br>




[トップページ]: http://scikit-learn.org/stable/tutorial/index.html

[序論]: http://scikit-learn.org/stable/tutorial/basic/tutorial.html

[機械学習：問題設定]: http://scikit-learn.org/stable/tutorial/basic/tutorial.html#machine-learning-the-problem-setting

[サンプルデータセットの読込]: http://scikit-learn.org/stable/tutorial/basic/tutorial.html#loading-an-example-dataset

[学習と予測]: http://scikit-learn.org/stable/tutorial/basic/tutorial.html#model-persistence

[科学データ処理のための統計学習チュートリアル]: http://scikit-learn.org/stable/tutorial/statistical_inference/index.html

[estimator]: http://en.wikipedia.org/wiki/Estimator

[support vector classification]: http://en.wikipedia.org/wiki/Support_vector_machine

[グリッドサーチ]: http://scikit-learn.org/stable/modules/grid_search.html#grid-search

[交差検証]: http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
