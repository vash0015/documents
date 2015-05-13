# サイトツリー

* [トップページ]
    * [序論]
        * [機械学習：問題設定]
        * [サンプルデータセットの読込]
        * [学習と予測]
    * [科学データ処理のための統計学習チュートリアル]


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

digitデータセットは10クラス(0から9までの数字)のデータセットである。このデータセットを使うことで未知のサンプルから数字を推定することが出来る。











[トップページ]: http://scikit-learn.org/stable/tutorial/index.html

[序論]: http://scikit-learn.org/stable/tutorial/basic/tutorial.html

[機械学習：問題設定]: http://scikit-learn.org/stable/tutorial/basic/tutorial.html#machine-learning-the-problem-setting

[サンプルデータセットの読込]: http://scikit-learn.org/stable/tutorial/basic/tutorial.html#loading-an-example-dataset

[学習と予測]: http://scikit-learn.org/stable/tutorial/basic/tutorial.html#model-persistence

[科学データ処理のための統計学習チュートリアル]: http://scikit-learn.org/stable/tutorial/statistical_inference/index.html
