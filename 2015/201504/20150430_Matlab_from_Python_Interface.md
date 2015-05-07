# Matlab を別言語から呼び出すための基本事項

Matlab と外部言語との関係性は下記のサイトの説明が分かりやすい。

[MATLAB & Python - スムーズワークス日想](
http://blog.smooth-works.net/archives/3607)

![m-file_pymex_python](
http://blog.smooth-works.net/wp-content/uploads/2013/05/blog_2013_05_20_00.png)

pymex というツールを経由して Matlab-Python 間の変数の変換を行っている。
基本的にはこの図式の通りになるのだけど pymex は既に更新が止まっている。
変換に使うツールは別のものを採用したほうが良いかもしれない。

基本的なデータハンドリングは Python でも出来るのだが、
Simulink みたいな拡張パックとかどうしても Matlab じゃないと出来ないことがある。
そこを考慮して、連携方法を取り決める必要がある。


# pymex が何をしているか確認する

[MATLAB & Python - スムーズワークス日想](
http://blog.smooth-works.net/archives/3607)

> 具体的には、pymexを使うことで次の事が実現できます。
>
> * MATLABから、Pythonモジュールを呼び出せる（引数つきで）  
> * Pythonからの戻り値を受け取れる
>
> ポイントは、MATLAB変数と、Python変数の間の変換をしてくれるところにあり、pymexのソースコードの大部分は、その処理に費やされています。

とのこと。自分で変換ロジックを組むのはよくないのだろうか？


# Matlab Engine for Python

[Python 用の MATLAB エンジン](
http://jp.mathworks.com/help/matlab/matlab-engine-for-python.html)

Mathworks 公式の呼び出しライブラリ。
とりあえず呼べるのは呼べる？

** MATLAB エンジンの開始 **

```python
import matlab.engine
eng = matlab.engine.start_matlab()
```

** 複数のエンジンの開始 **  

```python
import matlab.engine
eng1 = matlab.engine.start_matlab()
eng2 = matlab.engine.start_matlab()
```
