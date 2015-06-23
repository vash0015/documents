Python-Excel ライブラリのインストール方法


% 概要

インストールが必要なライブラリは下記の2つ。
・xlrd : Excel 入力ライブラリ
・xlwt : Excel 出力ライブラリ

いずれも pip インストールでよい。


% xlrt

通常のpipインストールで問題ない。

pip install xlrd


% xlwt

Python3.4 64-bitでnumpyなどインストール。
http://hennohito.cocolog-nifty.com/blog/2014/04/python34-64-b-1.html

どを参照してみると、現状 xlwt のメンテナンス状況がよくない。
> githubのpull requestが長いこと放置されている。

Python 3.x 系列との互換性も鑑みて、xlwt-futureをインストールする。
通常の xlwt と同様に import xlwt でインポートできるので、
単純に差し替えて使っていけると考えている。

pip install xlwt-future
