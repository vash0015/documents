# 調査の趣旨や雑感など

最近の機械学習のトレンドは Deep Leaning らしい。次期PJでは先進的な検出システム(超迅速多項目)の研究開発に関わるとのことで Deep Learning について概要だけでも触れておくべきと判断した。

ざっとスライド類を眺めてみて「従来職人芸で行われていた特徴抽出をメタレベルで設計し、自動的に行わせることができる」というのが最大の特徴ではないかと感じた。

![](http://theanalyticsstore.ie/wp-content/uploads/2013/03/DeepNetwork.png)
こういうことね。

このノード自体は自動的に生成されるので、実装者が行うべきは適切にノードが生成されるようメタレベルで設計することなのである。調整しなかったら同じ要素を検出するノードが複数できてしまって精度が落ちるとかあるらしい。

[はじめるDeep learning](
http://qiita.com/icoxfog417/items/65e800c3a2094457c3a0)

> つまるところ、Deep learningの特徴は「特徴の抽出までやってくれる」という点に尽きると思います。  
> 例えば相撲取りを判定するモデルを構築するとしたら、普通は「腰回りサイズ」「マゲの有無」「和装か否か」といった特徴を定義して、それを元にモデルを構築することになります。ちょうど関数の引数を決めるようなイメージです。  
> ところが、Deep learningではこの特徴抽出もモデルにやらせてしまいます。というか、そのために多層、つまりDeepになっています。  
> 具体的には頭のあたりの特徴、腰のあたりの特徴、そしてそれらを複合した上半身の特徴・・・というように、特徴の抽出を並列・多層に行って学習させて、それでもって判定させようというのが根本的なアイデアです  


# Deep Learning のインパクト

[特徴量研究者涙目](https://twitter.com/ambee_whisper/status/256039859528019968)
だとかなんとか。

スライド
[実装 ディープラーニング]( http://www.slideshare.net/yurieoka37/ss-28152060)
から引用。

![8ページ]( http://image.slidesharecdn.com/deeplearning-121109044234-phpapp01/95/deep-learning-8-638.jpg?cb=1352436208)

![9ページ]( http://image.slidesharecdn.com/deeplearning-121109044234-phpapp01/95/deep-learning-9-638.jpg?cb=1352436208)

![12ページ]( http://image.slidesharecdn.com/deeplearning-121109044234-phpapp01/95/deep-learning-12-638.jpg?cb=1352436208)






# キーワード

[活性化関数], [Rectifier], [ReLU(=A unit employing the rectifier)], 過学習回避手法, Dropout, 教師なし事前学習, restricted Boltzmann machine,  RBM, Autoencoder, Bag-of-visual-words, Convolutional neural network, ConvNets, Fine-tuning, Pre-trained feature,  [ImageNet], shallow lerning, [Caffe], [softmax]


# 参考サイト

* スライド
    * [一般向けの Deep Learning]( http://www.slideshare.net/pfi/deep-learning-22350063)
    * [実装 ディープラーニング]( http://www.slideshare.net/yurieoka37/ss-28152060)
    * [Deep Learning と画像認識]( http://www.slideshare.net/nlab_utokyo/deep-learning-40959442?next_slideshow=2)
    * [Deep learning]( http://www.slideshare.net/kazoo04/deep-learning-15097274)
    * [Tutorial on Deep Learning and Applications]( https://deeplearningworkshopnips2010.files.wordpress.com/2010/09/nips10-workshop-tutorial-final.pdf)
* 解説記事
    * [はじめる Deep learning]( http://qiita.com/icoxfog417/items/65e800c3a2094457c3a0)
    * []()
    * []()
    * []()




[活性化関数]: http://ja.wikipedia.org/wiki/ニューラルネットワーク#.E3.83.95.E3.82.A3.E3.83.BC.E3.83.89.E3.83.95.E3.82.A9.E3.83.AF.E3.83.BC.E3.83.89.E3.83.8B.E3.83.A5.E3.83.BC.E3.83.A9.E3.83.AB.E3.83.8D.E3.83.83.E3.83.88

[Caffe]: http://caffe.berkeleyvision.org/

[ImageNet]: http://smrmkt.hatenablog.jp/entry/2015/03/08/195625

[Rectifier]:  http://en.wikipedia.org/wiki/Rectifier_(neural_networks)

[ReLU(=A unit employing the rectifier)]: http://en.wikipedia.org/wiki/Rectifier_(neural_networks)

[softmax]: http://en.wikipedia.org/wiki/Softmax_function
