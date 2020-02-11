这是网上的一片博客讲的很好。原文链接：https://blog.csdn.net/cicisensy/article/details/82670191

看博客的时候有几个问题
1. encoder-decoder的思想很简单，encoder和decoder都可以使用rnn、lstm等。
   encoder可以使用cnn，但是decoder不能使用，因为cnn要针对一个矩阵，一个句子可以构成一个词向量矩阵。
   如果使用cnn、rnn、lstm作为encoder的话，很明显输出是个向量，这个向量配合解码出来的词向量做解码。
   思考：rnn做encoder可以方便使用attention，cnn怎样使用attention呢？
2. F函数的选择，没有看过相关论文这个应该怎么选。一个坑是：作者说选择F函数，F函数的功能是将这两个向量转换成一个权重，softmax的功能是对所有权重做一个归一化，而不是做分类。

