这是网上的一片博客讲的很好。原文链接：https://blog.csdn.net/cicisensy/article/details/82670191

看博客的时候有几个问题
1. encoder-decoder的思想很简单，encoder和decoder都可以使用rnn、cnn、lstm等。
2. F函数的选择，没有看过相关论文这个应该怎么选。但是根据经验来说，可以做拼接、做差这两个思想都是可以的，因为将两个向量做上述两种操作之后都可以做softmax。
   做softmax时候可能会造成样本的不平衡，因为一个正样本往往对应多个负样本，要不要进行采样。

