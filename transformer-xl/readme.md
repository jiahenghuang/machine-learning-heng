transformer-xl解决了transformer的两个问题：

transformer输入是固定长度的，并且由于self-attention的原因会产生O(L^2)的问题导致这个固定长度不能太长。
为了解决长文本问题，transformer通常采用将句子分解为segments，然后分别独立的处理segment，这样会丢失句子之间的关联性。
训练的时候transformer并没有将针不同segment的transformer进行关联。
但是，如果训练好的模型要拿来在线上做预测，必须将多个segment组合一起。
在预测的时候，会对固定长度的segment做计算，一般取最后一个位置的隐向量作为输出。为了充分利用上下文关系，在每做完一次预测之后，就对整个序列向右移动一个位置，再做一次计算，如上图(b)所示，这导致计算效率非常低。

1. Segment-Level Recurrence
   句子级别的循环神经网络。思想就是多层encoder，
   Trm-XL提出了一个改进，在对当前segment进行处理的时候，缓存并利用上一个segment中所有layer的隐向量序列，而且上一个segment的所有隐向量序列只参与前向计算，不再进行反向传播，这就是所谓的segment-level Recurrence。

2. Relative Position Encodings
   
   在vanilla Trm中，为了表示序列中token的顺序关系，在模型的输入端，对每个token的输入embedding，加一个位置embedding。位置编码embedding或者采用正弦\余弦函数来生成，或者通过学习得到。在Trm-XL中，这种方法行不通，每个segment都添加相同的位置编码，多个segments之间无法区分位置关系。Trm-XL放弃使用绝对位置编码，而是采用相对位置编码，在计算当前位置隐向量的时候，考虑与之依赖token的相对位置关系。具体操作是，在算attention score的时候，只考虑query向量与key向量的相对位置关系，并且将这种相对位置关系，加入到每一层Trm的attention的计算中。
