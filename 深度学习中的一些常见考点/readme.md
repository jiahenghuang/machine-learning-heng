1. Layer Normalization
   归一化的本质都是将数据转化为均值为0，方差为1的数据。这样可以减小数据的偏差，规避训练过程中梯度消失或爆炸的情况。我们在训练网络中比较常见的归一化方法是Batch Normalization，即在每一层输出的每一批数据上进行归一化。而Layer Normalization与BN稍有不同，即在每一层输出的每一个样本上进行归一化。

2. Batch Normalization

3. Residual connection
   残差连接其实在很多网络机构中都有用到。原理很简单，假设一个输入向量x，经过一个网络结构，得到输出向量f(x)，加上残差连接，相当于在输出向量中加入输入向量，即输出结构变为f(x)+x，这样做的好处是在对x求偏导时，加入一项常数项1，避免了梯度消失的问题。

4. 对于RNN这种没法用mini-batch的网络，没办法用BN，所以提出了Layer Normalization







