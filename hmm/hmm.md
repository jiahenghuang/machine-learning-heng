## 状态转移概率、发射概率、初始状态概率分布
初始状态就是假设我们有很多个句子，每个句子都有会有第一个字。
初始概率就是我们由N个隐状态，统计每个隐状态作为句子第一个字的次数记为k，总共由n个句子，那这个隐状态对应的概率值为n/k。初始概率分布的维度和状态数的维度一致，是一个列表，而不是二维矩阵

```
class HMM(object):
    def __init__(self, N, M):
        """Args:
            N: 状态数，这里对应存在的标注的种类
            M: 观测数，这里对应有多少不同的字
        """
        self.N = N
        self.M = M

        # 状态转移概率矩阵 A[i][j]表示从i状态转移到j状态的概率
        self.A = torch.zeros(N, N)
        # 观测概率矩阵, B[i][j]表示i状态下生成j观测的概率
        self.B = torch.zeros(N, M)
        # 初始状态概率  Pi[i]表示初始时刻为状态i的概率
        self.Pi = torch.zeros(N)
```
## 理解三个参数中有些值为0之后的操作方法
### 当A中有值为0
转移概率为0，意味着某个状态到另外一个状态是不可能发生转移，
这个在更新维特比矩阵的时候会出问题，因为需要做乘法，为0意味着永远不可能为值。另外一点从哲学上讲不可能发生的事几乎没有，为很小的概率是可接受的。
### 当B中有值为0

### 当pi中有值为0
这意味着，某个隐状态并没有出现在句子中的第一个字过

## E步和M步怎么体现过来的？
通过如下代码，发现估计A B Pi都是直接统计频率而已，并没有迭代过程。EM算法如何体现出来的？
因为em算法针对的是无监督的hmm，而像命名实体识别这种任务是有监督的，直接统计频率即可，因为多项分布的极大似然估计就是频率

```
class HMM(object):
    def __init__(self, N, M):
        ....
    def train(self, word_lists, tag_lists, word2id, tag2id):
        """HMM的训练，即根据训练语料对模型参数进行估计,
           因为我们有观测序列以及其对应的状态序列，所以我们
           可以使用极大似然估计的方法来估计隐马尔可夫模型的参数
        参数:
            word_lists: 列表，其中每个元素由字组成的列表，如 ['担','任','科','员']
            tag_lists: 列表，其中每个元素是由对应的标注组成的列表，如 ['O','O','B-TITLE', 'E-TITLE']
            word2id: 将字映射为ID
            tag2id: 字典，将标注映射为ID
        """

        assert len(tag_lists) == len(word_lists)

        # 估计转移概率矩阵
        for tag_list in tag_lists:
            seq_len = len(tag_list)
            for i in range(seq_len - 1):
                current_tagid = tag2id[tag_list[i]]
                next_tagid = tag2id[tag_list[i+1]]
                self.A[current_tagid][next_tagid] += 1
        # 一个重要的问题：如果某元素没有出现过，该位置为0，这在后续的计算中是不允许的
        # 解决方法：我们将等于0的概率加上很小的数
        self.A[self.A == 0.] = 1e-10
        self.A = self.A / self.A.sum(dim=1, keepdim=True)

        # 估计观测概率矩阵
        for tag_list, word_list in zip(tag_lists, word_lists):
            assert len(tag_list) == len(word_list)
            for tag, word in zip(tag_list, word_list):
                tag_id = tag2id[tag]
                word_id = word2id[word]
                self.B[tag_id][word_id] += 1
        self.B[self.B == 0.] = 1e-10
        self.B = self.B / self.B.sum(dim=1, keepdim=True)

        # 估计初始状态概率
        for tag_list in tag_lists:
            init_tagid = tag2id[tag_list[0]]
            self.Pi[init_tagid] += 1
        self.Pi[self.Pi == 0.] = 1e-10
        self.Pi = self.Pi / self.Pi.sum()
```

## 维特比解码
总体分为两部分，更新维特比矩阵和backpointer矩阵。
在计算得到的维特比矩阵上，进行回溯求最优路径。就是动态规划算法
### 为什么要引入维特比矩阵

### 为什么要引入backpointer矩阵
为了维特比回溯使用
### 如何从一个实际的例子上去解释维特比解码
当一个句子过来了，我们得到了一个可观测序列。同时A B Pi都是已知的。
首先更新维特比矩阵。维特比矩阵的元素的含义是：行是对应的字，
列是对应的隐状态，表示经过当前字和当前隐状态的二维表的这个点，对应点最优路径（序列概率最大）。当更新完成了这个矩阵，意味着，经过每个二维表的每个点的最优路径就求到了。当二维表更新完成之后，需要比较下最后一个字对应的隐状态的概率值，即可知道整个序列的概率值最大的为多少。最大的那个，就是最后一个字对应的隐状态。但是这个时候还不知道最优路径是什么，仅仅知道了最优路径的概率值。
这个时候就想到了动态规划算法。
知道了目标点，需要求路径问题。
当需要回溯的时候，我们需要知道到达这个点前的那个最优点是什么，这个就是backpointer矩阵的作用。backpointer矩阵的第i,j个元素的含义是：维特比矩阵的第i，j的点对应的前一个字的隐状态的id。就是最优的当前字是由前面哪个字转移过来的。这样就可以一点点往前回溯。
```
class HMM(object):
    ...
    def decoding(self, word_list, word2id, tag2id):
        """
        使用维特比算法对给定观测序列求状态序列， 这里就是对字组成的序列,求其对应的标注。
        维特比算法实际是用动态规划解隐马尔可夫模型预测问题，即用动态规划求概率最大路径（最优路径）
        这时一条路径对应着一个状态序列
        """
        # 问题:整条链很长的情况下，十分多的小概率相乘，最后可能造成下溢
        # 解决办法：采用对数概率，这样源空间中的很小概率，就被映射到对数空间的大的负数
        #  同时相乘操作也变成简单的相加操作
        A = torch.log(self.A)
        B = torch.log(self.B)
        Pi = torch.log(self.Pi)

        # 初始化 维比特矩阵viterbi 它的维度为[状态数, 序列长度]
        # 其中viterbi[i, j]表示标注序列的第j个标注为i的所有单个序列(i_1, i_2, ..i_j)出现的概率最大值
        seq_len = len(word_list)
        viterbi = torch.zeros(self.N, seq_len)
        # backpointer是跟viterbi一样大小的矩阵
        # backpointer[i, j]存储的是 标注序列的第j个标注为i时，第j-1个标注的id
        # 等解码的时候，我们用backpointer进行回溯，以求出最优路径
        backpointer = torch.zeros(self.N, seq_len).long()

        # self.Pi[i] 表示第一个字的标记为i的概率
        # Bt[word_id]表示字为word_id的时候，对应各个标记的概率
        # self.A.t()[tag_id]表示各个状态转移到tag_id对应的概率

        # 所以第一步为
        start_wordid = word2id.get(word_list[0], None)
        Bt = B.t()
        if start_wordid is None:
            # 如果字不再字典里，则假设状态的概率分布是均匀的
            bt = torch.log(torch.ones(self.N) / self.N)
        else:
            bt = Bt[start_wordid]
        viterbi[:, 0] = Pi + bt
        backpointer[:, 0] = -1

        # 递推公式：
        # viterbi[tag_id, step] = max(viterbi[:, step-1]* self.A.t()[tag_id] * Bt[word])
        # 其中word是step时刻对应的字
        # 由上述递推公式求后续各步
        for step in range(1, seq_len):
            wordid = word2id.get(word_list[step], None)
            # 处理字不在字典中的情况
            # bt是在t时刻字为wordid时，状态的概率分布
            if wordid is None:
                # 如果字不再字典里，则假设状态的概率分布是均匀的
                bt = torch.log(torch.ones(self.N) / self.N)
            else:
                bt = Bt[wordid]  # 否则从观测概率矩阵中取bt
            for tag_id in range(len(tag2id)):
                max_prob, max_id = torch.max(
                    viterbi[:, step-1] + A[:, tag_id],
                    dim=0
                )
                viterbi[tag_id, step] = max_prob + bt[tag_id]
                backpointer[tag_id, step] = max_id

        # 终止， t=seq_len 即 viterbi[:, seq_len]中的最大概率，就是最优路径的概率
        best_path_prob, best_path_pointer = torch.max(
            viterbi[:, seq_len-1], dim=0
        )

        # 回溯，求最优路径
        best_path_pointer = best_path_pointer.item()
        best_path = [best_path_pointer]
        for back_step in range(seq_len-1, 0, -1):
            best_path_pointer = backpointer[best_path_pointer, back_step]
            best_path_pointer = best_path_pointer.item()
            best_path.append(best_path_pointer)

        # 将tag_id组成的序列转化为tag
        assert len(best_path) == len(word_list)
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        tag_list = [id2tag[id_] for id_ in reversed(best_path)]

        return tag_list
```
