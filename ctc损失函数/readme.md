1. ctc解决什么问题？
   输入是一个序列，输出是一个序列，不好做标注或者没办法做标注。
   如：语音识别的输入是一串语音，这一串语音需要分解为帧，一帧往往是几ms，这个谁能标注的出来？要标注多久？
2. ctc缺点
   CTC做了一个假设就是不同时间步的输出之间是独立的。这个假设对于很多序列问题来说并不成立，输出序列之间往往存在联系。
   CTC只允许单调对齐，在语音识别中可能是有效的，但是在机器翻译中，比如目标语句中的一些比较后的词，可能与源语句中前面的一些词对应，这个CTC是没法做到的。
   CTC的输入和输出是多对一的关系。这意味着输出长度不能超过输入长度，这在手写字体识别或者语音中不是什么问题，因为通常输入都会大于输出，但是对于输出长度大于输入长度的问题CTC就无法处理了。
3. ctc算法的核心就是他的损失函数，求解参数的核心就是前向、后向算法。就是动态规划。
4. 当一个rnn-ctc算法训练完成了，拿去预测数据，得到的结果是什么？
   一个包含很多重复项的序列。需要加一个去除重复字符的那个功能就可以了。
   这里想到了一个数据结构算法。怎样得到ctc的标注输出？
   双指针