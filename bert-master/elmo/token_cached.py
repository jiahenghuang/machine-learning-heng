import tensorflow as tf
import os
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, dump_token_embeddings

raw_context = ['这 是 测试 .', '好的 .']

tokenized_context = [sentence.split() for sentence in raw_context]
tokenized_question = [['这', '是', '什么'],]

vocab_file = '../seg_words.txt'
options_file = '../log/options.json'
weight_file = '../log/weights.hdf5'
token_embedding_file = '../log/vocab_embedding.hdf5'

batcher = TokenBatcher(vocab_file)

context_token_ids = tf.placeholder('int32', shape=(None, None))
question_token_ids = tf.placeholder('int32', shape=(None, None))

bilm = BidirectionalLanguageModel(options_file, weight_file, use_character_inputs=False, embedding_weight_file=token_embedding_file)

context_embeddings_op = bilm(context_token_ids)
question_embeddgins_op = bilm(question_token_ids)

elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
with tf.variable_scope('', reuse=True):
    elmo_question_input = weight_layers('input', question_embeddgins_op, l2_coef=0.0)

elmo_context_output = weight_layers('output', context_embeddings_op, l2_coef=0.0)
with tf.variable_scope('', reuse=True):
    elmo_question_output = weight_layers('output', question_embeddgins_op, l2_coef=0.0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    context_ids = batcher.batch_sentences(tokenized_context)
    question_ids = batcher.batch_sentences(tokenized_question)

    elmo_context_input_, elmo_question_input_ = sess.run(
        [elmo_context_input['weighted_op'], elmo_question_input['weighted_op']],
        feed_dict={context_token_ids: context_ids,
                   question_token_ids: question_ids}
                   )

print(elmo_context_input_, elmo_question_input_)