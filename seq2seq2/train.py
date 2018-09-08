import  numpy as np
import time
import tensorflow as tf
with open("D:\资料\GitHub\seq2seq\data\letters_source.txt","r",encoding="utf-8") as f:
    source_data = f.read()

with open("D:\资料\GitHub\seq2seq\data\letters_target.txt","r",encoding="utf-8") as f:
    target_data = f.read()

#数据预览 表示预览前10个
print(source_data.split("\n")[:10])
print(target_data.split("\n")[:10])

#进行数据预处理
def extract_character_vocab(data):
    #构造映射表
    special_words = ['<PAD>','<UNK>','<GO>','<EOS>']
    set_words = list(set(character for line in data.split("\n") for character in line))
    #可遍历的数据对象 同时列出数据和数据下表
    int_to_vocab = {idx:word for idx,word in enumerate(special_words + set_words)}
    #以字典列表形式返回
    vocab_to_int = {word:idx for idx,word in int_to_vocab.items()}
    return int_to_vocab,vocab_to_int
#构造映射表
source_int_to_letter,source_letter_to_int = extract_character_vocab(source_data)
target_int_to_letter,target_letter_to_int = extract_character_vocab(target_data)
#对字母进行转换
# 构造映射表
# 构造映射表
source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data)
target_int_to_letter, target_letter_to_int = extract_character_vocab(target_data)

# 对字母进行转换
source_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>'])
               for letter in line] for line in source_data.split('\n')]
target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>'])
               for letter in line] + [target_letter_to_int['<EOS>']] for line in target_data.split('\n')]

print(source_int[:10])
#构造模型 输入层
def get_input():
    """
    模型输入tensor
    :return:

    """
    inputs = tf.placeholder(tf.int32,[None,None],name = "inputs")
    targets = tf.placeholder(tf.int32,[None,None],name = "targets")
    #学习率
    learning_rate = tf.placeholder(tf.float32,name = "learning_rate")
    #定义target序列的最大长度 之后target_sequence_length和source_sequence_length会作为feed_dict喂入
    target_sequence_length = tf.placeholder(tf.int32,(None,),name ="target_sequence_length")
    max_target_sequence_length = tf.reduce_max(target_sequence_length,name ="max_target_length")
    source_sequence_length = tf.placeholder(tf.int32,(None,),name = "souece_sequence_length")
    return inputs,targets,learning_rate,target_sequence_length,max_target_sequence_length,source_sequence_length


def get_encoder_layer(input_data,rnn_size,num_layers,source_sequence_length,
                      source_vocab_size,encoding_embedding_size):
    """

    :param input_data: 输入tensor
    :param rnn_size: rnn隐层节点的个数
    :param num_layers: 堆叠的rnn cell数量
    :param source_sequence_length: 源数据的序列长度
    :param source_vocab_size: 源数据的词典大小
    :param encoding_embedding_size: embedding的大小
    :return:
    """
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data,source_vocab_size,encoding_embedding_size)
    #RNN cell
    def get_lstm_cell(rnn_size):
        #随机值范围的下限 随机值范围的上限 seed用于创建随机种子
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size,initializer = tf.random_uniform_initializer(-0.1,0.1,seed = 2))
    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
    encoder_output,encoder_state = tf.nn.dynamic_rnn(cell,encoder_embed_input,
                                                     sequence_length = source_sequence_length)
    return encoder_output,encoder_state


def process_decoder_input(data,vocab_to_int,batch_size):
    """
    补充<GO>,并且移除最后一个字符
    :param data:
    :param vocab_to_int:
    :param batch_size:
    :return:
    """
    #end索引是开区间，截掉最后一列
    ending = tf.strided_slice(data,[0,0],[batch_size,-1],[1,1])
    decoder_input = tf.concat([tf.fill([batch_size,1],vocab_to_int['<GO>']),ending],1)
    return decoder_input