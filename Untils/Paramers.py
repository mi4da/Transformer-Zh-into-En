"""
Transformer模型参数
"""
# 编解码器的个数
num_layers = 6
# 词向量的维数
d_model = 512
# 隐藏神经元个数
dff = 2048
# n头注意力
num_heads = 8
# input_vocab_size = subword_encoder_zh.vocab_size + 2
# 输入词表大小
input_vocab_size = 2 ** 13
# 目标词表大小
# target_vocab_size = subword_encoder_en.vocab_size + 2
target_vocab_size = 2 ** 13
dropout_rate = 0.1

"""
模型训练参数
"""
batch_size = 64
Epoch = 1
