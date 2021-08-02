"""
Transformer模型参数
"""
num_layers = 6
d_model = 512
dff = 2048
num_heads = 8
# input_vocab_size = subword_encoder_zh.vocab_size + 2
input_vocab_size = 2 ** 13
# target_vocab_size = subword_encoder_en.vocab_size + 2
target_vocab_size = 2 ** 13
dropout_rate = 0.1

"""
模型训练参数
"""
batch_size = 64
Epoch = 1