import os
import time

import tensorflow as tf


from Transformer.transformer import Transformer
from Untils.Paramers import *

from Untils.untils import CustomSchedule, create_masks

from Untils import Paramers



# 设定损失函数与评测指标
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')
# 引入模型，优化器
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)




# 使用tf.function加速
# 该 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地
# 执行。该函数专用于参数张量的精确形状。为了避免由于可变序列长度或可变
# 批次大小（最后一批次较小）导致的再追踪，使用 input_signature 指定
# 更多的通用形状。

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


# 定义训练步骤
@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]# 最后一列之前的所有列
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)

def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)
# 开始训练
def run(train_data):
    # 引入检查点管理器
    checkpoint_path = "./training_data/checkpoints/train"

    create_dir_not_exist(checkpoint_path)

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    # 训练
    EPOCH = Paramers.Epoch
    for epoch in range(EPOCH):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inpit -> zh, target -> en
        for (batch, (en, zh)) in enumerate(train_data):
            train_step(zh, en)

            if batch % 50 == 0:
                print("Epoch {} batch {} loss {:.4f} Accuracy {:.4f}".format(epoch + 1,
                                                                             batch,
                                                                             train_loss.result(),
                                                                             train_accuracy.result()))

        end = time.time()
        print("训练完毕！ 总共耗时： ", end - start)
