
from TrainProcess import transformer,create_masks
import tensorflow as tf
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号



def run(input_pipe_line):
    # 引入评估方案
    def evaluate(inp_sentence):
        start_token = [input_pipe_line.subword_encoder_zh.vocab_size]
        end_token = [input_pipe_line.subword_encoder_zh.vocab_size + 1]

        # 输入语句是zh，增加开始和结束标记
        inp_sentence = start_token + input_pipe_line.subword_encoder_zh.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # 因为目标是英语，输入 transformer 的第一个词应该是
        # 英语的开始标记。
        decoder_input = [input_pipe_line.subword_encoder_en.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(input_pipe_line.MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = transformer(encoder_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

            # 从 seq_len 维度选择最后一个词
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # 如果 predicted_id 等于结束标记，就返回结果
            if predicted_id == input_pipe_line.subword_encoder_en.vocab_size + 1:
                return tf.squeeze(output, axis=0), attention_weights

            # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def plot_attention_weights(attention, sentence, result, layer):
        fig = plt.figure(figsize=(16, 8))
        sentence = input_pipe_line.subword_encoder_zh.encode(sentence)
        attention = tf.squeeze(attention[layer], axis=0)
        for head in range(attention.shape[0]):
            ax = fig.add_subplot(2, 4, head + 1)

            # 画出注意力权重
            ax.matshow(attention[head][:-1, :], cmap='viridis')
            # 设置字体属性
            fontdict = {'fontsize': 10}
            # 设置刻度
            ax.set_xticks(range(len(sentence) + 2))
            ###
            '''
            这里有一个坑！！！result长这样：
            [8173 4849 4849 4849 4849 4849 4849 4849 4849 4849 4849 4849 4849 4849
             4849 4849 4849 4849 4849 4849 4849 4849 4849 4849 4849 4849 4849 4849
             4849 4849 4849 4849 4849 4849 4849 4849 4849 4849 4849 4849 4849], shape=(41,), dtype=int32)
             发现了吗，头一个元素是“起始符”，他不会被编码成有效的字符，所以，
             由这串向量编码出来的的字符和它本身的长度不等！！！！！
             所以这里要减一！！！！！
            '''
            ######
            ax.set_yticks(range(len(result) - 1))  # 原本这里的长度是41，现在是40
            # 设置y的区间
            ax.set_ylim(len(result) - 1.5, -0.5)
            # 设置刻度标签
            ax.set_xticklabels(
                ['<start>'] + [input_pipe_line.subword_encoder_zh.decode([i]) for i in sentence] + ['<end>'],
                fontdict=fontdict, rotation=90)

            ax.set_yticklabels([input_pipe_line.subword_encoder_en.decode([i]) for i in result
                                if i < input_pipe_line.subword_encoder_en.vocab_size],
                               fontdict=fontdict)  # 这里的长度是40
            # 设置坐标标签
            ax.set_xlabel('Head {}'.format(head + 1))

        plt.tight_layout()
        plt.show()

    def translate(sentence, plot=''):
        result, attention_weights = evaluate(sentence)

        predicted_sentence = input_pipe_line.subword_encoder_en.decode([i for i in result
                                                                        if
                                                                        i < input_pipe_line.subword_encoder_en.vocab_size])

        print('Input: {}'.format(sentence))
        print('Predicted translation: {}'.format(predicted_sentence))
        #   print("'Predicted translation: In his talk, Karpathy talked about some of the things Tesla has been doing over the past few months")

        if plot:
            plot_attention_weights(attention_weights, sentence, result, plot)


    translate("你好.", plot='decoder_layer1_block2')
    print("Origainal Sentence : Hello.")

    translate("在演讲中，Karpathy 谈到了特斯拉在过去几个月中所做的一些事情", plot='decoder_layer2_block2')
    print("Origainal Sentence : In his talk, Karpathy talked about some of the things Tesla has been doing over the past few months")