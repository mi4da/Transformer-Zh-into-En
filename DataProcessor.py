import logging
import os
import time
from pprint import pprint

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from Untils import Paramers

class Dirs:
    def __init__(self):
        # 配置输出目录
        self.output_dir = "./nmt"  # 字典的位置
        # 英文字典
        self.en_vocab_file = os.path.join(self.output_dir, "en_vocab")
        # 中文字典
        self.zh_vocab_file = os.path.join(self.output_dir, "zh_vocab")
        # 保存点目录
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoints")
        # 日志目录
        self.logs_path = os.path.join(self.output_dir, "logs")
        # 下载目录
        self.download_dir = "./tenserflow-datasets/downloads"
        # 不存在则创建文件
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


class DataPipeLine(Dirs):
    def __init__(self):
        super().__init__()

    def CheckWtm(self):
        # 检查目录所包含数据
        tmp_builder = tfds.builder("wmt19_translate/zh-en")
        print(tmp_builder.subsets)

    def DownloadData(self):
        # 配置日志
        logging.basicConfig(level="ERROR")
        # 不要以科学计数法打印
        np.set_printoptions(suppress=True)
        ###### 开始下载wtm2019中英文平行语料 #######

        config = tfds.translate.wmt.WmtConfig(
            version=tfds.core.Version('0.0.3'),
            language_pair=("zh", "en"),
            subsets={
                tfds.Split.TRAIN: ["newscommentary_v14"],  # select the news comment train dataset to be the dataset,
                tfds.Split.VALIDATION: ["newstest2018"]
            }
        )
        self.builder = tfds.builder("wmt_translate", config=config)  # fetch the dataset
        self.builder.download_and_prepare(download_dir=self.download_dir)  # download it, and write it to disk

    def SplitTrainTest(self):
        # 划分训练集以及验证集
        self.train_examples, self.val_examples = self.builder.as_dataset(split="train[:90%]",
                                                                         as_supervised=True), self.builder.as_dataset(
            split="train[90%:]", as_supervised=True)
        # 构造测试集
        self.test_examples = self.builder.as_dataset(split="validation", as_supervised=True)
        # 在子类方法里重写输出流
        # return self.train_examples, self.val_examples, self.test_examples

    def GetSampleData(self):
        sample_examples = []
        for en, zh in self.train_examples.take(3):
            en = en.numpy().decode("utf-8")  # use numpy().decode the string into utf-8 format
            zh = zh.numpy().decode("utf-8")
            sample_examples.append((en, zh))
        return sample_examples


class InputPipeLine:
    def __init__(self, max_lenth, batch_size):
        self.data = DataPipeLine()
        self.MAX_LENGTH = max_lenth
        self.BATCH_SIZE = batch_size

    def start(self):
        st = time.time()
        # 创建数据流
        self.data.DownloadData()
        self.data.SplitTrainTest()

        # 创建输入流
        start = time.process_time()
        ## 建立英文词表
        try:
            self.subword_encoder_en = tfds.deprecated.text.SubwordTextEncoder.load_from_file(self.data.en_vocab_file)
            print("载入已建立的英文词表: {}".format(self.data.en_vocab_file))
        except:
            print("没有已建立的英文词表，从头建立")
            self.subword_encoder_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                (en.numpy() for en, _ in self.data.train_examples),
                target_vocab_size=2 ** 13  # 可以自定义字典大小
            )
            # 将字典保存以便下一次热启动
            self.subword_encoder_en.save_to_file(self.data.en_vocab_file)

        print("英文字典大小: {}".format(self.subword_encoder_en.vocab_size))
        end = time.process_time()
        print("建立英文字典用时：", end - start)
        # 建立中文词表
        start = time.process_time()
        try:
            self.subword_encoder_zh = tfds.deprecated.text.SubwordTextEncoder.load_from_file(self.data.zh_vocab_file)
            print(f"載入已建立的中文词表： {self.data.zh_vocab_file}")
        except:
            print("沒有已建立的中文词表，从头建立。")
            self.subword_encoder_zh = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                (zh.numpy() for _, zh in self.data.train_examples),
                # here should be _, zh, as the pair in training_set is like en-zh
                target_vocab_size=2 ** 13, max_subword_length=1)  # 有需要可以調整字典大小, 每一个中文字是一个单位

            # 將字典檔案存下以方便下次 warmstart
            self.subword_encoder_zh.save_to_file(self.data.zh_vocab_file)

        print(f"中文字典大小：{self.subword_encoder_zh.vocab_size}")
        # print(f"前 10 個 subwords：{subword_encoder_en.subwords[:10]}")
        # print()
        end = time.process_time()
        print("建立中文字典耗时：", end - start)

        # 将数据编码在tf静态图上
        self.train_dataset = self.data.train_examples.map(self.tf_encode)
        # 筛选长度小于40的句子
        self.train_dataset = self.train_dataset.filter(self.filter_max_length)

        # construct a batch
        # when constructing a batch, the length of each sequence need to be padded, so that there are in the same shape
        # padded_shapes: when None and -1, that means the shape of each element before batch will be padded 0 utill reach the maximun length of element in the batch
        self.train_dataset = self.train_dataset.padded_batch(self.BATCH_SIZE, padded_shapes=([-1], [-1]))
        ed = time.time()
        print("输入流总耗时：", ed - st)

    def end(self):
        return self.train_dataset

    def CheckOneBatchData(self):
        print("after batch and pad: ")
        en_batch, zh_batch = next(iter(self.train_dataset))
        print("English batch:")
        print(en_batch)
        print(20 * '-')
        print("Chinese batch:")
        print(zh_batch)

    def CheckSentenceAfertSubword(self):
        # zh
        sample_examples = self.data.GetSampleData()
        string = sample_examples[0]
        zh_string = string[1]
        print("each in sample_example: ", string, 10 * "-", "\nthe Chinese Part", zh_string,
              10 * "-", "\nis the item in sample_examples a tuple?", isinstance(string, tuple)
              )
        sample_string = sample_examples[0][1]

        indices = self.subword_encoder_zh.encode(sample_string)
        print("index of the string:", indices)

        for index in indices:
            print(index, 5 * ' ', self.subword_encoder_zh.decode([index]))

        # en
        # test with a sentence
        sample_string = 'Guangzhou is beautiful.'
        indices = self.subword_encoder_en.encode(sample_string)
        print("sample string: ", sample_string, "index of sample string: ", indices)
        print(100 * '-')

        # recover from the indices
        for index in indices:
            print(index, 5 * ' ', self.subword_encoder_en.decode([index]))

    def CheckTranslateSubwordIndices(self):
        en = "The eurozone’s collapse forces a major realignment of European politics."
        zh = "欧元区的瓦解强迫欧洲政治进行一次重大改组。"

        # 將文字轉成為 subword indices
        en_indices = self.subword_encoder_en.encode(en)
        zh_indices = self.subword_encoder_zh.encode(zh)

        print("[英中原文]（轉換前）")
        print(en)
        print(zh)
        print()
        print('-' * 20)
        print()
        print("[英中序列]（轉換後）")
        print(en_indices)
        print(zh_indices)
        print(100 * '-')

    # 插入起始符号与停止符号
    def encode(self, en_t, zh_t):
        # 因為字典的索引從 0 開始，
        # 我們可以使用 subword_encoder_en.vocab_size 這個值作為 BOS 的索引值
        # 用 subword_encoder_en.vocab_size + 1 作為 EOS 的索引值
        en_indices = [self.subword_encoder_en.vocab_size] + self.subword_encoder_en.encode(
            en_t.numpy()) + [self.subword_encoder_en.vocab_size + 1]
        # 同理，不過是使用中文字典的最後一個索引 + 1
        zh_indices = [self.subword_encoder_zh.vocab_size] + self.subword_encoder_zh.encode(
            zh_t.numpy()) + [self.subword_encoder_zh.vocab_size + 1]

        return en_indices, zh_indices

    def tf_encode(self, en_t, zh_t):
        # because in the dataset.map(), which is run in Graph mode instead of eager mode,
        # so the en_t, zh_t are not eager tensor, which do not contain the .numpy()
        return tf.py_function(self.encode, [en_t, zh_t], [tf.int64, tf.int64])
        # this will wrap the encode() into a eager mode enabled function in Graph mode when do the map() later on.

    def CheckInsertSEToken(self):
        # en, zh = next(iter(
        #     self.data.train_examples))  # here en,zh are just Tensor:<tf.Tensor: id=248, shape=(), dtype=string, numpy=b'Making Do With More'>
        # en_t, zh_t = self.encode(en, zh)
        # pprint((en, zh))
        # print("after pre-process:")
        # pprint((en_t, zh_t))

        print(100 * '-')
        print("after pre-processed the whole trainning dataset: (take one pair example)")
        en_indices, zh_indices = next(iter(self.train_dataset))
        pprint((en_indices.numpy(), zh_indices.numpy()))
        print(100 * '-')

    def filter_max_length(self, en, zh):
        return tf.logical_and(tf.size(en) <= self.MAX_LENGTH, tf.size(zh) <= self.MAX_LENGTH)

    def CheckAfterFilter(self):
        num_of_data = 0
        num_of_invaild = 0
        for each in self.train_dataset:
            en, zh = each
            if tf.size(en) <= self.MAX_LENGTH and tf.size(zh) <= self.MAX_LENGTH:
                num_of_data += 1
            else:
                num_of_invaild += 1

        print(f"the train_dateset has {num_of_invaild} invalid data, and total {num_of_data} remained valid data")
        print(100 * '-')

def run():

    # 构造输入流
    input_pipe_line = InputPipeLine(max_lenth=40, batch_size=Paramers.batch_size)
    # 开始
    input_pipe_line.start()
    # 结束，获取训练数据
    train_data = input_pipe_line.end()

    # 检查检查BPE模型效果
    # input_pipe_line.CheckSentenceAfertSubword()
    # # 检查词向量
    # input_pipe_line.CheckTranslateSubwordIndices()
    # # 检查添加了起始于终止字符的tokens
    # input_pipe_line.CheckInsertSEToken()
    # # 检查长度筛选效果
    # # input_pipe_line.CheckAfterFilter()
    # # 检查batch
    # input_pipe_line.CheckOneBatchData()

    return (train_data, input_pipe_line)

if __name__ == '__main__':
    train_data, inputpipline = run()
    print(next(iter(train_data)))