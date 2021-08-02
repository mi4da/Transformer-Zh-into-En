import tensorflow as tf
physical_device = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_device[0], True)
import TrainProcess,EvaluateProcess
from Untils.Pipers import PIPERS
import DataProcessor

import tensorflow as tf
# 配置环境
def EnvironmentNeeded(param=False):
    if param != False:
        piper = PIPERS()
        # piper.ChageResponsities()
        piper.InstallCommonTools({"tensorflow_datasets"})
    else:
        pass
if __name__ == '__main__':

    # 环境配置
    EnvironmentNeeded()
    # 启动输入批处理
    train_data, input_pipe_line = DataProcessor.run()
    # 启动训练进程
    TrainProcess.run(train_data)
    # 启动推理进程
    EvaluateProcess.run(input_pipe_line)