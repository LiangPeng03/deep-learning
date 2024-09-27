# 图像识别项目

这是一个基于 TensorFlow 的图像识别项目，使用现有的卷积神经网络（CNN）来识别电梯门的三种状态：开、中间状态和关。

## 运行项目需要虚拟环境

1. 需要自行安装 Anaconda。
2. 导入环境：命令行输入 `conda env update -f=environment.yaml`
3. 启动虚拟环境：命令行输入 `conda activate tf1`

## 训练模型

1. 修改 `cnn.py` 中：
   ```python
   train_dir = 'D:\\澳科读书\\实习\\深度学习\\data'  # 训练样本的读入路径
   logs_train_dir = 'D:\\澳科读书\\实习\\深度学习\\save'  # logs存储路径
   ```
  为你本机的目录。

2. 运行 train.py 开始训练。

3. 训练完成后，修改 test.py 中的 logs_train_dir = 'D:\\澳科读书\\实习\\深度学习\\save' 为你的目录。

4. 运行 test.py 或者 gui.py 查看结果。

-------------------------------   
# project
This image recognition project, based on TensorFlow, employs an existing convolutional neural network (CNN) to discern the three states of the elevator door: open, middle, and closed.

## Create a virtual environment
1. Install Anaconda.
2. Enter the command: `conda env update -f=environment.yaml`.
3. Enter the command: `conda activate tf1`
   
## Modifications to cnn.py 
1. The training directory is specified as follows:<br>
```python
    train_dir = `D:\\澳科读书\\实习\\深度学习\\data`  # Read path for training samples<br>
    logs_train_dir = `D:\\澳科读书\\实习\\深度学习\\save`  # logs storage path
```
2. Execute the train.py script.

3. Upon completion of the training phase, modify the `logs_train_dir = 'D:\\澳科读书\\实习\\深度学习\\save'` in test.py to reflect the directory on your computer.

4. Run test.py or gui.py.
