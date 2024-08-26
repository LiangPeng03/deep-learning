这是一个图像识别项目，基于 tensorflow，现有的 CNN 网络可以识别电梯门的三种状态（开、中间状态和关）。

运行项目需要虚拟环境：
  需要自行安装 Anaconda
  导入环境：命令行输入  conda env update -f=environment.yaml

训练模型：
  修改 cnn.py 中
    train_dir = 'D:\\澳科读书\\实习\\深度学习\\data'  # 训练样本的读入路径
    logs_train_dir = 'D:\\澳科读书\\实习\\深度学习\\save'  # logs存储路径
  为你本机的目录。

运行 train.py 开始训练。
  训练完成后，修改 test.py 中的logs_train_dir = 'D:\\澳科读书\\实习\\深度学习\\save'为你的目录。
  运行 test.py 或者 gui.py 查看结果。


This image recognition project, based on TensorFlow, employs an existing convolutional neural network (CNN) to 
discern the three states of the elevator door: open, middle, and closed.

Create a virtual environment:
  1. install Anaconda.
  2. enter the command:  conda env update -f=environment.yaml

Modifications to cnn.py are required. The training directory is specified as follows:
  train_dir = 'D:\\澳科读书\\实习\\深度学习\\data' # Read path for training samples
  logs_train_dir = 'D:\\澳科读书\\实习\\深度学习\\save' # logs storage path

Execute the train.py script. 
  Upon completion of the training phase, modify the logs_train_dir = 'D:\\澳科读书\\实习\\深度学习\\save' in test.py to reflect the directory on your machine.
  run test.py or gui.py.
