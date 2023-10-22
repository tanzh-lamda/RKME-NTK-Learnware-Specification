# NTK-RF Experiments

## 实验复现指南

我们提供了一个启动脚本**auto.bash**，只需要对脚本进行一些简单的修改，就可以多核并行复现我们的实验结果。

### 配置启动脚本

**Step 1.** 将auto.bash中的**nt**修改为运行设备的conda环境名。

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nt
```

**Step 2.** 将auto.bash中的**num**修改为运行设备支持的最大cuda核心数，或者在启动auto.bash时键入参数。
```bash
num=8
if [ $# -eq 1 ]
  then
    num=$1
fi
```

### 启动与终止

运行脚本启动实验，
```
./auto.bash
```

脚本会在log文件夹下创建一个以Unix时间戳{token}命名的文件夹，用来储存程序输出，即log/{token}。 如果需要终止后台程序，可以使用命令，

```
./kill.bash {token}
```

### 生成数据集划分

修改auto.bash中的param参数，并执行脚本。
```bash
--mode resplit --n_uploaders 50 --n_users 50 -K 48 --data cifar10
```

### 基于RBF实验

修改auto.bash中的param参数，并执行脚本。

```bash
--mode auto --n_uploaders 50 --n_users 50 -K 48 --auto_param data_id --spec rbf --data cifar10
```

### 基于NTK实验

修改auto.bash中的param参数，并执行脚本。注意，由于计算量较大，通常NTK实验需要较长时间。

```bash
--mode auto --n_uploaders 50 --n_users 50 -K 48 --auto_param data_id --spec ntk --data cifar10
```

## 注意事项

请将数据集文件cifar-10-python.tar.gz提前拷入image_models/data文件夹中。多进程同时自动下载数据可能导致意想不到的错误。


## 项目结构

preprocess中用于加载数据，划分数据集，为上传者训练模型。

utils中提供了随机模型生成工具，集成学习，ntk实现等核心算法实现。

build_market.py和evaluate_market.py是实验的主要工作流。

benchmark.py中提供了理论上单个学习器的最优性能评测，也即论文中的ORACLE情景。

diagram中提供了实验的绘图部分实现。
