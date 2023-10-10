# NTK-RF Experiments

坦白地讲，这个实验启动起来有些复杂，但这些复杂是半自动化搜索参数必要的代价。

**Update**：我更新了nohop后台炼丹脚本，参见**auto.bash**。

## 启动说明

该实验的启动文件为根目录下**main.py**。一共提供了4种启动模式，通过--mode参数控制。下面，我分别介绍**resplit**、**regular**、**auto**三种模式，grid模式并不推荐使用。

![](.\pictures\启动参数.png)

### 须知

请先执行resplit模式，为上传者和用户生成数据集，再执行**auto**或**regular**模式。或在执行**auto**或**regular**模式时，加入参数--resplit，

### resplit 模式

这一模式用于对数据集随机抽样为上传者和用户生成数据集，并为上传者训练模型使用。通常，在使用**regular**或**auto**模式搜索参数前，用于重新生成上传者和用户的数据集。

其中，--id参数指定了数据集输出的位置。这一参数的存在，使得同时在多个随机抽样得到的数据集上测试模型成为可能。默认情况下，--id为0。

```bash
--id 0 --mode resplit --n_uploaders 50 --n_users 50 -K 50
--id 1 --mode resplit --n_uploaders 50 --n_users 50 -K 50
...
--id 7 --mode resplit --n_uploaders 50 --n_users 50 -K 50
```

### regular 模式

在这一模式下，所有的参数都由操作者手动控制，可以同时使用不同的参数启动程序。不过，在参数搜索时，修改若干个启动参数会降低工作效率，因此，这一模式只用于常规化运行实验，并不推荐用于参数搜索。

```bash
--mode regular --n_uploaders 50 --n_users 50 --spec ntk -K 50 --net_width 128 --ntk_steps 35
```

默认情况下，**regular**模式将使用**resplit**模式中--id 0所生成的上传者和用户数据集。如果希望使用其他上传者和用户数据集，可将其--id x对应--data_id x传入。

```bash
--mode regular --data_id 3 --n_uploaders 50 --n_users 50 --spec ntk -K 50 --net_width 128 --ntk_steps 35
```

### auto 模式

这一模式一般用于**参数搜索**。程序会根据--id（从0开始），使用不同的候选参数进行试验。注意，该模式下不应该显式使用--data_id参数。

**Update**：我更新了nohop后台炼丹脚本，参见**auto.bash**。

```bash
--mode auto --id 0 --n_uploaders 50 --n_users 50 -K 50
--mode auto --id 1 --n_uploaders 50 --n_users 50 -K 50
...
--mode auto --id 7 --n_uploaders 50 --n_users 50 -K 50
```

这些候选参数的设置，被记录在在main.py中的**CANDIDATES**和**auto_param**中，其中，--auto_param是要搜索的参数。特别需要说明的是，候选参数中data_id较为特殊。使用该参数时，不同的程序会使用相同的参数上，在不同的抽样上开展实验，这可以用于**评估实验精度**。

```python
CANDIDATES = {
    "model": ['conv', 'resnet'],
    "ntk_steps": [30, 35, 40, 45, 50, 55, 60],
    "sigma": [0.003, 0.004, 0.005, 0.006, 0.01, 0.025, 0.05, 0.1],
    "n_random_features": [32, 64, 96, 128, 196, 256],
    "net_width": [32, 64, 96, 128, 160, 196],
    "data_id": [0, 1, 2, 3, 4, 5, 6, 7],
    "net_depth": [3, 3, 4, 4, 5, 5, 6, 6]
}
```

## 参数设置

参数主要分为两类，其中一大类用于控制随机模型结构，另外一部分用于控制实验。

特别说明：代码中add_argument对应的help不都是非常准确。

### 随机模型结构参数

这一部分主要是复用了《Efficients》那篇文章的代码：

+ model：使用conv或resnet模型结构。推荐：conv。
+ ntk_steps: 使用ntk时，reduced set的优化轮数。推荐：35。
+ sigma: 使用ntk时，高斯初始化参数。推荐：None，让代码根据结构自己算。
+ n_random_features: 根据《Efficients》，**推荐：4096**。
+ net_width: conv网络宽度，根据《Efficients》，**推荐：128**。
+ net_depth: conv网络宽度，根据《Efficients》，**推荐：3**。

### 实验参数

+ data_id: 用于指定某个数据集划分，当AUTO_PARAM被设置为该参数时，可以用于评估模型。
+ mode: 运行模式。
+ n_uploaders: 上传者数量
+ n_users: 用户数量
+ K: reduced set大小
+ spec: ntk或rbf
+ id: 见上文
+ max_search_num: 取前若干个模型集成，推荐：3

## 项目结构

preprocess中用于加载数据，划分数据集，为上传者训练模型。

utils中提供了随机模型生成工具，集成学习，ntk实现等等。

build_market.py和evaluate_market.py是实验的主要工作流。

benchmark.py中提供了理论上单个学习器的最优性能评测。

