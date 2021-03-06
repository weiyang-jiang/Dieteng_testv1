# test_v1

### 1. 问题分析

- 数据具有6维的特征值和1维的目标值，数据值大部分都处于0-1之间。目标值为连续变量，此类问题为回归问题。需要建立一个多元回归模型用于预测数据。

### 1.数据清洗及预处理

- 删除掉数据中特征值为0所在行（这也可以用整体数据的均值或相邻数据填充）删减了大约50行的数据,删减后为(8243, 7)
- 特征值进行中心化和标准化处理
- 将数据划分为训练集80%测试集20%


### 3. 训练模型选择

- 总共有6维的特征值，每个特征值应有自己的权重值，参数过少会出现欠拟合，过多会出现过拟合。考虑使用三层全连接神经网络（隐藏层为64维包含激活层）
### 4. 损失函数和优化器

- 我将此类回归问题的损失函数定义为MSE（均方误差损失函数）
- 优化器采用SGDM（动量随机梯度下降）

### 5. 评价模型

- 均方差
- R_squared 拟合评价系数

### 6. 训练过程
- 训练参数如下：
- 训练过程设置了随机数种子seed为0,为了通过控制变量保证模型准确的评价
        config = {
            'n_epochs': 3000,  # maximum number of epochs
            'batch_size': 64,  # mini-batch size for dataloader
            'optimizer': 'SGD',  # optimization algorithm
            'optimizer_hparas': {  # hyper-parameters for the optimizer
                'lr': 0.00065,
                "weight_decay": 1e-4,  # Using for L2 Norm
                'momentum': 0.9
            },
            'early_stop': 200,  # early stopping epochs
            'save_path': 'models/model.pth'
        }

- 训练过程设计为最大3000次的迭代，随着梯度下降的进行神经网络不断更新，每次参与迭代的数据集batch大小为64. early_stopping只关注test dataset中的loss值。当有连续200次迭代loss值都没有下降可以认为训练已经陷入critical point，停止训练。

### 7. 训练结果

- loss 曲线 最低值为0.006298770211137754

  ![avatar](https://github.com/weiyang-jiang/Dieteng_testv1/raw/main/models/loss_base.png)
- Predict 曲线

  ![avatar](https://github.com/weiyang-jiang/Dieteng_testv1/raw/main/models/Predict_base.png)
- 均方差为0.07936479201218734, R_suqared 0.40738731622695923


### 8. 模型优化
#### 8.1 PCA降维
- 由于特征量有6个维度，不一定每个维度的数据都会对模型有很大影响，有的特征量可能只有很小的影响，
但是先前方法将所有特征量都涵盖到训练当中（将每个特征量视为等权重训练），这会导致部分无关信息会干扰训练
造成模型不准确。采用PCA主成分分析法，通过求解协方差矩阵的特征值和特征向量，选取最大特征值对应的特征向量(6, 6)
我选取了特征向量中前5个维度，删减后相当于一个(5, 6)维度的特征向量乘以(8243, 6)得到新的数据为shape为(8243, 5)
- loss 曲线 最低值为0.006155287731707711

  ![avatar](https://github.com/weiyang-jiang/Dieteng_testv1/raw/main/models/loss_PCA.png)
  
- Predict 曲线 

  ![avatar](https://github.com/weiyang-jiang/Dieteng_testv1/raw/main/models/Predict_PCA.png)
  
  ![avatar](https://github.com/weiyang-jiang/Dieteng_testv1/raw/main/models/PCAtable.png)
            

#### 8.2 BN层
- 基于PCA降维4维，模型在神经网络上增加两层BN，用于对激活后的参数进行标准化（降低因为激活函数导致的数据极化）， 
这对于结果来说又有提升,预测数据也更为集中
- loss 曲线 最低值为0.006155287731707711

  ![avatar](https://github.com/weiyang-jiang/Dieteng_testv1/raw/main/models/loss_BN.png)
  
- Predict 曲线 

  ![avatar](https://github.com/weiyang-jiang/Dieteng_testv1/raw/main/models/Predict_BN.png)

- 均方差 0.0726433829930089, R2 0.39611345529556274

#### 8.3 神经网络降维
- 可以看到预测数据和真实数据生成的图像，集中在0.8-0.9之间预测较为准确，但是0-0.7的数据几乎无法预测，
我任务这是因为标签值处于0.8-0.9之间的数据量太大，导致训练存在过拟合。先前中间隐藏层为64维，我将其降低为32维
得到的模型效果是目前最优的。
- 均方差 0.07139627942679824, R2 0.41647982597351074

### 9. 总结
- 本次数据分析采用全连接神经网络，SGDM，均方误差损失函数和L2正则项。使用均方差（越小越好）和R2（越大越相关，但不是绝对的）评价训练模型的好坏程度。初次训练结束后，模型的均方差为0.079,R2值为0.40,这是一个较为合理的值（可以很好的预测数据，但并不是最优的）
- 根据7中predict图，大部分数据集中在0.8-0.9之间，部分数据过于离散。这可能是特征量中无关信息干扰所致
- 采用PCA主成分分析法进行降维。提取了主要信息。可以看出将数据降至4维时，模型到达相对最优解。当继续降维会导致有关特征缺失从而使模型无法更好的得到信息。
- BN层和32维度的神经网络证实可以提升模型的表现。

### 10. 可能的提升手段
1. 应该采集目标值分布更为全面的数据，不应该出现数据一边倒的情况
2. 数据不足可以考虑使用对抗网络生成数据，最好生成目标值为【0-0.7】的数据
3. 时间原因我没有做学习率的调参和优化器的选择（Adam也可以考虑使用，SGDM比较适合简单模型）
4. 学习率可以使用交叉验证进行调节（因为训练时间短可以进行学习率网格搜索调参）
5. 关于loss函数无法下降到最优点，可以考虑使用lr-schedule和warm-up跳出critical point
6. 此类模型也可以使用决策树中信息增益去甄别各个特征量对最终目标值的影响。
7. 模型在0-0.7之间无法判断的很好，我觉得可以单独筛选出处于此区间的数据单独训练（也就是设计两套系统（0-0.7为一类）（0.7 - 1为一类））
