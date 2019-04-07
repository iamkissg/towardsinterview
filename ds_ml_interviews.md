# 数据科学, 机器学习面试

## 目录

[解释机器学习中的偏差和方差，并提出解决他们带来的问题的方法.](#解释机器学习中的偏差和方差，并提出解决他们带来的问题的方法)
[机器学习中的偏差-方差权衡是什么?](#机器学习中的偏差-方差权衡是什么)
[什么是梯度下降?](#什么是梯度下降)
[假设检验的一般步骤.](#假设检验的一般步骤)

## 正文

### 解释机器学习中的偏差和方差，并提出解决他们带来的问题的方法

`偏差`度量了学习算法的期望预测与真实结果的偏离程度, 即刻画了学习算法本身的拟合能力; `方差`度量了同样大小的训练集的变动所导致的学习性能的变化, 即刻画了数据扰动所造成的影响( ;`噪声`表达了在当前任务上任何学习算法所能达到的期望泛化误差的下界, 即刻画了学习问题本身的难度). (西瓜书 2.5)

`Bias` measures the expected deviation from the true value of the function or parameter. `Variance` on the other hand, provides a measure of the deviation from the expected estimator value that any particular sampling of the data is likely to cause. (deeplearningbook 5.4.4)

`Low Bias`: Suggests less assumptions about the form of the target function.
`High Bias`: Suggests more assumptions about the form of the target function.
`Low Variance`: Suggests small changes to the estimate of the target function with changes to the training dataset.
`High Variance`: Suggests large changes to the estimate of the target function with changes to the training dataset. [引用出处](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/)

### 机器学习中的偏差-方差权衡是什么

偏差-方差权衡是机器学习模型的一个属性, 它来源于偏差与方差的冲突: 减小偏差会增大方差, 减小方差会增大偏差, 无法同时减小二者. [归纳自维基百科](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)

泛化性能由学习算法的能力, 数据的充分性以及学习任务本身的难度共同决定. 给定学习任务, 为了取得好的泛化性能, 需要使偏差较小, 即能够充分拟合数据, 并使方差较小, 即使得数据扰动产生的影响小. 一般来说, 偏差和方差是冲突的. 训练不足时, 学习器的拟合能力不够, 训练数据的扰动不足以使学习器产生显著变化, 此时偏差主导了泛化错误率; 随着训练程度的加深, 学习器的拟合能力逐渐增强, 训练数据发生的扰动渐渐能被学习器学到, 方差逐渐主导了泛化错误率; 在训练程度充足后, 学习器的拟合能力已经非常强, 训练数据发生的轻微扰动都会导致学习器发生显著变化, 若训练数据本身, 非全局的特性被学习器学到了, 则将发生过拟合. (西瓜书 2.5)

![bias-variance trade-off](materials/imgs/bias-variance_trade-off.png)
<small>[deeplearningbook Figure5.6](https://www.deeplearningbook.org/contents/ml.html)</small>

The most common way to negotiate this trade-oﬀ is to use **cross-validation**. Alternatively, we can compare the **mean squared error(MSE)** of the estimates: The MSE measures the overall expected deviation between the estimator and the true value of the parameter $\theta$. Evaluating the MSE incorporates both the bias and the variance. Desirable estimators are those with small MSE error and these are estimators that manage to keep both their bias and variance somewhat in check. (deeplearningbook 5.4.4)

判断偏差-方差权衡最常用的方法是**交叉验证**. 另外, 也可以比较这些估计的**均方误差 MSE**, 它度量了估计和真实参数$\theta$之间的平方误差的总体期望偏差. MSE 估计包含了偏差和方差, 理想的估计具有较小的 MSE, 或者在检查中会稍微约束它们的偏差和方差. (深度学习 5.4.4)

### 什么是梯度下降

大多数深度学习算法都涉及某种形式的优化. **优化**指的是, 改变 x 以最大化或最小化某个函数 $f(x)$ 的任务. 将要最大化或最小化的函数称为**目标函数**, 或者**准则 criterion**. 当对其进行最小化时, 也把它称为**代价函数**, **损失函数**或**误差函数**. 利用损失函数的**导数**来指导参数更新, 实现最小化函数的技术就称为**梯度下降 gradient descent**. 导数代表 $f(x)$ 在点 x 处的斜率, 它表明如何缩放输入的小变化才能在输出获得相应的变化. 梯度下降利用了这一特点, 将 x 往导数的反方向移动一小步来减小 $f(x)$. 对于多维输入的函数 (输入向量), 梯度是相对于一个向量求导的导数, 是包含所有偏导数的向量. (汇总自深度学习 4.3)

### 假设检验的一般步骤

> **统计假设检验 (hypothesis test)** 为进行学习器性能比较提供了重要依据. 基于假设检验结果可以推断出, 若在测试集上观察到学习器 A 比 B 好, 则 A 的泛化性能是否在统计意义上优于 B, 以及这个结论的把握有多大. (西瓜书 2.4)

假设检验的一般步骤如下 [维基百科](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing):

1. 研究之初, 假设真相不明;
2. 提出相关的**零假设 null hypothesis**和**备择假设 alternative hypothesis**;
3. 考虑检验中对样本作出的**统计假设 statistical assumptions**, 比如独立性假设 (#???, 不太懂是什么意思);
4. 决定哪个检验是合适的, 并确定相关的**检验统计量 test statistic**;
5. 在零假设下推导检验统计量的分布. 在标准情况下, 应得到一个熟知的结果, 比如检验统计量可能符合正态分布;
6. 选择一个显著性水平, 若低于这个阈值, 就会拒绝零假设, 常用的是 5% 和 1%;
7. 根据在零假设成立时的检验统计量分布, 找到数值最接近备择假设, 且机率为显著性水平的区域 (**拒绝域 critical region**);
8. 针对检验统计量, 根据样本计算其估计值;
9. 若估计值未落在拒绝域, 接受零假设, 否则拒绝零假设.

维基百科上另一种更简短的过程:

1. (提出零假设;)
2. 基于观测, 计算出检验统计量的观测值;
3. 计算 **p值**;
4. 若 p值小于显著性水平, 拒绝零假设, 接受备择假设.

> 对于给定的统计模型, **p值 p-value (probability value)** 是零假设成立的的概率. [维基百科](https://en.wikipedia.org/wiki/P-value)