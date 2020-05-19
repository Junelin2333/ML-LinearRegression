import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import linear_model


class LinearRegression(object):

    def __init__(self):
        pass

    @tf.function
    def linear_reg(self, X):
        '''
        Return the predicted value
        '''
        # 语法糖@ 嵌套tf.function() 生成自动图，某些情况下可以加速计算
        return tf.matmul(X, self.W) + self.b

    def dataloader(self, features, labels, mini_batch):
        '''
        input data pipeline
        - Args:
        -- features: features matrix, size =[nsamples,features]
        -- labels: size =[nsamples,1] 
        -- mini_batch: size of a batch
        '''
        # list to array
        features = np.array(features)
        labels = np.array(labels)

        # shuffle the index
        indeces = list(range(len(features)))
        random.shuffle(indeces)

        for i in range(0, len(indeces), mini_batch):
            j = np.array(indeces[i:min(i+mini_batch, len(features))])
            # yield功能可以看做return，但是节省内存，是动态的
            yield features[j], labels[j]

    def L2_Loss(self, y_pred, y_true):
        '''
        Compute L2_Loss, also MeanSquareLoss
        '''
        n = len(y_true)
        y_true = tf.reshape(y_true, (-1, 1))
        y_pred = tf.reshape(y_pred, (-1, 1))

        return tf.matmul(tf.transpose(y_pred - y_true), y_pred - y_true)  / n / 2
        # 下面这行代码会导致收敛速度和上面的差别很大, 原因未知
        # return tf.reduce_mean(tf.math.square(y_pred - y_true)) / 2

    def R2_score(self, y_pred, y_true):
        '''
        Compute average error between y_pred and label
        '''
        return 1 - 2*self.L2_Loss(y_pred,y_true) / np.var(y_true)

    def BGD(self, X, y, lr):
        '''
        Batch Gradient Descent Optimizer
        Compute gradient and updata parameters in the model
        - Args:
        -- X: features matrix
        -- y: labels
        -- lr: learing rate
        '''
        # 要把含参数的所有表达式放在梯度带里面，否则会出现没有梯度的情况
        # GradientTape()  梯度带
        with tf.GradientTape(persistent=True) as t:
            t.watch([self.W, self.b])  # 记录参数
            y_pred = self.linear_reg(X)
            loss = self.L2_Loss(y_pred, y)  # 计算损失

        # 梯度下降, 更新参数
        # assign_sub 相当于 -= , 但张量必须使用assign方法而不是运算符
        self.W.assign_sub(lr * t.gradient(loss, self.W))
        self.b.assign_sub(lr * t.gradient(loss, self.b))

        pass

    def SGD(self, X, y, lr):
        '''
        Stochastic Gradient Descent Optimizer
        Compute gradient and updata parameters in the model
        '''
        nsamples = len(y)
        i = random.randint(0, nsamples-1)

        # 要把含参数计算的所有表达式放在with里面，否则会出现没有梯度的情况
        # GradientTape()  梯度带
        with tf.GradientTape(persistent=True) as t:
            t.watch([self.W, self.b])  # 记录参数
            y_pred = self.linear_reg(X)
            loss = self.L2_Loss(y_pred, y)  # 计算损失

        # 梯度下降, 更新参数
        # assign_sub 相当于 -= , 但张量必须使用assign方法而不是运算符
        self.W.assign_sub(lr * t.gradient(loss, self.W))
        self.b.assign_sub(lr * t.gradient(loss, self.b))

        pass

    def fit(self, features, labels, epochs=20, learning_rate=0.001, mini_batch=20):
        '''
        Training the LinearRegression model
        - Args:
        -- features: features matrix
        -- labels: labels matrix
        -- epochs: training steps
        -- learning_rate: gradient descent step length
        '''
        feature_dim = features.shape[1]
        self.W = tf.Variable(tf.random.normal([feature_dim, 1], stddev=0.01))
        self.b = tf.Variable(tf.zeros(1,))

        score_list = []
        loss_list = []

        for i in range(epochs):
            for X, y in self.dataloader(features, labels, mini_batch):
                # 默认选用了SGD
                self.SGD(X, y, learning_rate)
                #self.BGD(X, y, learning_rate)

            # 计算本次迭代的 总损失 以及 预测值和真实值的平均误差
            # 这里的平均误差可以作为预测准确率的一种考量
            loss = self.L2_Loss(self.linear_reg(features), labels)
            loss_list.append(loss)
            score = self.R2_score(self.linear_reg(features), labels)
            score_list.append(score)
            print('epoch %d: \nloss: %f, score: %f' % (i+1, loss, score))
            
        return loss_list, score_list

    
    def evaluate(self, features, labels):
        '''
        用于测试集的评估
        '''
        y_pred = self.linear_reg(features)
        loss = self.L2_Loss(y_pred,labels)
        score = self.R2_score(y_pred,labels)

        print('evaluate: \nloss: %f, score: %f' % (loss, score))
        return loss, score
        

if __name__ == "__main__":

    # 要求TensorFlow版本为2.x
    print("Your TensorFlow Version is ", tf.__version__)
    if tf.__version__.split('.')[0] == '1' :
        print("ERROR! Please upgrade your TensorFlow to 2.x verison!")
        exit()

    # 生成线性回归数据集 y = wx + b
    w_real = [2.899, -3.475, 7.6621]
    b_real = 4.272
    # 产生特征, 符合正态分布, 标准差为1，共10000个, 维度为3
    features = tf.random.normal((10000, 3), stddev=1)
    labels = features[:,0]*w_real[0] + features[:,1]*w_real[1] + features[:,2]*w_real[2] + b_real 
    # 给 labels 加上噪声数据
    labels += tf.random.normal(labels.shape, stddev=0.01)

    # 分割测试集和训练集
    x_train, y_train = features[:-1000],labels[:-1000]
    x_test, y_test = features[-1000:], labels[-1000:]

    # Hyperparameters
    epochs = 20
    learning_rate = 0.001
    mini_batch = 20

    lrg = LinearRegression()
    train_loss,train_score = lrg.fit(x_train, y_train, epochs, learning_rate, mini_batch)
    lrg.evaluate(x_test, y_test)

    # 绘制训练过程中loss曲线
    ax1 = plt.subplot(2,1,1)
    ax1.plot([i+1 for i in range(epochs)],np.reshape(train_loss,(-1,)))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    # 绘制训练过程中error曲线
    ax2 = plt.subplot(2,1,2)
    ax2.plot([i+1 for i in range(epochs)],np.reshape(train_score,(-1,)))
    plt.xlabel('epochs')
    plt.ylabel('score')
    
    plt.show()

    # 与sklearn里的线性回归训练出来的权重和偏置对比
    lrg2 = linear_model.LinearRegression()
    lrg2.fit(x_train, y_train)
    
    print("--------------------------------")
    print('Using myModel, Weights:{}, bias:{}'.format(lrg.W.numpy().reshape(-1,), lrg.b.numpy()))
    print('Using Sklearn, Weights:{}, bias:{}'.format(lrg2.coef_, [lrg2.intercept_]))
    
