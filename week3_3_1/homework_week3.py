import numpy as np
import random
import time
import cv2
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics

def load_dataset():
    # 导入scikit-learn鸢尾花数据集
    iris = datasets.load_iris()
    print(dir(iris))
    print(iris.data.shape)
    print(iris.feature_names)
    print(iris.target.shape)
    print(np.unique(iris.target))
    idx = (iris.target != 2)  # 二分法，只考虑标签为0和1的样点
    myData = iris.data[idx].astype(np.float32)
    myLabel = iris.target[idx].astype(np.float32)
    print(myData[:, 0])  # 花萼长
    print(myData[:, 1])  # 花萼宽

    # QC数据集
    plt.scatter(myData[:, 0], myData[:, 1], c=myLabel, )
    plt.show()

    # 切分数据第1列
    my_Data_x1 = myData[:, 0]
    print(my_Data_x1)
    # 切分数据第2列
    my_Data_x2 = myData[:, 1]
    print(my_Data_x2)

    # 切分数据第1列和第2列
    my_Data_X1AndX2 = myData[:, 0:2]
    print(my_Data_X1AndX2)

    # 检查数据与标签样点数目是否一致
    print(len(myLabel))
    print(len(my_Data_x1))
    print(len(my_Data_x2))

def inference(w, b, x_list):
    pred_y_list = x_list * w + b
    return pred_y_list

def sigmoid(z):
    new_x = 1 / (1 + np.exp(-z))
    return new_x

def eval_loss(w, b, x_list, gt_y_list):
    pred_y_list = np.array(x_list) * w + b
    sum_loss = 0.5 * (pred_y_list - gt_y_list)**2
    avg_loss = np.mean(sum_loss)
    return avg_loss

def gradient(pred_y_list, gt_y_list, x_list):
    diff_list = pred_y_list - gt_y_list
    dw_list = np.array([diff * x for diff, x in zip(diff_list, x_list)])
    db_list = diff_list
    return dw_list, db_list

def cal_step_gradient(batch_x_list, batch_gt_y_list, w, b, lr):
    pred_y_list = inference(w, b, batch_x_list)
    dw_list, db_list = gradient(pred_y_list, batch_gt_y_list, batch_x_list)
    avg_dw = np.mean(dw_list)
    avg_db = np.mean(db_list)
    w -= lr * avg_dw
    b -= lr * avg_db
    return  w, b

def gen_sample_data(num_sample):
    w = random.randint(0, 10) + random.random()
    b = random.randint(0, 5) + random.random()
    x_list = np.random.random((num_sample)) * 100 * random.random()
    y_list = np.array([(w * x + b + random.random() * random.randint(-1, 100)) for x in x_list])
    print(w, b)
    return x_list, y_list

def show_result(x_list, gt_y_list, w, b):
    plt.figure()
    plt.scatter(x_list, gt_y_list)
    x = np.linspace(np.min(x_list), np.max(x_list), 1000, endpoint=True)
    y = w * x + b
    plt.plot(x, y, color='red')
    plt.title('data and linear regression')
    plt.show()

def train(x_list, gt_y_list, batch_size, lr, max_iter):
    w = 0
    b = 0
    plt.figure()
    loss = []
    lr_0 = lr
    for i in range(max_iter):
        batch_idx = np.random.choice(len(x_list), batch_size)
        batch_x = np.array([x_list[j] for j in batch_idx])
        batch_y = np.array([gt_y_list[j] for j in batch_idx])
        lr = lr_0 * 0.99**i # learning rate decay, to avoid divergence or vibration
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        print('w:{0}, b:{1}, lr:{2}'.format(w, b, lr))
        loss.append(eval_loss(w, b, x_list, gt_y_list))
        print('loss is {}'.format(loss[i]))
    plt.plot(np.linspace(0, max_iter, len(loss)), loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('learning curve')
    plt.show()

    return w, b

if __name__ == '__main__':
    for i in range(1):
        load_dataset()
        # x_list, gt_y_list = gen_sample_data(num_sample=2000)
        # w, b = train(x_list, gt_y_list, batch_size=500, lr=0.001, max_iter=1000)
        # show_result(x_list, gt_y_list, w, b)
        # print(w, b)