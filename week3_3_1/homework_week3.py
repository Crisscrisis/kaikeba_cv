'''
dscription: linear regression and logistic regression
'''
import numpy as np
import random
import time
import cv2
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics

############################################################
# logistic regression
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
    return my_Data_X1AndX2, myLabel

def sigmoid(z):
    pred_y_list = 1 / (1 + np.exp(-z))
    return pred_y_list

def initialize_with_zeros(dim):
    w = np.zeros((dim))
    b = 0
    return w, b

def inference_logistic(w, b, x_list):
    z = np.dot(w, x_list.T) + b
    pred_y_list = sigmoid(z)
    return pred_y_list

def propagate(w, b, x_list, gt_y_list):

    m = x_list.shape[0]

    pred_y_list = inference_logistic(w, b, x_list)
    loss = -(np.sum(gt_y_list * np.log(pred_y_list) + (1 - gt_y_list) * np.log(1 - pred_y_list))) / m

    dz = pred_y_list - gt_y_list
    dw = (np.dot(x_list.T, dz)) / m
    db = np.sum(dz) / m

    grads = {"dw": dw,
             "db": db}

    return grads, loss

def train_logistic(w, b, x_list, gt_y_list, lr, max_iter):
    loss_list = []
    lr_0 = lr
    for i in range(max_iter):
        grads, loss = propagate(w, b, x_list, gt_y_list)
        dw = grads["dw"]
        db = grads["db"]

        w = w - lr * dw
        b = b - lr * db

        # lr = lr_0 * 0.99999 ** i  # learning rate decay, to avoid divergence or vibration

        loss_list.append(loss)
        print("loss: {0}".format(loss))
    params = {"w": w,
              "b": b}
    return params, loss_list

def predict_losigtic(w, b, x_list):
    m = x_list.shape[0]
    y_prediction = np.zeros((1, m))
    pred_y_list = inference_logistic(w, b, x_list)
    for i in range(m):
        if pred_y_list[i] > 0.5:
            y_prediction[0, i] = 1
        if pred_y_list[i] < 0.5:
            y_prediction[0, i] = 0
    return  y_prediction

def logisitic_model(x_train, y_train, x_test, y_test, lr, max_iter):
    # init
    dim = x_train.shape[1]
    w, b = initialize_with_zeros(dim)

    params, loss_list = train_logistic(w, b, x_train, y_train, lr, max_iter)
    w = params["w"]
    b = params["b"]

    predict_train = predict_losigtic(w, b, x_train)
    predict_test = predict_losigtic(w, b, x_test)

    accuracy_train = 1 - np.mean(np.abs(predict_train - y_train))
    accuracy_test = 1 - np.mean(np.abs(predict_test - y_test))

    print("train accuracy: {0}".format(accuracy_train))
    print("test accuracy: {0}".format(accuracy_test))
    plt.figure()
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    x = np.linspace(np.min(x_train[:, 0]), np.max(x_train[:, 0]), 1000, endpoint=True)
    y = (-b - w[0] * x)/w[1]
    plt.plot(x, y, color='red')
    plt.show()


############################################################
# linear regression: gradient descent
def inference_linear(w, b, x_list):
    pred_y_list = x_list * w + b
    return pred_y_list

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
    pred_y_list = inference_linear(w, b, batch_x_list) # linear regression
    # pred_y_list = inference_logistic(w, b, batch_x_list) # logistic regression
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

def train_linear(x_list, gt_y_list, batch_size, lr, max_iter):
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

func = 'logistic regression'

if __name__ == '__main__':
    if func == 'logistic regression':
        data_x1_x2, y_label = load_dataset()
        x_train = data_x1_x2[0:70, :]
        x_test = data_x1_x2[70:, :]
        y_train = y_label[0:70]
        y_test = y_label[70:]
        logisitic_model(x_train, y_train, x_test, y_test, lr=0.001, max_iter=50000)
    elif func == 'linear regression':
        try_times = 1 # times of running linear regression
        for i in range(try_times):
            x_list, gt_y_list = gen_sample_data(num_sample=2000)
            w, b = train_linear(x_list, gt_y_list, batch_size=500, lr=0.001, max_iter=1000)
            show_result(x_list, gt_y_list, w, b)
            print(w, b)