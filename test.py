import numpy as np
import torch
def sigmod(x):
    return 1 / (1 + np.exp(-x))
def sigmodDerivative(x):

    return np.multiply(x, np.subtract(1, x))
def forward(weightsA, weightsB, bias):
    # 前向传播
    # 隐层
    neth1 = inputX[0] * weightsA[0][0] + inputX[1] * weightsA[0][1] + bias[0]*weightsA[0][2]
    outh1 = sigmod(neth1)
    print("隐层第一个神经元", neth1, outh1)

    neth2 = inputX[0] * weightsA[1][0] + inputX[1] * weightsA[1][1] + bias[1]*weightsA[1][2]
    outh2 = sigmod(neth2)
    print("隐层第二个神经元", neth2, outh2)

    # 输出层
    neto1 = outh1 * weightsB[0] + outh2 * weightsB[1] + bias[2]* weightsB[2]
    outo1 = sigmod(neto1)
    print("输出层第一个神经元", neto1, outo1)

    # 向量化
    outA = np.array([outh1,outh2,1])
    outB = np.array([outo1])

    Etotal = 0.5 * np.subtract(y, outB) ** 2
    print("误差值：", Etotal)
    return outA, outB

def backpagration(outA, outB):
    LR=0.1
    # 反向传播
    num1 = np.subtract(outB, y)
    num2 = sigmodDerivative(outB)
    inter1 = np.multiply(num1, num2)
    inter2 = np.multiply(inter1, outA)
    deltaB = np.multiply(num1, num2 ,outA)
    print("deltaB：", deltaB)

    deltaA = np.multiply(np.matmul(np.transpose(weightsB), deltaB), sigmodDerivative(outA))
    print("deltaA：", deltaA)

    deltaWB = np.matmul(deltaB.reshape(1, 1), outA.reshape(1, 3))
    print("deltaWB：", deltaWB)

    deltaWA = np.matmul(deltaA.reshape(1, 1), inputX.reshape(1, 3))
    print("deltaWA", deltaWA)

    # 权重参数更新
    weightsB_new = np.subtract(weightsB, deltaWB)
    print("weightsB_new", weightsB_new)

    bias[3] = np.subtract(bias[3], LR*deltaB[1])
    print("biasB", bias[3])
    bias[2] = np.subtract(bias[2], LR*deltaB[0])
    print("biasB", bias[2])

    weightsA_new = np.subtract(weightsA, deltaWA)
    print("weightsA_new", weightsA_new)

    bias[1] = np.subtract(bias[1], LR*deltaA[1])
    print("biasA", bias[1])
    bias[0] = np.subtract(bias[0], LR*deltaA[0])
    print("biasA", bias[0])
    print("all bias", bias)

    return weightsA_new, weightsB_new, bias
if __name__=="__main__":
    # # 初始化数据
    # # 权重参数
    # bias = np.array([1, 1, 1.0, 1.0])
    # weightsA = np.array([[2.5, 1,1.5], [-1.5, -3,2.0]])
    # weightsB = np.array([1.0,0.5,-1.0])
    # # 期望值
    # y = np.array([1])
    # # 输入层
    # inputX = np.array([0, 1.0])
    #
    # print("第一次前向传播")
    # outA, outB = forward(weightsA, weightsB, bias)
    # print("反向传播-参数更新")
    # weightsA_new, weightsB_new, bias = backpagration(outA, outB)
    # # 更新完毕
    # # 验证权重参数--第二次前向传播
    # print("第二次前向传播")
    # forward(weightsA_new, weightsB_new, bias)
    # ta = torch.tensor([1,2,3])
    # torch.reshape(ta)
    torch.nn.Module

    class A(object):
        """"""
        def __init__(self):
            self.print()

        def print(self):
            """"""
            print(1)

    a = A()
