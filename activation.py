import matplotlib.pyplot as plt
import numpy as np


def Sigmoid(x):
    y = np.exp(x) / (np.exp(x) + 1)
    return y


def Tanh(x):
    y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    # y = np.tanh(x)
    return y


def ReLU(x):
    y = np.where(x < 0, 0, x)
    return y


def LeakyReLU(x, a):
    # LeakyReLU的a参数不可训练，人为指定。
    y = np.where(x < 0, a * x, x)
    return y

def ELU(x, a):
    y = np.where(x < 0, x, a*(np.exp(x)-1))
    return y


def PReLU(x, a):
    # PReLU的a参数可训练
    y = np.where(x < 0, a * x, x)
    return y


def ReLU6(x):
    y = np.minimum(np.maximum(x, 0), 6)
    return y


def Swish(x, b):
    y = x * (np.exp(b * x) / (np.exp(b * x) + 1))
    return y


def Mish(x):
    # 这里的Mish已经经过e和ln的约运算
    temp = 1 + np.exp(x)
    y = x * ((temp * temp - 1) / (temp * temp + 1))
    return y


def Grad_Swish(x, b):
    y_grad = np.exp(b * x) / (1 + np.exp(b * x)) + x * (b * np.exp(b * x) / ((1 + np.exp(b * x)) * (1 + np.exp(b * x))))
    return y_grad


def Grad_Mish(x):
    temp = 1 + np.exp(x)
    y_grad = (temp * temp - 1) / (temp * temp + 1) + x * (4 * temp * (temp - 1)) / (
                (temp * temp + 1) * (temp * temp + 1))
    return y_grad


if __name__ == '__main__':
    x = np.arange(-10, 10, 0.01)

    plt.plot(x, Sigmoid(x), color = '#00ff00')
    plt.title("Sigmoid")
    plt.grid()
    plt.savefig("Sigmoid.png", bbox_inches='tight')
    plt.show()

    plt.plot(x, Tanh(x), color = '#00ff00')
    plt.title("Tanh")
    plt.grid()
    plt.savefig("Tanh.png", bbox_inches='tight')
    plt.show()

    plt.plot(x, ReLU(x), color = '#00ff00')
    plt.title("ReLU")
    plt.grid()
    plt.savefig("ReLU.png", bbox_inches='tight')
    plt.show()

    plt.plot(x, LeakyReLU(x, 0.1), color = '#00ff00')
    plt.title("LeakyReLU")
    plt.grid()
    plt.savefig("LeakyReLU.png", bbox_inches='tight')
    plt.show()

    plt.plot(x, ELU(x, 1),  color = '#00ff00')
    plt.title("ELU")
    plt.grid()
    plt.savefig("ELU.png", bbox_inches='tight')
    plt.show()

    plt.plot(x, PReLU(x, 0.25), color = '#00ff00')
    plt.title("PReLU")
    plt.grid()
    plt.savefig("PReLU.png", bbox_inches='tight')
    plt.show()

    plt.plot(x, ReLU6(x), color = '#00ff00')
    plt.title("ReLU6")
    plt.grid()
    plt.savefig("ReLU6.png", bbox_inches='tight')
    plt.show()

    plt.plot(x, Swish(x, 1), color = '#00ff00')
    plt.title("Swish")
    plt.grid()
    plt.savefig("Swish.png", bbox_inches='tight')
    plt.show()

    plt.plot(x, Mish(x), color = '#00ff00')
    plt.title("Mish")
    plt.grid()
    plt.savefig("Mish.png", bbox_inches='tight')
    plt.show()


    plt.plot(x, Grad_Mish(x))
    plt.plot(x, Grad_Swish(x, 1))
    plt.title("Gradient of Mish and Swish")
    plt.legend(['Mish', 'Swish'])
    plt.grid()
    plt.savefig("MSwith.png", bbox_inches='tight')
    plt.show()


