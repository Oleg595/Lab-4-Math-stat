from scipy.integrate import quad
import numpy as np
from scipy.stats import poisson
import math
import matplotlib.pyplot as plt

def normal_distr(x, size):
    result = np.zeros(size)
    for i in range(size):
        result[i] = (1 / math.sqrt(2 * math.pi)) * math.exp(- x[i] ** 2 / 2)
    return result

def standart_Cauchy_distr(x, size):
    result = np.zeros(size)
    for i in range(size):
        result[i] = (1 / math.pi) * (1 / (1 + x[i] ** 2))
    return result

def laplace_distr(x, size):
    result = np.zeros(size)
    for i in range(size):
        result[i] = (1 / math.sqrt(2)) * math.exp(- math.sqrt(2) * math.fabs(x[i]))
    return result

def uniform_distr(x, size):
    result = np.zeros(size)
    for i in range(size):
        if math.fabs(x[i]) <= math.sqrt(3):
            result[i] = 1 / (2 * math.sqrt(3))
        else:
            result[i] = 0
    return result

def puasson_distr(x, size):
    result = np.zeros(size)
    for i in range(size):
        result[i] = (10 ** x[i] / fact(x[i])) * math.exp(-10)
    return result

def normal(size):
    result = np.random.normal(size=size)
    result = sorted(result)
    return result

def standart_Cauchy(size):
    result = np.random.standard_cauchy(size=size)
    result = sorted(result)
    return result

def laplace(size):
    result = np.random.laplace(size=size)
    result = sorted(result)
    return result

def puasson(size):
    result = poisson.rvs(size=size, mu=10)
    result = sorted(result)
    return result

def uniform(size):
    x = range(size)
    result = [np.random.uniform(-(3 ** 0.5), (3 ** 0.5)) for j in x]
    result = sorted(result)
    return result

def normal_func(size):
    a = -4.
    b = 4.
    result = np.zeros(size)
    def func(x):
        return (1. / np.sqrt(2. * np.pi)) * np.exp(-x * x / 2)
    for i in range(0, size):
        result[i] = quad(func, a=a, b=a + i * (b - a) / size)[0]
    return result

def standartCauchy_func(size):
    a = -4.
    b = 4.
    result = np.zeros(size)
    def func(x):
        return (1 / np.pi) * (1 / (x * x + 1))
    for i in range(0, size):
        result[i] = quad(func, a=a, b=a + i * (b - a) / size)[0]
    num = result[size - 1]
    for i in range(0, size):
        result[i] /= num
    return result

def laplace_func(size):
    a = -4.
    b = 4.
    result = np.zeros(size)
    def func(x):
        return (1 / np.sqrt(2)) * np.exp(-np.sqrt(2) * np.fabs(x))
    for i in range(0, size):
        result[i] = quad(func, a=a, b=a + i * (b - a) / size)[0]
    return result

def uniform_func(size):
    a = -4.
    b = 4.
    result = np.zeros(size)
    def func(x):
        if math.fabs(x) <= math.sqrt(3):
            return 1 / (2 * math.sqrt(3))
        else:
            return 0
    for i in range(0, size):
        result[i] = quad(func, a=a, b=a + i * (b - a) / size)[0]
    return result

def fact(x):
    if x > 1:
        return x * fact(x - 1)
    else:
        return 1

def puasson_func(size):
    a = -4.
    b = 4.
    result = np.zeros(size)
    def func(k):
        return (10 ** k / fact(k)) * math.exp(-10)
    for i in range(0, size):
        result[i] = quad(func, a=a, b=a + i * (b - a) / size)[0]
    return result

def points_StatSeries(x):
    result = np.zeros(len(x))
    num = 1. / len(x)
    result[0] = num
    for i in range(1, len(x)):
        if x[i] > x[i - 1]:
            num += 1. / len(x)
            result[i] = num
        else:
            result[i] = num
            num += 1. / len(x)
    return result

def average(x, size):
    num = 0.
    for i in x:
        num += i
    num /= size
    return num

def despertion(x, size):
    y = average(x, size)
    result = 0.
    for i in x:
        result += (i - y) * (i - y)
    result /= size
    return result

def get_Nums(size):
    a = -4.
    b = 4.
    result = np.zeros(size)
    for i in range(0, size):
        result[i] = a + i * (b - a) / size
    return result

def get_Ker(x, nums, coef):
    result = np.zeros(len(nums))
    h = coef * math.sqrt(despertion(x, len(x)))
    h *= 1.06 / (len(x) ** 0.2)
    for i in range(0, len(nums)):
        if i == 0:
            result[i] = 0.
        else:
            result[i] = 0.
        num = 0.
        for j in range(0, len(x)):
            num += K((nums[i] - x[j]) / h)
        num /= (len(x) * h)
        result[i] += num
    return result

def K(u):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(- (u ** 2 / 2))

def in_Cut(x, size):
    result = {}
    j = 0
    for i in range(0, size):
        if x[i] > -4. and x[i] < 4.:
            result[j] = x[i]
            j += 1
    return result

def to_List(data):
    result = np.zeros(len(data))
    for i in range(0, len(data)):
        result[i] = data[i]
    return result

size_arr = {20, 60, 100}
coefs = {0.5, 1., 2.}

for i in size_arr:
    x = standart_Cauchy(i)
    data = in_Cut(x, i)
    y = points_StatSeries(data)
    x = to_List(data)
    plt.step(x, y)
    data = get_Nums(1000)
    y = standartCauchy_func(1000)
    plt.plot(data, y)
    plt.xlim(-4.2, 4.2)
    plt.title('Num points ' + str(i))
    plt.show()

    for coef in coefs:
        data = get_Nums(1000)
        y = get_Ker(x, data, coef)
        plt.plot(data, y)
        y = standart_Cauchy_distr(data, 1000)
        plt.plot(data, y)
        plt.title('Num points ' + str(i) + ' Coeffitient ' + str(coef))
        plt.show()
