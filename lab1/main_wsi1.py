import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
from numdifftools import Hessian
import timeit


class Optymalization:
    EPSILON = 0.00001 # wartosc uzywana do liczenia pochodnych i warunku stopu algorytmu

    function_values = [] # zawiera kolejne wartosci funkcji wystepujace przy uzyciu ostatniego algorytmu

    def __init__(self, f):
        self.f = f

    def get_function_values(self):
        return self.function_values

    def gradient(self, x): # funkcja licząca aproksymowany gradient
        gradient = []
        for i in range(len(x)):
            x_prim = x.copy()
            x_prim[i] += self.EPSILON
            gradient.append((self.f(x_prim) - self.f(x)) / self.EPSILON) # wyliczanie aproksymowanej pochodnej
        return gradient

    def gradient_algorithm(self, x): # szukanie minimum przy użyciu metody gradientu prostego
        x = x.astype(float)
        self.function_values = []
        x2 = x
        step = 0.005 # wartosc kroku
        delta = 1
        while delta > self.EPSILON or delta < -self.EPSILON: # warunek stopu
            x = x2
            self.function_values.append(self.f(x)) # zapisywanie wartosci f(x) w celu pozniejszej analizy
            delta = self.gradient(x)
            delta = np.multiply(delta, step)
            x2 = np.subtract(x,delta) # punkt po mutacji
            delta = np.array(np.subtract(x, x2))
            delta[delta < 0] *= -1
            delta = sum(delta) / len(delta)
        return x2

    def gradient_adaptative_algorithm(self, x): # szukanie minimum przy użyciu metody gradientu prostego z nawrotami
        x = x.astype(float)
        self.function_values = []
        x2 = x
        delta = 1
        step = 0.5 # poczatkowa wartosc kroku
        while delta > self.EPSILON or delta < -self.EPSILON: # warunek stopu
            x = x2
            self.function_values.append(self.f(x)) # zapisywanie wartosci f(x) w celu pozniejszej analizy
            delta = self.gradient(x)
            delta = np.multiply(delta, step)
            x2 = np.subtract(x,delta) # punkt po mutacji
            delta = np.array(np.subtract(x, x2))
            delta[delta < 0] *= -1
            delta = sum(delta) / len(delta)
            if self.f(x2) - self.f(x) > 0: # sprawdzamy czy przeskoczylismy minimum
                step /= 2 # zmiejszenie wartosci kroku w celu polepszeni eksploatacji
        return x2

    def newton_algorithm(self, x): # szukanie minimum przy użyciu metody newtona
        x = x.astype(float)
        self.function_values = []
        x2 = []
        delta = 1
        while delta > self.EPSILON or delta < -self.EPSILON: # warunek stopu
            self.function_values.append(self.f(x)) # zapisywanie wartosci f(x) w celu pozniejszej analizy
            delta = np.dot(self.gradient(x), self.inv_hessian(x))
            x2 = np.subtract(x, delta) # punkt po mutacji
            delta = np.array(np.subtract(x, x2))
            delta[delta < 0] *= -1
            delta = sum(delta) / len(delta)
            x = x2
        return x2

    def inv_hessian(self, x): # liczenie odwrotnosci hesianu
        hf = Hessian(self.f)
        m = hf(x)
        return inv(m)


def f(x, a, n): # zadana funkcja
    e = 0
    for i in range(n):
        e += a ** (i / (n - 1)) * x[i] * x[i]
    return e


def create_value_graph(algorithm): # rysowanie grafu wartosci funkci podczas dzialania algorytmu
    fig, ax = plt.subplots(nrows=1, ncols=2)
    for n in [10, 20]:
        for a in [1, 10, 100]:
            op = Optymalization(lambda x: f(x, a, n))
            x = np.random.randint(-100, 101, n) # punkt startowy
            print(algorithm(op, x))
            ax[int(n / 10 - 1)].plot(
                op.get_function_values()[:],
                label="a = " + str(a),
            )
        ax[int(n / 10 - 1)].legend(loc="upper right")
        ax[int(n / 10 - 1)].set_title("funtion values n = " + str(n))
        ax[int(n / 10 - 1)].set_xlabel("Repetytion")
        ax[int(n / 10 - 1)].set_ylabel("f(x)")
        ax[int(n / 10 - 1)].set_yscale("log")
    plt.show()


def create_time_graph(): # rysowanie grafu czasow dzialania algorytmu
    fig, ax = plt.subplots(nrows=2, ncols=3)
    label = ("gradient", "newton", "gradient_adaptative")
    y_pos = np.arange(len(label))
    for n in [10, 20]:
        for a in [1, 10, 100]:
            time = []
            my_setup = (
                """
from __main__ import Optymalization
from __main__ import f
import numpy as np
op = Optymalization(lambda x : f(x,"""
                + str(a)
                + ""","""
                + str(n)
                + """))"""
            )
            for algorithm in label:
                my_code = (
                    "op."
                    + algorithm
                    + "_algorithm(np.random.randint(-100,101,"
                    + str(n)
                    + "))"
                )
                time.append(timeit.timeit(setup=my_setup, stmt=my_code, number=10)) # pomiar czasu
            x = int(n / 10 - 1)
            y = int(np.log10(a))
            ax[x, y].barh(y_pos, time, align="center", color="green")
            ax[x, y].set_yticks(y_pos)
            if a == 1:
                ax[x, y].set_yticklabels(label)
            if n == 20:
                ax[x, y].set_xlabel("time")
            ax[x, y].set_title("a =" + str(a) + " n =" + str(n))

    plt.show()


if __name__ == "__main__":
    #procedury wywolane do otrzymania grafow w sprawozdaniu

    # create_value_graph(Optymalization.gradient_algorithm)
    # create_value_graph(Optymalization.newton_algorithm)
    # create_value_graph(Optymalization.gradient_adaptative_algorithm)
    # create_time_graph()
    print("done")
