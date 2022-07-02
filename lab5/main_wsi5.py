import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
class aproximation:
    # n - liczba aksonów
    # n_layer - liczba warstw
    # u - funkcja aktywacji
    # f - funkcja która ma być odwzorowana
    # D - dziedzina w postaci [x_minimimalne, x_maksymalne]
    # number_of_points liczba punktów trenujących
    def __init__(self,n,layer_n,u=None,f=None,D=None,number_of_points=None):
        self.weights = np.array(range(n*2+(layer_n-2)*(n+1)*n+n+1))/10
        self.n = n
        self.layer_n = layer_n
        if u is None:
            self.u = (lambda y:1 / (1 + np.exp(-y)))
        else:
            self.u = u
        if f is None:
            self.f = (lambda x:(x*x+100*np.cos(x)) * np.sin(x))
        else:
            self.f = f
        self.D = ([-40,40] if D is None else D)
        self.number_of_points = (21 if number_of_points is None else number_of_points)
        self.x = self.given_x()
        self.y = self.given_y()
    # zwraca macież warstwy layer_ind , pierwsza wartwa ma nr 1
    def w(self,layer_ind):
        n = self.n
        #first
        if (layer_ind == 1):
            return self.weights[0:2*n].reshape(2,n)
        #last
        if (layer_ind == self.layer_n):
            ind = 2*n + (self.layer_n-2)*(n+1)*n
            return self.weights[ind:]
        ind = 2*n + (layer_ind-2)*(n+1)*n
        #middle
        return self.weights[ind:ind+(n+1)*n].reshape(n+1,n)
    # zastosowanie funkcji aktywacji
    def transfer(self,y):
        return np.array([self.u(yi) for yi in y])
    # aproksymuje wartość f(x) dla x, przy użyciu wag w sieci
    def calculate(self,x,w=None):
        if (w is not None):
            self.weights = w
        y = x
        for i in range(self.layer_n):
            y = np.append(y,1)
            y = np.dot(y,self.w(i+1))
            if (i+1 != self.layer_n):
                y = self.transfer(y)
        return y
    # funkcja optymalizowana
    # sumuje róznice aproksymacji Y - f(x) po wszystkich iksach oraz dodaje długość wektora wag
    def quality(self,w):
        r = 0
        for xi in self.x:
            i =self.calculate(xi,w) - self.f(xi)
            r += i*i*4
        return r + np.dot(w,w)
    #generuje zbiór iksów uczących
    def given_x(self):
        return np.linspace(self.D[0],self.D[-1],self.number_of_points)
    def given_y(self):
        return np.array([self.f(xi) for xi in self.x])
    # uczy model
    def train(self):
        res = sp.optimize.minimize(self.quality, self.weights, method='BFGS')
        self.weights = res.x
    # wykres :  czerwony - funkcja zadana f(x)
    #           zielony - aproksymacja
    #           czarne punkty - zbiór punktów uczących
    def plot_f(self):
        x = np.linspace(-40,40,100)
        # setting the axes at the centre
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        # plot the function
        plt.plot(x,self.f(x), 'r')
        plt.plot(x,[self.calculate(xi) for xi in x], 'g')
        plt.plot(np.linspace(self.D[0],self.D[-1],self.number_of_points), self.f(np.linspace(self.D[0],self.D[-1],self.number_of_points)), 'o', color='black');
        # show the plot
        plt.show()

#if __name__ == "__main__":
F = lambda x:x*x+100*np.cos(x)
U = lambda y:max(0,y)

apr = aproximation(30,2,u=U,f=F)
apr.train()

apr.plot_f()

# from sklearn.metrics import accuracy_score 
# import numpy as np
# from scipy.misc import derivative
# import matplotlib.pyplot as plt
# import scipy as sp
# class aproximation:
#     # n - liczba aksonów
#     # n_layer - liczba warstw
#     # u - funkcja aktywacji
#     # f - funkcja która ma być odwzorowana
#     # D - dziedzina w postaci [x_minimimalne, x_maksymalne]
#     # number_of_points liczba punktów trenujących
#     def __init__(self,n,layer_n,u=None,f=None,D=None,number_of_points=None):
#         self.weights = np.array(range(n*2+(layer_n-2)*(n+1)*n+n+1))/10
#         self.n = n
#         self.layer_n = layer_n
#         if u is None:
#             self.u = (lambda y:1 / (1 + np.exp(-y)))
#         else:
#             self.u = u
#         if f is None:
#             self.f = (lambda x:(x*x+100*np.cos(x)) * np.sin(x))
#         else:
#             self.f = f
#         self.D = ([-40,40] if D is None else D)
#         self.number_of_points = (21 if number_of_points is None else number_of_points)
#         self.x = self.given_x()
#         self.y = self.given_y()
#         self.u_der = ((lambda x: x) if u is None else (lambda x: x))
#     # zwraca macież warstwy layer_ind , pierwsza wartwa ma nr 1
#     def w(self,layer_ind):
#         n = self.n
#         #first
#         if (layer_ind == 1):
#             return self.weights[0:2*n].reshape(2,n)
#         #last
#         if (layer_ind == self.layer_n):
#             ind = 2*n + (self.layer_n-2)*(n+1)*n
#             return self.weights[ind:]
#         ind = 2*n + (layer_ind-2)*(n+1)*n
#         #middle
#         return self.weights[ind:ind+(n+1)*n].reshape(n+1,n)
#     # zastosowanie funkcji aktywacji
#     def transfer(self,y):
#         return np.array([self.u(yi) for yi in y])
#     # aproksymuje wartość f(x) dla x, przy użyciu wag w sieci
#     def calculate(self,x,w=None):
#         if (w is not None):
#             self.weights = w
#         y = x
#         self.temp_y = []
#         for i in range(self.layer_n):
#             y = np.append(y,1)
#             self.temp_y.append(y)
#             y = np.dot(y,self.w(i+1))
#             if (i+1 != self.layer_n):
#                 y = self.transfer(y)
#         return y
#     # funkcja optymalizowana
#     # sumuje róznice aproksymacji Y - f(x) po wszystkich iksach oraz dodaje długość wektora wag
#     def quality(self,w):
#         r = 0
#         for i in range(len(self.x)):
#             i =self.calculate(self.x[i],w) - self.y[i]
#             r += i*i*4
#         return r + np.dot(w,w)
#     #generuje zbiór iksów uczących
#     def given_x(self):
#         return np.linspace(self.D[0],self.D[-1],self.number_of_points)
#     def given_y(self):
#         return np.array([self.f(xi) for xi in self.x])
#     # uczy model
#     def train(self):
#         res = sp.optimize.minimize(self.quality, self.weights, method='BFGS')
#         self.weights = res.x
#     # wykres :  czerwony - funkcja zadana f(x)
#     #           zielony - aproksymacja
#     #           czarne punkty - zbiór punktów uczących
#     def plot_f(self):
#         x = np.linspace(-40,40,100)
#         # setting the axes at the centre
#         fig = plt.figure()
#         ax = fig.add_subplot(1, 1, 1)
#         ax.spines['left'].set_position('center')
#         ax.spines['bottom'].set_position('center')
#         ax.spines['right'].set_color('none')
#         ax.spines['top'].set_color('none')
#         ax.xaxis.set_ticks_position('bottom')
#         ax.yaxis.set_ticks_position('left')
#         # plot the function
#         plt.plot(x,self.f(x), 'r')
#         plt.plot(x,[self.calculate(xi) for xi in x], 'g')
#         plt.plot(self.x, self.y, 'o', color='black')
#         # show the plot
#         plt.show()

#     def gradient(self,x):
#         grd = []
#         der = []
#         for n in range(self.layer_n):
#             grd.append(self.w(n+1))
#             n = self.layer_n - n
#             if (n == self.layer_n):
#                 der.append(sum(self.w(n)))
#             else:
#                 der.append(der[-1] * sum([sum(row) for row in apr.w(n)]))
#         for i in range(self.layer_n):
#             for j in range(len(self.temp_y[i])):
#                 grd[i][j] = self.temp_y[i][j]

#             grd[i] *= der[i]
#         grd_end = grd[-1]
#         grd = [item for sublist in grd[:-1] for item in sublist]
#         grd = [item for sublist in grd for item in sublist]
#         return [*grd , *grd_end]

#     def gradient_adaptative_algorithm(self): # szukanie minimum przy użyciu metody gradientu prostego z nawrotami
#             self.EPSILON = 0.001
#             x = self.weights
#             self.calculate([-40],x)
#             self.function_values = []
#             x2 = x
#             delta = 1
#             step = 0.5 # poczatkowa wartosc kroku
#             i = 2000

#             while i>0 and (delta > self.EPSILON or delta < -self.EPSILON): # warunek stopu
#                 i -= 1
#                 x = x2
#                 delta = self.gradient(x)
#                 delta = np.multiply(delta, step)
#                 x2 = np.subtract(x,delta) # punkt po mutacji
#                 delta = np.array(np.subtract(x, x2))
#                 delta[delta < 0] *= -1
#                 delta = sum(delta) / len(delta)
#                 if self.quality(x2) - self.quality(x) > 0: # sprawdzamy czy przeskoczylismy minimum
#                     step /= 2 # zmiejszenie wartosci kroku w celu polepszeni eksploatacji
#             self.weights = x2
#             return x2

        

# def f(x):
#     return (x*x+100*np.cos(x)) * np.sin(x)

# if __name__ == "__main__":
#     F = lambda x:x*x+100*np.cos(x)
#     U = lambda y:max(0,y)

#     apr = aproximation(5,2,u=U,f=F)
#     apr.gradient_adaptative_algorithm()
#     #print(apr.x,apr.y)
#     #apr.train()

#     apr.plot_f()
    
#     #kolejne macierzee wag:
#     for i in range(apr.layer_n):
#         print(i+1,"i",apr.w(i+1),"i/n")
#     print("q",apr.quality(apr.weights))
#     #print([sum(row) for row in apr.w(1)])
    
#     #apr.calculate(apr.x[0])
#     #print(apr.temp_y)
#     #print([i for i in range(apr.layer_n,0,-1)])
#     #print(apr.gradient(apr.x[0]))
#     #print(apr.gradient(apr.x[0]))