import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn import svm # for pyplot only

class SVM:

    def __init__(self,X,Y,LAMBDA):
        #noramlized X
        X_normalized = MinMaxScaler().fit_transform(X.values)
        X = pd.DataFrame(X_normalized)
        #added constant value = 1 to X
        X.insert(loc=len(X.columns), column='const', value=1)
        #separate training and testing data
        sep_data = train_test_split(X, Y, test_size=0.2, random_state=23)
        self.X_train, self.X_test, self.Y_train, self.Y_test = sep_data
        self.W = np.zeros(X.shape[1])
        self.kernel = self.linear
        self.LAMBDA = LAMBDA

    def compute_cost(self,W):
        N = self.X_train.shape[0]
        # max(0 ; 1 - y_i * f(x_i))
        i_expr = self.kernel(self.X_train, W)
        i_expr = i_expr * -self.Y_train +1
        i_expr[i_expr < 0] = 0
        # E_i (max(0 ; 1 - y_i * f(x_i)))
        E_i = np.sum(i_expr)
        # lambda/2 * ||w||^2
        w_expr = np.dot(W, W) * 0.5 * self.LAMBDA
        return w_expr + E_i / N
    
    def linear(self, X, W):
        return np.dot(X , W)

    def poly(self, X, W):
        if(X.shape[1] != len(W)):
            N = X.shape[1] - 1
            for i in range(N):
                X[str(i)+"^2"] = X[i] * X[i]
        return np.dot(X , W)

    def train(self, kernel):
        # set kernel
        if (kernel=='linear'):
            self.kernel = self.linear
            self.W = np.zeros(self.X_train.shape[1])
        elif(kernel=='poly'):
            self.kernel = self.poly
            self.W = np.zeros(self.X_train.shape[1]*2-1)
        else:
            return
        # train
        res = sp.optimize.minimize(self.compute_cost, self.W, method='BFGS')
        self.W = res.x
    
    def test(self):
            # predict quality
            y_test_predicted = self.kernel(self.X_test, self.W)
            y_test_predicted[y_test_predicted < 0] = -1
            y_test_predicted[y_test_predicted > 0] = 1
            # compare and print accuracy
            print("Accuracy:", accuracy_score(self.Y_test, y_test_predicted))

def get_data(csv_file_name):
        #read csv
        data = pd.read_csv("./"+csv_file_name+".csv", sep=';')
        #separate X and Y    
        Y = data[data.columns[-1]]
        X = data.iloc[:, :-1]
        return X, Y

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plot_svm(X,y,W,kernel):
    # prep input data
    X = X.iloc[:, [np.argmin(W),np.argmax(W)]].to_numpy()
    y = y.to_numpy()
    # model for graf
    model = svm.SVC(kernel=kernel)
    clf = model.fit(X, y)

    fig, ax = plt.subplots()
    # title for the plots
    title = ('Decision surface of '+kernel+' SVC ')
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    print(svm_machine.W)
    plt.show()

def plot_Y(Y):
    plt.hist(Y)
    plt.plot()
    plt.show()

if __name__ == "__main__":
    X, Y = get_data("winequality-red")
    #normalize quality
    Y[Y <= 5] = -1
    Y[Y > 5] = 1

    svm_machine = SVM(X,Y,0.025)   
    # train with linear kernel
    svm_machine.train('linear')
    svm_machine.test()
    plot_svm(svm_machine.X_test,svm_machine.Y_test,svm_machine.W,'linear')
    # # train with polynomial kernel
    # svm_machine.train('poly')
    # svm_machine.test()
    # plot_svm(svm_machine.X_test,svm_machine.Y_test,svm_machine.W,'poly')
    print("done")