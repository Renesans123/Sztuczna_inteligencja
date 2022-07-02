import numpy as np
from matplotlib import pyplot as plt
import math as m

def q(x): # zadana funkcja
    D = len(x)
    x_linalg_norm_2 = np.linalg.norm(x)**2
    q = 0.5
    q += abs(x_linalg_norm_2-D)**(1/4)
    q += (0.5*x_linalg_norm_2+sum(x))/D
    return q

def sphere_function(x):
    return sum( i*i for i in x)

def log_normal_mr(func,starting_x,params):
    sigma_glob = params['starting_sigma'] if 'starting_sigma' in params else 1 #sigma globalna 
    x_glob = starting_x #x globalne
    D = params['dimensions'] #ilosc wymiarow jest konieczna
    lambd = params['lambd'] if 'lambd' in params else D*5
    u = params['u'] if 'u' in params else int(lambd/4)+1
    maxit = params['maxit'] if 'maxit' in params else D*100
    stop_fitness = params['stop_fitness'] if 'stop_fitness' in params else 0.1

    T = 1/(D**(0.5)) #tau
    x = [[0]*D]*lambd 
    iteration = 0
    idx = None
    fitness = [999999]
    best_x, best_fx = x[0], 999999
    function_values = []
    while iteration<maxit and min(fitness)>stop_fitness:
        fitness = []
        for i in range(lambd):
            E = np.random.normal(0, 1)
            E = np.multiply(E,T) 
            z = np.random.normal(0, 1,D) #np.random.multivariate_normal([0]*D, np.identity(D))
            x[i] = np.add(x_glob,np.multiply(z,sigma_glob))
            fitness.append(func(x[i]))
        idx = np.argsort(fitness) #indeksy najefektywniejszych iksow
        sigma_glob *= m.exp(T*np.random.normal(0, 1))
        x_glob = 1/u * sum(np.array(x)[idx][:u])
        if (best_fx>min(fitness)):
            best_x, best_fx = np.array(x)[idx][0], min(fitness)
        function_values.append(min(fitness))
        iteration += 1
    return best_x, iteration, function_values

def self_adaptation(func,starting_x,params):
    # starting_sigma,D,lambd,u,maxit,stop_fitness
    sigma_glob = params['starting_sigma'] if 'starting_sigma' in params else 0.5 #0.5 #sigma globalna 
    x_glob = starting_x #x globalne
    D = params['dimensions'] #ilosc wymiarow jest konieczna
    lambd = params['lambd'] if 'lambd' in params else D*5
    u = params['u'] if 'u' in params else int(lambd/4)+1
    maxit = params['maxit'] if 'maxit' in params else D*100
    stop_fitness = params['stop_fitness'] if 'stop_fitness' in params else 0.1

    T = 1/(D**(0.5)) #tau
    x = [[0]*D]*lambd 
    sigma = [0]*lambd
    iteration = 0
    idx = None
    fitness = [999999]
    function_values = []
    while iteration<maxit and min(fitness)>stop_fitness:
        fitness = []
        for i in range(lambd):
            E = np.random.normal(0, 1)
            E = np.multiply(E,T) 
            z = np.random.normal(0, 1,D) #np.random.multivariate_normal([0]*D, np.identity(D))
            sigma[i] = sigma_glob*(m.exp(E))
            x[i] = np.add(x_glob,np.multiply(z,sigma[i]))
            fitness.append(func(x[i]))
        idx = np.argsort(fitness) #indeksy najefektywniejszych iksow
        sigma_glob = 1/u * sum(np.array(sigma)[idx][:u])
        x_glob = 1/u * sum(np.array(x)[idx][:u])
        function_values.append(min(fitness))
        iteration += 1
    return np.array(x)[idx][0] , iteration, function_values

def basic_test():
    starting_x = np.random.uniform(-100,100)
    params = dict(starting_sigma = 0.5, dimensions = 10, lambd = 20, maxit = 1000, stop_fitness = 0.1)
    #
    x,iterrations = self_adaptation(q,starting_x,params)[:2]
    print(f"algorytm SA funkcja q \nmin x: {x}\nf(x):{q(x)}\nliczba iteracji:{iterrations}\n")
    #
    x,iterrations = self_adaptation(sphere_function,starting_x,params)[:2]
    print(f"algorytm SA funkcja sferyczna \nmin x: {x}\nf(x):{sphere_function(x)}\nliczba iteracji:{iterrations}\n")
    #
    x,iterrations = log_normal_mr(q,starting_x,params)[:2]
    print(f"algorytm LMR funkcja q \nmin x: {x}\nf(x):{q(x)}\nliczba iteracji:{iterrations}\n")
    #
    x,iterrations = log_normal_mr(sphere_function,starting_x,params)[:2]
    print(f"algorytm LMR funkcja sferyczna \nmin x: {x}\nf(x):{sphere_function(x)}\nliczba iteracji:{iterrations}\n")

def lambda_test(lambd_range,func):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    params = dict(starting_sigma = 0.5, dimensions = 10, lambd = 50, maxit = 1000, stop_fitness = 0.1)
    for n in [0, 1]:
        data = []
        for lambd in lambd_range:
            iterrations = 0
            params[lambd] = lambd
            starting_x = np.random.uniform(-100,100) # punkt startowy
            if n == 0:
                iterrations += sum(self_adaptation(func,starting_x,params)[1] for i in range(20))
            else:
                iterrations += sum(log_normal_mr(func,starting_x,params)[1] for i in range(20))
            iterrations /= 20
            data.append(iterrations)
            print(lambd)
        ax[n].plot(lambd_range, data)
        #ax[n].legend(loc="upper right")
        ax[n].set_title("Self-Adaptation" if n == 0 else "Log-Normal Mutation Rule")
        ax[n].set_xlabel("population")
        ax[n].set_ylabel("iterations")
        ax[n].set_ylim([0,1000])
    plt.show()

def lambda_test2(lambd_range,func):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    params = dict(starting_sigma = 0.5, dimensions = 10, lambd = 50, maxit = 1000, stop_fitness = 0.1)
    for n in [0, 1]:
        data = []
        for lambd in lambd_range:
            convergence = 0
            params[lambd] = lambd
            starting_x = np.random.uniform(-100,100) # punkt startowy
            if n == 0:
                convergence += sum(func(self_adaptation(func,starting_x,params)[0])<0.1 for i in range(10))
            else:
                convergence += sum(func(log_normal_mr(func,starting_x,params)[0])<0.1 for i in range(10))
            data.append(convergence)
            print(lambd)
        ax[n].plot(lambd_range, data)
        ax[n].set_title("Self-Adaptation" if n == 0 else "Log-Normal Mutation Rule")
        ax[n].set_xlabel("population")
        ax[n].set_ylabel("convergence")
        ax[n].set_ylim([0,10])
    plt.show()

def sigma_test(sigma_range,func):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    params = dict(starting_sigma = 0.5, dimensions = 10, lambd = 50, maxit = 1000, stop_fitness = 0.1)
    for n in [0, 1]:
        data = []
        for sigma in sigma_range:
            iterrations = 0
            params['starting_sigma'] = sigma
            starting_x = np.random.uniform(-100,100) # punkt startowy
            if n == 0:
                iterrations += sum(self_adaptation(func,starting_x,params)[1] for i in range(20))
            else:
                iterrations += sum(log_normal_mr(func,starting_x,params)[1] for i in range(20))
            iterrations /= 20
            data.append(iterrations)
            print(sigma)
        ax[n].plot(sigma_range, data)
        #ax[n].legend(loc="upper right")
        ax[n].set_title("Self-Adaptation" if n == 0 else "Log-Normal Mutation Rule")
        ax[n].set_xlabel("starting sigma")
        ax[n].set_ylabel("iterations")
        ax[n].set_ylim([0,1000])
        ax[n].set_xscale("log")
    plt.show()

def track_sa_values():
    fig, ax = plt.subplots(nrows=2, ncols=3)
    starting_x = np.random.uniform(-100,100) # punkt startowy
    params = dict(starting_sigma = 0.5, dimensions = 10, lambd = 20, maxit = 1000, stop_fitness = 0.1)
    for n in [0, 1]:
        data = []
        for i in range(3):
            if n == 0:
                x,iterrations,data = self_adaptation(sphere_function,starting_x,params)
                print(iterrations,x)
            else:
                x,iterrations,data = self_adaptation(q,starting_x,params)
                print(iterrations,x)
            ax[n,i].plot(data)
            if (i == 0):
                ax[n,i].set_ylabel("Sphere function" if n == 0 else "q(x)")
            ax[n,i].set_xlabel("iteration")
            ax[n,i].set_ylim([0,300])
            ax[n,i].set_xlim([0,200])
    plt.show()

def track_lmr_values():
    fig, ax = plt.subplots(nrows=2, ncols=3)
    starting_x = np.random.uniform(-100,100) # punkt startowy
    params = dict(starting_sigma = 0.5, dimensions = 10, lambd = 20, maxit = 1000, stop_fitness = 0.1)
    for n in [0, 1]:
        data = []
        for i in range(3):
            if n == 0:
                x,iterrations,data = log_normal_mr(sphere_function,starting_x,params)
                print(iterrations,x)
            else:
                x,iterrations,data = log_normal_mr(q,starting_x,params)
                print(iterrations,x)
            ax[n,i].plot(data)
            if (i == 0):
                ax[n,i].set_ylabel("Sphere function" if n == 0 else "q(x)")
            ax[n,i].set_xlabel("iteration")
            ax[n,i].set_ylim([0,500])
            ax[n,i].set_xlim([0,500])
    plt.show()


if __name__ == "__main__":
    np.random.seed(100)
    # lambda_test(range(5,100,5),q)
    #lambda_test2(range(5,100,5),q)
    #sigma_test([1 / (2**i) for i in range(0,11)], q)
    track_sa_values()
    #track_lmr_values()

#dangerous space' ­'