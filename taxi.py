from sklearn.model_selection import train_test_split

data = []
with open('iris.data') as f:
    data = f.readlines()
train, test = train_test_split(data, test_size=0.2)
l_lenght = len(data[0].split(','))-1

weights = {}
for line in train:
    line = line.replace("\n","").split(',')
    x = line[:l_lenght]
    y = line[-1]
    if (y not in weights):
        weights[y] = [0]*l_lenght
    for i in range(l_lenght):
        weights[y][i] += float(x[i]) 
print(weights)
#normalizacja
for y in weights:
    E = sum(weights[y])
    for i in range(l_lenght):
        weights[y][i] /= E
print(weights)

for line in test:
    print(line)

for line in test:
    line = line.replace("\n","").split(',')
    x = line[:l_lenght]
    result = {}
    for y in weights:
        prop = 0
        E = 0
        for i in range(l_lenght):
            E += float(x[i])
        for i in range(l_lenght):
            prop += weights[y][i] ** (float(x[i])/E)
        result[prop] = y
    print(line,result[max(result)],line[-1]==result[max(result)])
