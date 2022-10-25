import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

import yaml

params = yaml.safe_load(open("params.yaml"))["predict"]

df_train = pd.read_csv("data/iris_train.data")
df_test = pd.read_csv("data/iris_test.data")


x_train = df_train.iloc[:,:4]
y_train = df_train.iloc[:,4]


x_test = df_test.iloc[:,:4]
y_test = df_test.iloc[:,4]

n = params["n_neighbors"]

knn = KNeighborsClassifier(n_neighbors=n).fit(x_train, y_train)

score = knn.score(x_test, y_test)

f = open("predict.txt", "a")
f.write(str(score))
f.close()
