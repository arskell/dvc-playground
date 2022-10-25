import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

import yaml

params = yaml.safe_load(open("params.yaml"))["predict"]

df = pd.read_csv("data/iris.data")

x = df.iloc[:,:4]
y = df.iloc[:,4]

n = params["n_neighbors"]

knn = KNeighborsClassifier(n_neighbors=n).fit(x, y)

score = knn.score(x, y)

f = open("predict.txt", "a")
f.write(str(score))
f.close()
