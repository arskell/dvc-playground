import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df_train = pd.read_csv("data/iris_train.data")
df_test = pd.read_csv("data/iris_test.data")

x_train = df_train.iloc[:,:4]
y_train = df_train.iloc[:,4]

x_test = df_test.iloc[:,:4]
y_test = df_test.iloc[:,4]

dtree = DecisionTreeClassifier(random_state=0).fit(x_train, y_train)

score = dtree.score(x_test, y_test)


f = open("predict.txt", "a")
f.write(str(score))
f.close()