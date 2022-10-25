import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/iris.data")

df_train, df_test = train_test_split(df, test_size=0.2)

x = df.iloc[:,:4]
y = df.iloc[:,4]


x_train = df_train.iloc[:,:4]
y_train = df_train.iloc[:,4]

x_test = df_test.iloc[:,:4]
y_test = df_test.iloc[:,4]

dtree = DecisionTreeClassifier(random_state=0).fit(x_train, y_train)

score = dtree.score(x_test, y_test)

print(score)

#f = open("predict.txt", "a")
#f.write(str(score))
#f.close()