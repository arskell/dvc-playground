import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/iris.data")

df_train, df_test = train_test_split(df, test_size=0.2)

df_train.to_csv("data/iris_train.data", index = False)
df_test.to_csv("data/iris_test.data", index = False)
