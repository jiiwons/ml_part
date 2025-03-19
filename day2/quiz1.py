import pandas as pd

wine = pd.read_csv('https://bit.ly/wine-date')
wine.info()
print(wine.head(15))
print(wine['class'].unique())

from sklearn.model_selection import train_test_split

# X = wine.iloc[:, :3].values
# y = wine['class']

## 강사님 코드
X = wine[['alcohol', 'sugar', 'pH']].to_numpy()
y = wine['class'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

## 강사님 코드
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(x_train)

x_train_scaled = ss.transform(x_train)
x_test_scaled = ss.transform(x_test)

# 로지스틱

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train_scaled, y_train)
print('Logistic Regression (train) accuracy: ', lr.score(x_train_scaled, y_train))
print('Logistic Regression (test) accuracy: ', lr.score(x_test_scaled, y_test))


# 결정트리
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train_scaled, y_train)
print('DecisionTree Regression (train) accuracy: ', dt.score(x_train_scaled, y_train))
print('DecisionTree Regression (test) accuracy: ', dt.score(x_test_scaled, y_test))

dt2 = DecisionTreeClassifier(random_state=42, max_depth=3)
dt2.fit(x_train_scaled, y_train)
print('DecisionTree Regression (train) accuracy: ', dt2.score(x_train_scaled, y_train))
print('DecisionTree Regression (test) accuracy: ', dt2.score(x_test_scaled, y_test))
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
plot_tree(dt2)
plt.show()