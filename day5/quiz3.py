from sklearn.datasets import load_wine
from sklearn.svm import LinearSVC

wine = load_wine(as_frame=True)
print(wine)
print(wine.data)
print(type(wine.data))
print(wine.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=42)

# pipeline 사용해서
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
svm_sc = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, random_state=42))
])
svm_sc.fit(x_train, y_train)
print(svm_sc.predict(x_train[:10]))
print(svm_sc.score(x_test, y_test))