from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=500, noise= 0.30, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True, random_state=42
)

bag_clf.fit(x_train, y_train)
y_pred = bag_clf.predict(x_test)
print('bagging accuracy', accuracy_score(y_test, y_pred))

bag_clf2 = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=False, random_state=42
)

bag_clf2.fit(x_train, y_train)
y_pred = bag_clf2.predict(x_test)
print('pasting accuracy', accuracy_score(y_test, y_pred))
