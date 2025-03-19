# 1. moons 데이터셋을 사용해 결정 트리를 훈련시키고 세밀하게 튜닝하여 그 값을 출력합니다.
# ㄱ. train데이터와 test의 비율은 8:2라고 합니다.
# ㄴ. 결정 트리 파라미터의 max_leaf_node와 min_sample_split을 세밀하게 튜닝합니다.
# 3.튜닝한 파라미터를 이용하여 결정 트리 정확도를 출력합니다.
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


params = {'max_leaf_nodes':range(2,100),
          'min_samples_split':range(2,11,1)}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(x_train, y_train)
gb = gs.best_estimator_
ypred = gb.predict(x_test)
print(ypred)
print(gs.best_params_)
from sklearn.metrics import accuracy_score
print('accuracy', accuracy_score(y_test, ypred))