from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=500, noise= 0.30, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

# log_clf = LogisticRegression(random_state=42)
# rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
# svm_clf = SVC(random_state=42)
#
# voting_clf = VotingClassifier(
#     estimators=[('lr', log_clf), ('rf', rnd_clf),('svc', svm_clf)],
#     voting='hard'
# )
#
#
# for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
#     clf.fit(x_train, y_train)
#     y_pred = clf.predict(x_test)
#     print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# 간접 투표
log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(probability=True, random_state=42)

# soft방식을 못 쓰는 모델들도 있음
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf),('svc', svm_clf)],
    voting='soft'
)


for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))