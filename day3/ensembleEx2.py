from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


mnist = load_digits()
print(mnist.keys())
print(mnist.DESCR)
print(mnist.data.shape)
print(mnist.target)

x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2,random_state=42)

dtree = DecisionTreeClassifier(max_depth=5, random_state=42)
dtree = dtree.fit(x_train, y_train)
dtree_predicted = dtree.predict(x_test)

knn = KNeighborsClassifier(n_neighbors=299)
knn = knn.fit(x_train, y_train)
knn_predicted = knn.predict(x_test)

svm = SVC(C=0.1, gamma=0.003, probability=True, random_state=42)
svm = svm.fit(x_train, y_train)
svm_predicted = svm.predict(x_test)
print('accuracy')
print('dtree : ', accuracy_score(y_test, dtree_predicted))
print('knn : ', accuracy_score(y_test, knn_predicted))
print('svm : ', accuracy_score(y_test, svm_predicted))

voting_clf = VotingClassifier(
    estimators=[
        ('dt', dtree), ('knn', knn), ('svm', svm)
    ],
    voting='hard'
)
hard_voting_predicted = voting_clf.fit(x_train, y_train).predict(x_test)
print('voting(hard):', accuracy_score(y_test, hard_voting_predicted))

voting_clf = VotingClassifier(
    estimators=[
        ('dt', dtree), ('knn', knn), ('svm', svm)
    ],
    voting='soft'
)
hard_voting_predicted = voting_clf.fit(x_train, y_train).predict(x_test)
print('voting(soft):', accuracy_score(y_test, hard_voting_predicted))


### warning 뜨는데 강사님 코드랑 비교해보기