import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from lazypredict.Supervised import LazyClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv("diabetes.csv")
# profile = ProfileReport(data, title="Report", explorative=True)
# profile.to_file("diabetes_report.html")

target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2024)

print(y_train.dtype)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# clf = RandomForestClassifier(n_estimators=50, criterion="entropy", random_state=33)
# clf.fit(x_train, y_train)
# y_predict = clf.predict(x_test)

param_grid = {
    "n_estimators": [50, 100, 200],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 2, 5, 10]
}
grid_seacrh = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=4, verbose=1, scoring="recall")
grid_seacrh.fit(x_train, y_train)
y_predict = grid_seacrh.predict(x_test)
print(grid_seacrh.best_params_)
print(grid_seacrh.best_score_)
print(classification_report(y_test, y_predict))

# for i, j in zip(y_predict, y_test):
#     print("Predicted value:{}. Acutual value:{}".format(i, j))

# print(classification_report(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))
#
cm = np.array(confusion_matrix(y_test, y_predict, labels=[0, 1]))
dataframe = pd.DataFrame(cm, index=["Not diabetic", "diabetic"], columns=["Predicted healthy", "Predicted diabetic"])
sns.heatmap(dataframe, annot=True)
plt.savefig("diabetes.png")

# clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
# models, predictions = clf.fit(x_train, x_test, y_train, y_test)
# print(models)