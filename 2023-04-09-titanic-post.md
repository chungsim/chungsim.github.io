```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```


```python
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
```


```python
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
```


```python
women = train_data.loc[train_data.Sex == 'female']['Survived']
rate_women = sum(women)/len(women)
print(rate_women)
```


```python
men = train_data.loc[train_data.Sex == 'male']['Survived']
rate_men = sum(men)/len(men)
print(rate_men)
```


```python
from sklearn.ensemble import RandomForestClassifier

y = train_data['Survived']

features = ['Pclass', 'Sex', 'SibSp', 'Parch']
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=2023)
model.fit(X, y)
pred = model.predict(X_test)

output = pd.DataFrame({
    'PassengerId' : test_data.PassengerId,
    'Survived' : pred
})
output.to_csv('submission.csv', index = False)
```
