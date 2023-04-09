---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.7.12
  nbformat: 4
  nbformat_minor: 5
  papermill:
    default_parameters: {}
    duration: 15.018739
    end_time: "2023-04-07T14:40:08.919559"
    environment_variables: {}
    input_path: \_\_notebook\_\_.ipynb
    output_path: \_\_notebook\_\_.ipynb
    parameters: {}
    start_time: "2023-04-07T14:39:53.900820"
    version: 2.4.0
---

::: {#6ffa3c6f .cell .code execution_count="1" _cell_guid="a17ad9c0-be84-481c-81c0-27746860558c" _uuid="213e5099-39ab-4bc3-9007-99009f7563d7" collapsed="false" execution="{\"iopub.execute_input\":\"2023-04-07T14:40:05.532170Z\",\"iopub.status.busy\":\"2023-04-07T14:40:05.531688Z\",\"iopub.status.idle\":\"2023-04-07T14:40:05.546589Z\",\"shell.execute_reply\":\"2023-04-07T14:40:05.545728Z\"}" jupyter="{\"outputs_hidden\":false}" papermill="{\"duration\":2.6686e-2,\"end_time\":\"2023-04-07T14:40:05.550323\",\"exception\":false,\"start_time\":\"2023-04-07T14:40:05.523637\",\"status\":\"completed\"}" tags="[]"}
``` python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

::: {.output .stream .stdout}
    /kaggle/input/titanic/train.csv
    /kaggle/input/titanic/test.csv
    /kaggle/input/titanic/gender_submission.csv
:::
:::

::: {#9f9b7c35 .cell .code execution_count="2" execution="{\"iopub.execute_input\":\"2023-04-07T14:40:05.561380Z\",\"iopub.status.busy\":\"2023-04-07T14:40:05.560941Z\",\"iopub.status.idle\":\"2023-04-07T14:40:05.621661Z\",\"shell.execute_reply\":\"2023-04-07T14:40:05.620518Z\"}" papermill="{\"duration\":7.0091e-2,\"end_time\":\"2023-04-07T14:40:05.625303\",\"exception\":false,\"start_time\":\"2023-04-07T14:40:05.555212\",\"status\":\"completed\"}" tags="[]"}
``` python
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
```

::: {.output .execute_result execution_count="2"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#6b946fd5 .cell .code execution_count="3" execution="{\"iopub.execute_input\":\"2023-04-07T14:40:05.636981Z\",\"iopub.status.busy\":\"2023-04-07T14:40:05.636329Z\",\"iopub.status.idle\":\"2023-04-07T14:40:05.663556Z\",\"shell.execute_reply\":\"2023-04-07T14:40:05.662116Z\"}" papermill="{\"duration\":3.6882e-2,\"end_time\":\"2023-04-07T14:40:05.666860\",\"exception\":false,\"start_time\":\"2023-04-07T14:40:05.629978\",\"status\":\"completed\"}" tags="[]"}
``` python
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
```

::: {.output .execute_result execution_count="3"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#e2dfb4b4 .cell .code execution_count="4" execution="{\"iopub.execute_input\":\"2023-04-07T14:40:05.678784Z\",\"iopub.status.busy\":\"2023-04-07T14:40:05.678249Z\",\"iopub.status.idle\":\"2023-04-07T14:40:05.693229Z\",\"shell.execute_reply\":\"2023-04-07T14:40:05.691796Z\"}" papermill="{\"duration\":2.4745e-2,\"end_time\":\"2023-04-07T14:40:05.696372\",\"exception\":false,\"start_time\":\"2023-04-07T14:40:05.671627\",\"status\":\"completed\"}" tags="[]"}
``` python
women = train_data.loc[train_data.Sex == 'female']['Survived']
rate_women = sum(women)/len(women)
print(rate_women)
```

::: {.output .stream .stdout}
    0.7420382165605095
:::
:::

::: {#9e3ef264 .cell .code execution_count="5" execution="{\"iopub.execute_input\":\"2023-04-07T14:40:05.708815Z\",\"iopub.status.busy\":\"2023-04-07T14:40:05.708268Z\",\"iopub.status.idle\":\"2023-04-07T14:40:05.718334Z\",\"shell.execute_reply\":\"2023-04-07T14:40:05.716252Z\"}" papermill="{\"duration\":2.0605e-2,\"end_time\":\"2023-04-07T14:40:05.721903\",\"exception\":false,\"start_time\":\"2023-04-07T14:40:05.701298\",\"status\":\"completed\"}" tags="[]"}
``` python
men = train_data.loc[train_data.Sex == 'male']['Survived']
rate_men = sum(men)/len(men)
print(rate_men)
```

::: {.output .stream .stdout}
    0.18890814558058924
:::
:::

::: {#48dc21e5 .cell .code execution_count="6" execution="{\"iopub.execute_input\":\"2023-04-07T14:40:05.733493Z\",\"iopub.status.busy\":\"2023-04-07T14:40:05.732883Z\",\"iopub.status.idle\":\"2023-04-07T14:40:07.684132Z\",\"shell.execute_reply\":\"2023-04-07T14:40:07.682577Z\"}" papermill="{\"duration\":1.961942,\"end_time\":\"2023-04-07T14:40:07.687820\",\"exception\":false,\"start_time\":\"2023-04-07T14:40:05.725878\",\"status\":\"completed\"}" tags="[]"}
``` python
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
:::
