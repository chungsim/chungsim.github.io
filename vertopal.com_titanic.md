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
    duration: 13.119041
    end_time: "2023-04-07T07:05:30.686117"
    environment_variables: {}
    input_path: \_\_notebook\_\_.ipynb
    output_path: \_\_notebook\_\_.ipynb
    parameters: {}
    start_time: "2023-04-07T07:05:17.567076"
    version: 2.4.0
---

::: {#5bae0d58 .cell .code execution_count="1" _cell_guid="a17ad9c0-be84-481c-81c0-27746860558c" _uuid="213e5099-39ab-4bc3-9007-99009f7563d7" collapsed="false" execution="{\"iopub.execute_input\":\"2023-04-07T07:05:28.144418Z\",\"iopub.status.busy\":\"2023-04-07T07:05:28.143998Z\",\"iopub.status.idle\":\"2023-04-07T07:05:28.159092Z\",\"shell.execute_reply\":\"2023-04-07T07:05:28.157884Z\"}" jupyter="{\"outputs_hidden\":false}" papermill="{\"duration\":2.3528e-2,\"end_time\":\"2023-04-07T07:05:28.162121\",\"exception\":false,\"start_time\":\"2023-04-07T07:05:28.138593\",\"status\":\"completed\"}" tags="[]"}
``` python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

::: {.output .stream .stdout}
    /kaggle/input/titanic/train.csv
    /kaggle/input/titanic/test.csv
    /kaggle/input/titanic/gender_submission.csv
:::
:::

::: {#5a43aa0f .cell .code execution_count="2" execution="{\"iopub.execute_input\":\"2023-04-07T07:05:28.169666Z\",\"iopub.status.busy\":\"2023-04-07T07:05:28.168972Z\",\"iopub.status.idle\":\"2023-04-07T07:05:28.213712Z\",\"shell.execute_reply\":\"2023-04-07T07:05:28.212494Z\"}" papermill="{\"duration\":5.1488e-2,\"end_time\":\"2023-04-07T07:05:28.216541\",\"exception\":false,\"start_time\":\"2023-04-07T07:05:28.165053\",\"status\":\"completed\"}" tags="[]"}
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

::: {#25d16f71 .cell .code execution_count="3" execution="{\"iopub.execute_input\":\"2023-04-07T07:05:28.224368Z\",\"iopub.status.busy\":\"2023-04-07T07:05:28.223659Z\",\"iopub.status.idle\":\"2023-04-07T07:05:28.246762Z\",\"shell.execute_reply\":\"2023-04-07T07:05:28.245510Z\"}" papermill="{\"duration\":2.9803e-2,\"end_time\":\"2023-04-07T07:05:28.249366\",\"exception\":false,\"start_time\":\"2023-04-07T07:05:28.219563\",\"status\":\"completed\"}" tags="[]"}
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

::: {#279a9fad .cell .code execution_count="4" execution="{\"iopub.execute_input\":\"2023-04-07T07:05:28.257450Z\",\"iopub.status.busy\":\"2023-04-07T07:05:28.257012Z\",\"iopub.status.idle\":\"2023-04-07T07:05:28.269885Z\",\"shell.execute_reply\":\"2023-04-07T07:05:28.268484Z\"}" papermill="{\"duration\":1.9591e-2,\"end_time\":\"2023-04-07T07:05:28.272295\",\"exception\":false,\"start_time\":\"2023-04-07T07:05:28.252704\",\"status\":\"completed\"}" tags="[]"}
``` python
women = train_data.loc[train_data.Sex == 'female']['Survived']
rate_women = sum(women)/len(women)
print(rate_women)
```

::: {.output .stream .stdout}
    0.7420382165605095
:::
:::

::: {#e2ec728c .cell .code execution_count="5" execution="{\"iopub.execute_input\":\"2023-04-07T07:05:28.280901Z\",\"iopub.status.busy\":\"2023-04-07T07:05:28.279985Z\",\"iopub.status.idle\":\"2023-04-07T07:05:28.289183Z\",\"shell.execute_reply\":\"2023-04-07T07:05:28.287865Z\"}" papermill="{\"duration\":1.5835e-2,\"end_time\":\"2023-04-07T07:05:28.291387\",\"exception\":false,\"start_time\":\"2023-04-07T07:05:28.275552\",\"status\":\"completed\"}" tags="[]"}
``` python
men = train_data.loc[train_data.Sex == 'male']['Survived']
rate_men = sum(men)/len(men)
print(rate_men)
```

::: {.output .stream .stdout}
    0.18890814558058924
:::
:::

::: {#04e21663 .cell .code execution_count="6" execution="{\"iopub.execute_input\":\"2023-04-07T07:05:28.300071Z\",\"iopub.status.busy\":\"2023-04-07T07:05:28.298895Z\",\"iopub.status.idle\":\"2023-04-07T07:05:29.957828Z\",\"shell.execute_reply\":\"2023-04-07T07:05:29.956726Z\"}" papermill="{\"duration\":1.666197,\"end_time\":\"2023-04-07T07:05:29.960829\",\"exception\":false,\"start_time\":\"2023-04-07T07:05:28.294632\",\"status\":\"completed\"}" tags="[]"}
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
