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
    version: 3.6.6
  nbformat: 4
  nbformat_minor: 4
---
# **Logistic Regression Classifier Tutorial with Python**

안녕, 친구들,

이 커널에서 저는 Python과 Scikit-Learn으로 로지스틱 회귀 분석을 구현합니다. 저는 내일 호주에 비가 올지 여부를 예측하기 위해 로지스틱 회귀 분류기를 만듭니다. 로지스틱 회귀 분석을 사용하여 이항 분류 모델을 교육합니다.

# **Table of Contents**

1.  [Introduction to Logistic Regression](#1)
2.  [Logistic Regression intuition](#2)
3.  [Assumptions of Logistic Regression](#3)
4.  [Types of Logistic Regression](#4)
5.  [Import libraries](#5)
6.  [Import dataset](#6)
7.  [Exploratory data analysis](#7)
8.  [Declare feature vector and target variable](#8)
9.  [Split data into separate training and test set](#9)
10. [Feature engineering](#10)
11. [Feature scaling](#11)
12. [Model training](#12)
13. [Predict results](#13)
14. [Check accuracy score](#14)
15. [Confusion matrix](#15)
16. [Classification metrices](#16)
17. [Adjusting the threshold level](#17)
18. [ROC - AUC](#18)
19. [k-Fold Cross Validation](#19)
20. [Hyperparameter optimization using GridSearch CV](#20)
21. [Results and conclusion](#21)
22. [References](#22)


# **1. Introduction to Logistic Regression** `<a class="anchor" id="1">`{=html}`</a>`{=html} {#1-introduction-to-logistic-regression-}

데이터 과학자들이 새로운 분류 문제를 발견할 수 있는 경우, 가장 먼저 떠오르는 알고리즘은 로지스틱 회귀 분석입니다. 개별 클래스 집합에 대한 관찰을 예측하는 데 사용되는 지도 학습 분류 알고리즘입니다. 실제로 관측치를 여러 범주로 분류하는 데 사용됩니다. 따라서, 그것의 출력은 본질적으로 별개입니다. 로지스틱 회귀 분석을 로짓 회귀 분석이라고도 합니다. 분류 문제를 해결하는 데 사용되는 가장 단순하고 간단하며 다용도의 분류 알고리즘 중 하나입니다.

::: {.cell .markdown}
# **2. Logistic Regression intuition** `<a class="anchor" id="2">`{=html}`</a>`{=html} {#2-logistic-regression-intuition-}

통계학에서 로지스틱 회귀 모형은 주로 분류 목적으로 사용되는 널리 사용되는 통계 모형입니다. 즉, 관측치 집합이 주어지면 로지스틱 회귀 알고리즘을 사용하여 관측치를 두 개 이상의 이산 클래스로 분류할 수 있습니다. 따라서 대상 변수는 본질적으로 이산적입니다.

로지스틱 회귀 분석 알고리즘은 다음과 같이 작동합니다

## **Implement linear equation**

로지스틱 회귀 분석 알고리즘은 반응 값을 예측하기 위해 독립 변수 또는 설명 변수가 있는 선형 방정식을 구현하는 방식으로 작동합니다. 예를 들어, 우리는 공부한 시간의 수와 시험에 합격할 확률의 예를 고려합니다. 여기서 연구된 시간 수는 설명 변수이며 x1로 표시됩니다. 합격 확률은 반응 변수 또는 목표 변수이며 z로 표시됩니다.

만약 우리가 하나의 설명 변수(x1)와 하나의 반응 변수(z)를 가지고 있다면, 선형 방정식은 다음과 같은 방정식으로 수학적으로 주어질 것입니다

    z = β0 + β1x1    

여기서 계수 β0과 β1은 모형의 모수입니다.

설명 변수가 여러 개인 경우, 위의 방정식은 다음과 같이 확장될 수 있습니다

    z = β0 + β1x1+ β2x2+……..+ βnxn

여기서 계수 β0, β1, β2 및 βn은 모델의 매개변수입니다.

따라서 예측 반응 값은 위의 방정식에 의해 주어지며 z로 표시됩니다.

## **Sigmoid Function**

z로 표시된 이 예측 반응 값은 0과 1 사이에 있는 확률 값으로 변환됩니다. 우리는 예측 값을 확률 값에 매핑하기 위해 시그모이드 함수를 사용합니다. 그런 다음 이 시그모이드 함수는 실제 값을 0과 1 사이의 확률 값으로 매핑합니다.

기계 학습에서 시그모이드 함수는 예측을 확률에 매핑하는 데 사용됩니다. 시그모이드 함수는 S자형 곡선을 가지고 있습니다. 그것은 시그모이드 곡선이라고도 불립니다.

Sigmoid 함수는 로지스틱 함수의 특수한 경우입니다. 그것은 다음과 같은 수학 공식에 의해 주어집니다.

다음 그래프로 시그모이드 함수를 그래픽으로 표현할 수 있습니다.

### Sigmoid Function {#sigmoid-function}

![Sigmoid
Function](vertopal_ca93cbc8c7524834a21e37d8a1959b04/3d802ba40a4e95f9c3e7ba257488dc4da0cd5fe4.png)
:::

::: {.cell .markdown}
## **Decision boundary**

시그모이드 함수는 0과 1 사이의 확률 값을 반환합니다. 그런 다음 이 확률 값은 "0" 또는 "1"인 이산 클래스에 매핑됩니다. 이 확률 값을 이산 클래스(통과/실패, 예/아니오, 참/거짓)에 매핑하기 위해 임계값을 선택합니다. 이 임계값을 의사결정 경계라고 합니다. 이 임계값을 초과하면 확률 값을 클래스 1에 매핑하고 클래스 0에 매핑합니다.

수학적으로 다음과 같이 표현할 수 있습니다

p ≥ 0.5 =\> class = 1

p \< 0.5 =\> class = 0

Generally, the decision boundary is set to 0.5. So, if the probability
value is 0.8 (\> 0.5), we will map this observation to class 1.
Similarly, if the probability value is 0.2 (\< 0.5), we will map this
observation to class 0. This is represented in the graph below-
:::

::: {.cell .markdown}
![Decision boundary in sigmoid
function](vertopal_ca93cbc8c7524834a21e37d8a1959b04/b7f4443f8c2153c2c7b994d7b8f21de575acf06f.png)
:::

::: {.cell .markdown}
## **Making predictions**

Now, we know about sigmoid function and decision boundary in logistic
regression. We can use our knowledge of sigmoid function and decision
boundary to write a prediction function. A prediction function in
logistic regression returns the probability of the observation being
positive, Yes or True. We call this as class 1 and it is denoted by
P(class = 1). If the probability inches closer to one, then we will be
more confident about our model that the observation is in class 1,
otherwise it is in class 0.
:::

::: {.cell .markdown}
# **3. Assumptions of Logistic Regression** `<a class="anchor" id="3">`{=html}`</a>`{=html} {#3-assumptions-of-logistic-regression-}

[Table of Contents](#0.1)

The Logistic Regression model requires several key assumptions. These
are as follows:-

1.  Logistic Regression model requires the dependent variable to be
    binary, multinomial or ordinal in nature.

2.  It requires the observations to be independent of each other. So,
    the observations should not come from repeated measurements.

3.  Logistic Regression algorithm requires little or no
    multicollinearity among the independent variables. It means that the
    independent variables should not be too highly correlated with each
    other.

4.  Logistic Regression model assumes linearity of independent variables
    and log odds.

5.  The success of Logistic Regression model depends on the sample
    sizes. Typically, it requires a large sample size to achieve the
    high accuracy.
:::

::: {.cell .markdown}
# **4. Types of Logistic Regression** `<a class="anchor" id="4">`{=html}`</a>`{=html} {#4-types-of-logistic-regression-}

[Table of Contents](#0.1)

Logistic Regression model can be classified into three groups based on
the target variable categories. These three groups are described below:-

### 1. Binary Logistic Regression {#1-binary-logistic-regression}

In Binary Logistic Regression, the target variable has two possible
categories. The common examples of categories are yes or no, good or
bad, true or false, spam or no spam and pass or fail.

### 2. Multinomial Logistic Regression {#2-multinomial-logistic-regression}

In Multinomial Logistic Regression, the target variable has three or
more categories which are not in any particular order. So, there are
three or more nominal categories. The examples include the type of
categories of fruits - apple, mango, orange and banana.

### 3. Ordinal Logistic Regression {#3-ordinal-logistic-regression}

In Ordinal Logistic Regression, the target variable has three or more
ordinal categories. So, there is intrinsic order involved with the
categories. For example, the student performance can be categorized as
poor, average, good and excellent.
:::

::: {.cell .markdown}
# **5. Import libraries** `<a class="anchor" id="5">`{=html}`</a>`{=html} {#5-import-libraries-}

[Table of Contents](#0.1)
:::

::: {.cell .code execution_count="1"}
``` python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```

::: {.output .stream .stdout}
    /kaggle/input/weather-dataset-rattle-package/weatherAUS.csv
:::
:::

::: {.cell .code execution_count="2"}
``` python
import warnings

warnings.filterwarnings('ignore')
```
:::

::: {.cell .markdown}
# **6. Import dataset** `<a class="anchor" id="6">`{=html}`</a>`{=html} {#6-import-dataset-}

[Table of Contents](#0.1)
:::

::: {.cell .code execution_count="3"}
``` python
data = '/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv'

df = pd.read_csv(data)
```
:::

::: {.cell .markdown}
# **7. Exploratory data analysis** `<a class="anchor" id="7">`{=html}`</a>`{=html} {#7-exploratory-data-analysis-}

[Table of Contents](#0.1)

Now, I will explore the data to gain insights about the data.
:::

::: {.cell .code execution_count="4"}
``` python
# view dimensions of dataset

df.shape
```

::: {.output .execute_result execution_count="4"}
    (142193, 24)
:::
:::

::: {.cell .markdown}
We can see that there are 142193 instances and 24 variables in the data
set.
:::

::: {.cell .code execution_count="5"}
``` python
# preview the dataset

df.head()
```

::: {.output .execute_result execution_count="5"}
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
      <th>Date</th>
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>...</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday</th>
      <th>RISK_MM</th>
      <th>RainTomorrow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-12-01</td>
      <td>Albury</td>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>44.0</td>
      <td>W</td>
      <td>...</td>
      <td>22.0</td>
      <td>1007.7</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>No</td>
      <td>0.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-12-02</td>
      <td>Albury</td>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WNW</td>
      <td>44.0</td>
      <td>NNW</td>
      <td>...</td>
      <td>25.0</td>
      <td>1010.6</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>No</td>
      <td>0.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-12-03</td>
      <td>Albury</td>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WSW</td>
      <td>46.0</td>
      <td>W</td>
      <td>...</td>
      <td>30.0</td>
      <td>1007.6</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>No</td>
      <td>0.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-12-04</td>
      <td>Albury</td>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NE</td>
      <td>24.0</td>
      <td>SE</td>
      <td>...</td>
      <td>16.0</td>
      <td>1017.6</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>No</td>
      <td>1.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-12-05</td>
      <td>Albury</td>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>41.0</td>
      <td>ENE</td>
      <td>...</td>
      <td>33.0</td>
      <td>1010.8</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>No</td>
      <td>0.2</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="6"}
``` python
col_names = df.columns

col_names
```

::: {.output .execute_result execution_count="6"}
    Index(['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
           'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
           'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
           'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
           'Temp3pm', 'RainToday', 'RISK_MM', 'RainTomorrow'],
          dtype='object')
:::
:::

::: {.cell .markdown}
### Drop RISK_MM variable

It is given in the dataset description, that we should drop the
`RISK_MM` feature variable from the dataset description. So, we should
drop it as follows-
:::

::: {.cell .code execution_count="7"}
``` python
df.drop(['RISK_MM'], axis=1, inplace=True)
```
:::

::: {.cell .code execution_count="8"}
``` python
# view summary of dataset

df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 142193 entries, 0 to 142192
    Data columns (total 23 columns):
    Date             142193 non-null object
    Location         142193 non-null object
    MinTemp          141556 non-null float64
    MaxTemp          141871 non-null float64
    Rainfall         140787 non-null float64
    Evaporation      81350 non-null float64
    Sunshine         74377 non-null float64
    WindGustDir      132863 non-null object
    WindGustSpeed    132923 non-null float64
    WindDir9am       132180 non-null object
    WindDir3pm       138415 non-null object
    WindSpeed9am     140845 non-null float64
    WindSpeed3pm     139563 non-null float64
    Humidity9am      140419 non-null float64
    Humidity3pm      138583 non-null float64
    Pressure9am      128179 non-null float64
    Pressure3pm      128212 non-null float64
    Cloud9am         88536 non-null float64
    Cloud3pm         85099 non-null float64
    Temp9am          141289 non-null float64
    Temp3pm          139467 non-null float64
    RainToday        140787 non-null object
    RainTomorrow     142193 non-null object
    dtypes: float64(16), object(7)
    memory usage: 25.0+ MB
:::
:::

::: {.cell .markdown}
### Types of variables

In this section, I segregate the dataset into categorical and numerical
variables. There are a mixture of categorical and numerical variables in
the dataset. Categorical variables have data type object. Numerical
variables have data type float64.

First of all, I will find categorical variables.
:::

::: {.cell .code execution_count="9"}
``` python
# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
```

::: {.output .stream .stdout}
    There are 7 categorical variables

    The categorical variables are : ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
:::
:::

::: {.cell .code execution_count="10"}
``` python
# view the categorical variables

df[categorical].head()
```

::: {.output .execute_result execution_count="10"}
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
      <th>Date</th>
      <th>Location</th>
      <th>WindGustDir</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-12-01</td>
      <td>Albury</td>
      <td>W</td>
      <td>W</td>
      <td>WNW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-12-02</td>
      <td>Albury</td>
      <td>WNW</td>
      <td>NNW</td>
      <td>WSW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-12-03</td>
      <td>Albury</td>
      <td>WSW</td>
      <td>W</td>
      <td>WSW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-12-04</td>
      <td>Albury</td>
      <td>NE</td>
      <td>SE</td>
      <td>E</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-12-05</td>
      <td>Albury</td>
      <td>W</td>
      <td>ENE</td>
      <td>NW</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
### Summary of categorical variables

-   There is a date variable. It is denoted by `Date` column.

-   There are 6 categorical variables. These are given by `Location`,
    `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday` and
    `RainTomorrow`.

-   There are two binary categorical variables - `RainToday` and
    `RainTomorrow`.

-   `RainTomorrow` is the target variable.
:::

::: {.cell .markdown}
## Explore problems within categorical variables

First, I will explore the categorical variables.

### Missing values in categorical variables
:::

::: {.cell .code execution_count="11"}
``` python
# check missing values in categorical variables

df[categorical].isnull().sum()
```

::: {.output .execute_result execution_count="11"}
    Date                0
    Location            0
    WindGustDir      9330
    WindDir9am      10013
    WindDir3pm       3778
    RainToday        1406
    RainTomorrow        0
    dtype: int64
:::
:::

::: {.cell .code execution_count="12"}
``` python
# print categorical variables containing missing values

cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

print(df[cat1].isnull().sum())
```

::: {.output .stream .stdout}
    WindGustDir     9330
    WindDir9am     10013
    WindDir3pm      3778
    RainToday       1406
    dtype: int64
:::
:::

::: {.cell .markdown}
We can see that there are only 4 categorical variables in the dataset
which contains missing values. These are `WindGustDir`, `WindDir9am`,
`WindDir3pm` and `RainToday`.
:::

::: {.cell .markdown}
### Frequency counts of categorical variables

Now, I will check the frequency counts of categorical variables.
:::

::: {.cell .code execution_count="13"}
``` python
# view frequency of categorical variables

for var in categorical: 
    
    print(df[var].value_counts())
```

::: {.output .stream .stdout}
    2014-04-15    49
    2013-08-04    49
    2014-03-18    49
    2014-07-08    49
    2014-02-27    49
                  ..
    2007-11-01     1
    2007-12-30     1
    2007-12-12     1
    2008-01-20     1
    2007-12-05     1
    Name: Date, Length: 3436, dtype: int64
    Canberra            3418
    Sydney              3337
    Perth               3193
    Darwin              3192
    Hobart              3188
    Brisbane            3161
    Adelaide            3090
    Bendigo             3034
    Townsville          3033
    AliceSprings        3031
    MountGambier        3030
    Ballarat            3028
    Launceston          3028
    Albany              3016
    Albury              3011
    PerthAirport        3009
    MelbourneAirport    3009
    Mildura             3007
    SydneyAirport       3005
    Nuriootpa           3002
    Sale                3000
    Watsonia            2999
    Tuggeranong         2998
    Portland            2996
    Woomera             2990
    Cobar               2988
    Cairns              2988
    Wollongong          2983
    GoldCoast           2980
    WaggaWagga          2976
    NorfolkIsland       2964
    Penrith             2964
    SalmonGums          2955
    Newcastle           2955
    CoffsHarbour        2953
    Witchcliffe         2952
    Richmond            2951
    Dartmoor            2943
    NorahHead           2929
    BadgerysCreek       2928
    MountGinini         2907
    Moree               2854
    Walpole             2819
    PearceRAAF          2762
    Williamtown         2553
    Melbourne           2435
    Nhil                1569
    Katherine           1559
    Uluru               1521
    Name: Location, dtype: int64
    W      9780
    SE     9309
    E      9071
    N      9033
    SSE    8993
    S      8949
    WSW    8901
    SW     8797
    SSW    8610
    WNW    8066
    NW     8003
    ENE    7992
    ESE    7305
    NE     7060
    NNW    6561
    NNE    6433
    Name: WindGustDir, dtype: int64
    N      11393
    SE      9162
    E       9024
    SSE     8966
    NW      8552
    S       8493
    W       8260
    SW      8237
    NNE     7948
    NNW     7840
    ENE     7735
    ESE     7558
    NE      7527
    SSW     7448
    WNW     7194
    WSW     6843
    Name: WindDir9am, dtype: int64
    SE     10663
    W       9911
    S       9598
    WSW     9329
    SW      9182
    SSE     9142
    N       8667
    WNW     8656
    NW      8468
    ESE     8382
    E       8342
    NE      8164
    SSW     8010
    NNW     7733
    ENE     7724
    NNE     6444
    Name: WindDir3pm, dtype: int64
    No     109332
    Yes     31455
    Name: RainToday, dtype: int64
    No     110316
    Yes     31877
    Name: RainTomorrow, dtype: int64
:::
:::

::: {.cell .code execution_count="14"}
``` python
# view frequency distribution of categorical variables

for var in categorical: 
    
    print(df[var].value_counts()/np.float(len(df)))
```

::: {.output .stream .stdout}
    2014-04-15    0.000345
    2013-08-04    0.000345
    2014-03-18    0.000345
    2014-07-08    0.000345
    2014-02-27    0.000345
                    ...   
    2007-11-01    0.000007
    2007-12-30    0.000007
    2007-12-12    0.000007
    2008-01-20    0.000007
    2007-12-05    0.000007
    Name: Date, Length: 3436, dtype: float64
    Canberra            0.024038
    Sydney              0.023468
    Perth               0.022455
    Darwin              0.022448
    Hobart              0.022420
    Brisbane            0.022230
    Adelaide            0.021731
    Bendigo             0.021337
    Townsville          0.021330
    AliceSprings        0.021316
    MountGambier        0.021309
    Ballarat            0.021295
    Launceston          0.021295
    Albany              0.021211
    Albury              0.021175
    PerthAirport        0.021161
    MelbourneAirport    0.021161
    Mildura             0.021147
    SydneyAirport       0.021133
    Nuriootpa           0.021112
    Sale                0.021098
    Watsonia            0.021091
    Tuggeranong         0.021084
    Portland            0.021070
    Woomera             0.021028
    Cobar               0.021014
    Cairns              0.021014
    Wollongong          0.020979
    GoldCoast           0.020957
    WaggaWagga          0.020929
    NorfolkIsland       0.020845
    Penrith             0.020845
    SalmonGums          0.020782
    Newcastle           0.020782
    CoffsHarbour        0.020768
    Witchcliffe         0.020761
    Richmond            0.020753
    Dartmoor            0.020697
    NorahHead           0.020599
    BadgerysCreek       0.020592
    MountGinini         0.020444
    Moree               0.020071
    Walpole             0.019825
    PearceRAAF          0.019424
    Williamtown         0.017954
    Melbourne           0.017125
    Nhil                0.011034
    Katherine           0.010964
    Uluru               0.010697
    Name: Location, dtype: float64
    W      0.068780
    SE     0.065467
    E      0.063794
    N      0.063526
    SSE    0.063245
    S      0.062936
    WSW    0.062598
    SW     0.061867
    SSW    0.060552
    WNW    0.056726
    NW     0.056283
    ENE    0.056205
    ESE    0.051374
    NE     0.049651
    NNW    0.046142
    NNE    0.045241
    Name: WindGustDir, dtype: float64
    N      0.080123
    SE     0.064434
    E      0.063463
    SSE    0.063055
    NW     0.060144
    S      0.059729
    W      0.058090
    SW     0.057928
    NNE    0.055896
    NNW    0.055136
    ENE    0.054398
    ESE    0.053153
    NE     0.052935
    SSW    0.052380
    WNW    0.050593
    WSW    0.048125
    Name: WindDir9am, dtype: float64
    SE     0.074990
    W      0.069701
    S      0.067500
    WSW    0.065608
    SW     0.064574
    SSE    0.064293
    N      0.060952
    WNW    0.060875
    NW     0.059553
    ESE    0.058948
    E      0.058667
    NE     0.057415
    SSW    0.056332
    NNW    0.054384
    ENE    0.054321
    NNE    0.045319
    Name: WindDir3pm, dtype: float64
    No     0.768899
    Yes    0.221213
    Name: RainToday, dtype: float64
    No     0.775819
    Yes    0.224181
    Name: RainTomorrow, dtype: float64
:::
:::

::: {.cell .markdown}
### Number of labels: cardinality

The number of labels within a categorical variable is known as
**cardinality**. A high number of labels within a variable is known as
**high cardinality**. High cardinality may pose some serious problems in
the machine learning model. So, I will check for high cardinality.
:::

::: {.cell .code execution_count="15"}
``` python
# check for cardinality in categorical variables

for var in categorical:
    
    print(var, ' contains ', len(df[var].unique()), ' labels')
```

::: {.output .stream .stdout}
    Date  contains  3436  labels
    Location  contains  49  labels
    WindGustDir  contains  17  labels
    WindDir9am  contains  17  labels
    WindDir3pm  contains  17  labels
    RainToday  contains  3  labels
    RainTomorrow  contains  2  labels
:::
:::

::: {.cell .markdown}
We can see that there is a `Date` variable which needs to be
preprocessed. I will do preprocessing in the following section.

All the other variables contain relatively smaller number of variables.
:::

::: {.cell .markdown}
### Feature Engineering of Date Variable
:::

::: {.cell .code execution_count="16"}
``` python
df['Date'].dtypes
```

::: {.output .execute_result execution_count="16"}
    dtype('O')
:::
:::

::: {.cell .markdown}
We can see that the data type of `Date` variable is object. I will parse
the date currently coded as object into datetime format.
:::

::: {.cell .code execution_count="17"}
``` python
# parse the dates, currently coded as strings, into datetime format

df['Date'] = pd.to_datetime(df['Date'])
```
:::

::: {.cell .code execution_count="18"}
``` python
# extract year from date

df['Year'] = df['Date'].dt.year

df['Year'].head()
```

::: {.output .execute_result execution_count="18"}
    0    2008
    1    2008
    2    2008
    3    2008
    4    2008
    Name: Year, dtype: int64
:::
:::

::: {.cell .code execution_count="19"}
``` python
# extract month from date

df['Month'] = df['Date'].dt.month

df['Month'].head()
```

::: {.output .execute_result execution_count="19"}
    0    12
    1    12
    2    12
    3    12
    4    12
    Name: Month, dtype: int64
:::
:::

::: {.cell .code execution_count="20"}
``` python
# extract day from date

df['Day'] = df['Date'].dt.day

df['Day'].head()
```

::: {.output .execute_result execution_count="20"}
    0    1
    1    2
    2    3
    3    4
    4    5
    Name: Day, dtype: int64
:::
:::

::: {.cell .code execution_count="21"}
``` python
# again view the summary of dataset

df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 142193 entries, 0 to 142192
    Data columns (total 26 columns):
    Date             142193 non-null datetime64[ns]
    Location         142193 non-null object
    MinTemp          141556 non-null float64
    MaxTemp          141871 non-null float64
    Rainfall         140787 non-null float64
    Evaporation      81350 non-null float64
    Sunshine         74377 non-null float64
    WindGustDir      132863 non-null object
    WindGustSpeed    132923 non-null float64
    WindDir9am       132180 non-null object
    WindDir3pm       138415 non-null object
    WindSpeed9am     140845 non-null float64
    WindSpeed3pm     139563 non-null float64
    Humidity9am      140419 non-null float64
    Humidity3pm      138583 non-null float64
    Pressure9am      128179 non-null float64
    Pressure3pm      128212 non-null float64
    Cloud9am         88536 non-null float64
    Cloud3pm         85099 non-null float64
    Temp9am          141289 non-null float64
    Temp3pm          139467 non-null float64
    RainToday        140787 non-null object
    RainTomorrow     142193 non-null object
    Year             142193 non-null int64
    Month            142193 non-null int64
    Day              142193 non-null int64
    dtypes: datetime64[ns](1), float64(16), int64(3), object(6)
    memory usage: 28.2+ MB
:::
:::

::: {.cell .markdown}
We can see that there are three additional columns created from `Date`
variable. Now, I will drop the original `Date` variable from the
dataset.
:::

::: {.cell .code execution_count="22"}
``` python
# drop the original Date variable

df.drop('Date', axis=1, inplace = True)
```
:::

::: {.cell .code execution_count="23"}
``` python
# preview the dataset again

df.head()
```

::: {.output .execute_result execution_count="23"}
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
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>...</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Albury</td>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>44.0</td>
      <td>W</td>
      <td>WNW</td>
      <td>...</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albury</td>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WNW</td>
      <td>44.0</td>
      <td>NNW</td>
      <td>WSW</td>
      <td>...</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Albury</td>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WSW</td>
      <td>46.0</td>
      <td>W</td>
      <td>WSW</td>
      <td>...</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albury</td>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NE</td>
      <td>24.0</td>
      <td>SE</td>
      <td>E</td>
      <td>...</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Albury</td>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>41.0</td>
      <td>ENE</td>
      <td>NW</td>
      <td>...</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>
```
:::
:::

::: {.cell .markdown}
Now, we can see that the `Date` variable has been removed from the
dataset.
:::

::: {.cell .markdown}
### Explore Categorical Variables

Now, I will explore the categorical variables one by one.
:::

::: {.cell .code execution_count="24"}
``` python
# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
```

::: {.output .stream .stdout}
    There are 6 categorical variables

    The categorical variables are : ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
:::
:::

::: {.cell .markdown}
We can see that there are 6 categorical variables in the dataset. The
`Date` variable has been removed. First, I will check missing values in
categorical variables.
:::

::: {.cell .code execution_count="25"}
``` python
# check for missing values in categorical variables 

df[categorical].isnull().sum()
```

::: {.output .execute_result execution_count="25"}
    Location            0
    WindGustDir      9330
    WindDir9am      10013
    WindDir3pm       3778
    RainToday        1406
    RainTomorrow        0
    dtype: int64
:::
:::

::: {.cell .markdown}
We can see that `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday`
variables contain missing values. I will explore these variables one by
one.
:::

::: {.cell .markdown}
### Explore `Location` variable
:::

::: {.cell .code execution_count="26"}
``` python
# print number of labels in Location variable

print('Location contains', len(df.Location.unique()), 'labels')
```

::: {.output .stream .stdout}
    Location contains 49 labels
:::
:::

::: {.cell .code execution_count="27"}
``` python
# check labels in location variable

df.Location.unique()
```

::: {.output .execute_result execution_count="27"}
    array(['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
           'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
           'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
           'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
           'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
           'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
           'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
           'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
           'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
           'AliceSprings', 'Darwin', 'Katherine', 'Uluru'], dtype=object)
:::
:::

::: {.cell .code execution_count="28"}
``` python
# check frequency distribution of values in Location variable

df.Location.value_counts()
```

::: {.output .execute_result execution_count="28"}
    Canberra            3418
    Sydney              3337
    Perth               3193
    Darwin              3192
    Hobart              3188
    Brisbane            3161
    Adelaide            3090
    Bendigo             3034
    Townsville          3033
    AliceSprings        3031
    MountGambier        3030
    Ballarat            3028
    Launceston          3028
    Albany              3016
    Albury              3011
    PerthAirport        3009
    MelbourneAirport    3009
    Mildura             3007
    SydneyAirport       3005
    Nuriootpa           3002
    Sale                3000
    Watsonia            2999
    Tuggeranong         2998
    Portland            2996
    Woomera             2990
    Cobar               2988
    Cairns              2988
    Wollongong          2983
    GoldCoast           2980
    WaggaWagga          2976
    NorfolkIsland       2964
    Penrith             2964
    SalmonGums          2955
    Newcastle           2955
    CoffsHarbour        2953
    Witchcliffe         2952
    Richmond            2951
    Dartmoor            2943
    NorahHead           2929
    BadgerysCreek       2928
    MountGinini         2907
    Moree               2854
    Walpole             2819
    PearceRAAF          2762
    Williamtown         2553
    Melbourne           2435
    Nhil                1569
    Katherine           1559
    Uluru               1521
    Name: Location, dtype: int64
:::
:::

::: {.cell .code execution_count="29"}
``` python
# let's do One Hot Encoding of Location variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method

pd.get_dummies(df.Location, drop_first=True).head()
```

::: {.output .execute_result execution_count="29"}
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
      <th>Albany</th>
      <th>Albury</th>
      <th>AliceSprings</th>
      <th>BadgerysCreek</th>
      <th>Ballarat</th>
      <th>Bendigo</th>
      <th>Brisbane</th>
      <th>Cairns</th>
      <th>Canberra</th>
      <th>Cobar</th>
      <th>...</th>
      <th>Townsville</th>
      <th>Tuggeranong</th>
      <th>Uluru</th>
      <th>WaggaWagga</th>
      <th>Walpole</th>
      <th>Watsonia</th>
      <th>Williamtown</th>
      <th>Witchcliffe</th>
      <th>Wollongong</th>
      <th>Woomera</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>
```
:::
:::

::: {.cell .markdown}
### Explore `WindGustDir` variable
:::

::: {.cell .code execution_count="30"}
``` python
# print number of labels in WindGustDir variable

print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')
```

::: {.output .stream .stdout}
    WindGustDir contains 17 labels
:::
:::

::: {.cell .code execution_count="31"}
``` python
# check labels in WindGustDir variable

df['WindGustDir'].unique()
```

::: {.output .execute_result execution_count="31"}
    array(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE', 'SSE',
           'S', 'NW', 'SE', 'ESE', nan, 'E', 'SSW'], dtype=object)
:::
:::

::: {.cell .code execution_count="32"}
``` python
# check frequency distribution of values in WindGustDir variable

df.WindGustDir.value_counts()
```

::: {.output .execute_result execution_count="32"}
    W      9780
    SE     9309
    E      9071
    N      9033
    SSE    8993
    S      8949
    WSW    8901
    SW     8797
    SSW    8610
    WNW    8066
    NW     8003
    ENE    7992
    ESE    7305
    NE     7060
    NNW    6561
    NNE    6433
    Name: WindGustDir, dtype: int64
:::
:::

::: {.cell .code execution_count="33"}
``` python
# let's do One Hot Encoding of WindGustDir variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()
```

::: {.output .execute_result execution_count="33"}
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
      <th>ENE</th>
      <th>ESE</th>
      <th>N</th>
      <th>NE</th>
      <th>NNE</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="34"}
``` python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)
```

::: {.output .execute_result execution_count="34"}
    ENE    7992
    ESE    7305
    N      9033
    NE     7060
    NNE    6433
    NNW    6561
    NW     8003
    S      8949
    SE     9309
    SSE    8993
    SSW    8610
    SW     8797
    W      9780
    WNW    8066
    WSW    8901
    NaN    9330
    dtype: int64
:::
:::

::: {.cell .markdown}
We can see that there are 9330 missing values in WindGustDir variable.
:::

::: {.cell .markdown}
### Explore `WindDir9am` variable
:::

::: {.cell .code execution_count="35"}
``` python
# print number of labels in WindDir9am variable

print('WindDir9am contains', len(df['WindDir9am'].unique()), 'labels')
```

::: {.output .stream .stdout}
    WindDir9am contains 17 labels
:::
:::

::: {.cell .code execution_count="36"}
``` python
# check labels in WindDir9am variable

df['WindDir9am'].unique()
```

::: {.output .execute_result execution_count="36"}
    array(['W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE', nan, 'SSW', 'N',
           'WSW', 'ESE', 'E', 'NW', 'WNW', 'NNE'], dtype=object)
:::
:::

::: {.cell .code execution_count="37"}
``` python
# check frequency distribution of values in WindDir9am variable

df['WindDir9am'].value_counts()
```

::: {.output .execute_result execution_count="37"}
    N      11393
    SE      9162
    E       9024
    SSE     8966
    NW      8552
    S       8493
    W       8260
    SW      8237
    NNE     7948
    NNW     7840
    ENE     7735
    ESE     7558
    NE      7527
    SSW     7448
    WNW     7194
    WSW     6843
    Name: WindDir9am, dtype: int64
:::
:::

::: {.cell .code execution_count="38"}
``` python
# let's do One Hot Encoding of WindDir9am variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
```

::: {.output .execute_result execution_count="38"}
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
      <th>ENE</th>
      <th>ESE</th>
      <th>N</th>
      <th>NE</th>
      <th>NNE</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="39"}
``` python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)
```

::: {.output .execute_result execution_count="39"}
    ENE     7735
    ESE     7558
    N      11393
    NE      7527
    NNE     7948
    NNW     7840
    NW      8552
    S       8493
    SE      9162
    SSE     8966
    SSW     7448
    SW      8237
    W       8260
    WNW     7194
    WSW     6843
    NaN    10013
    dtype: int64
:::
:::

::: {.cell .markdown}
We can see that there are 10013 missing values in the `WindDir9am`
variable.
:::

::: {.cell .markdown}
### Explore `WindDir3pm` variable
:::

::: {.cell .code execution_count="40"}
``` python
# print number of labels in WindDir3pm variable

print('WindDir3pm contains', len(df['WindDir3pm'].unique()), 'labels')
```

::: {.output .stream .stdout}
    WindDir3pm contains 17 labels
:::
:::

::: {.cell .code execution_count="41"}
``` python
# check labels in WindDir3pm variable

df['WindDir3pm'].unique()
```

::: {.output .execute_result execution_count="41"}
    array(['WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE', 'NNW', 'SSW',
           'SW', 'SE', 'N', 'S', 'NNE', nan, 'NE'], dtype=object)
:::
:::

::: {.cell .code execution_count="42"}
``` python
# check frequency distribution of values in WindDir3pm variable

df['WindDir3pm'].value_counts()
```

::: {.output .execute_result execution_count="42"}
    SE     10663
    W       9911
    S       9598
    WSW     9329
    SW      9182
    SSE     9142
    N       8667
    WNW     8656
    NW      8468
    ESE     8382
    E       8342
    NE      8164
    SSW     8010
    NNW     7733
    ENE     7724
    NNE     6444
    Name: WindDir3pm, dtype: int64
:::
:::

::: {.cell .code execution_count="43"}
``` python
# let's do One Hot Encoding of WindDir3pm variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()
```

::: {.output .execute_result execution_count="43"}
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
      <th>ENE</th>
      <th>ESE</th>
      <th>N</th>
      <th>NE</th>
      <th>NNE</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="44"}
``` python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)
```

::: {.output .execute_result execution_count="44"}
    ENE     7724
    ESE     8382
    N       8667
    NE      8164
    NNE     6444
    NNW     7733
    NW      8468
    S       9598
    SE     10663
    SSE     9142
    SSW     8010
    SW      9182
    W       9911
    WNW     8656
    WSW     9329
    NaN     3778
    dtype: int64
:::
:::

::: {.cell .markdown}
There are 3778 missing values in the `WindDir3pm` variable.
:::

::: {.cell .markdown}
### Explore `RainToday` variable
:::

::: {.cell .code execution_count="45"}
``` python
# print number of labels in RainToday variable

print('RainToday contains', len(df['RainToday'].unique()), 'labels')
```

::: {.output .stream .stdout}
    RainToday contains 3 labels
:::
:::

::: {.cell .code execution_count="46"}
``` python
# check labels in WindGustDir variable

df['RainToday'].unique()
```

::: {.output .execute_result execution_count="46"}
    array(['No', 'Yes', nan], dtype=object)
:::
:::

::: {.cell .code execution_count="47"}
``` python
# check frequency distribution of values in WindGustDir variable

df.RainToday.value_counts()
```

::: {.output .execute_result execution_count="47"}
    No     109332
    Yes     31455
    Name: RainToday, dtype: int64
:::
:::

::: {.cell .code execution_count="48"}
``` python
# let's do One Hot Encoding of RainToday variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()
```

::: {.output .execute_result execution_count="48"}
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
      <th>Yes</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="49"}
``` python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0)
```

::: {.output .execute_result execution_count="49"}
    Yes    31455
    NaN     1406
    dtype: int64
:::
:::

::: {.cell .markdown}
There are 1406 missing values in the `RainToday` variable.
:::

::: {.cell .markdown}
### Explore Numerical Variables
:::

::: {.cell .code execution_count="50"}
``` python
# find numerical variables

numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)
```

::: {.output .stream .stdout}
    There are 19 numerical variables

    The numerical variables are : ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Year', 'Month', 'Day']
:::
:::

::: {.cell .code execution_count="51"}
``` python
# view the numerical variables

df[numerical].head()
```

::: {.output .execute_result execution_count="51"}
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
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>20.0</td>
      <td>24.0</td>
      <td>71.0</td>
      <td>22.0</td>
      <td>1007.7</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>2008</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>4.0</td>
      <td>22.0</td>
      <td>44.0</td>
      <td>25.0</td>
      <td>1010.6</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>2008</td>
      <td>12</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>46.0</td>
      <td>19.0</td>
      <td>26.0</td>
      <td>38.0</td>
      <td>30.0</td>
      <td>1007.6</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>2008</td>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>9.0</td>
      <td>45.0</td>
      <td>16.0</td>
      <td>1017.6</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>2008</td>
      <td>12</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.0</td>
      <td>7.0</td>
      <td>20.0</td>
      <td>82.0</td>
      <td>33.0</td>
      <td>1010.8</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>2008</td>
      <td>12</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
### Summary of numerical variables

-   There are 16 numerical variables.

-   These are given by `MinTemp`, `MaxTemp`, `Rainfall`, `Evaporation`,
    `Sunshine`, `WindGustSpeed`, `WindSpeed9am`, `WindSpeed3pm`,
    `Humidity9am`, `Humidity3pm`, `Pressure9am`, `Pressure3pm`,
    `Cloud9am`, `Cloud3pm`, `Temp9am` and `Temp3pm`.

-   All of the numerical variables are of continuous type.
:::

::: {.cell .markdown}
## Explore problems within numerical variables

Now, I will explore the numerical variables.

### Missing values in numerical variables
:::

::: {.cell .code execution_count="52"}
``` python
# check missing values in numerical variables

df[numerical].isnull().sum()
```

::: {.output .execute_result execution_count="52"}
    MinTemp            637
    MaxTemp            322
    Rainfall          1406
    Evaporation      60843
    Sunshine         67816
    WindGustSpeed     9270
    WindSpeed9am      1348
    WindSpeed3pm      2630
    Humidity9am       1774
    Humidity3pm       3610
    Pressure9am      14014
    Pressure3pm      13981
    Cloud9am         53657
    Cloud3pm         57094
    Temp9am            904
    Temp3pm           2726
    Year                 0
    Month                0
    Day                  0
    dtype: int64
:::
:::

::: {.cell .markdown}
We can see that all the 16 numerical variables contain missing values.
:::

::: {.cell .markdown}
### Outliers in numerical variables
:::

::: {.cell .code execution_count="53"}
``` python
# view summary statistics in numerical variables

print(round(df[numerical].describe()),2)
```

::: {.output .stream .stdout}
            MinTemp   MaxTemp  Rainfall  Evaporation  Sunshine  WindGustSpeed  \
    count  141556.0  141871.0  140787.0      81350.0   74377.0       132923.0   
    mean       12.0      23.0       2.0          5.0       8.0           40.0   
    std         6.0       7.0       8.0          4.0       4.0           14.0   
    min        -8.0      -5.0       0.0          0.0       0.0            6.0   
    25%         8.0      18.0       0.0          3.0       5.0           31.0   
    50%        12.0      23.0       0.0          5.0       8.0           39.0   
    75%        17.0      28.0       1.0          7.0      11.0           48.0   
    max        34.0      48.0     371.0        145.0      14.0          135.0   

           WindSpeed9am  WindSpeed3pm  Humidity9am  Humidity3pm  Pressure9am  \
    count      140845.0      139563.0     140419.0     138583.0     128179.0   
    mean           14.0          19.0         69.0         51.0       1018.0   
    std             9.0           9.0         19.0         21.0          7.0   
    min             0.0           0.0          0.0          0.0        980.0   
    25%             7.0          13.0         57.0         37.0       1013.0   
    50%            13.0          19.0         70.0         52.0       1018.0   
    75%            19.0          24.0         83.0         66.0       1022.0   
    max           130.0          87.0        100.0        100.0       1041.0   

           Pressure3pm  Cloud9am  Cloud3pm   Temp9am   Temp3pm      Year  \
    count     128212.0   88536.0   85099.0  141289.0  139467.0  142193.0   
    mean        1015.0       4.0       5.0      17.0      22.0    2013.0   
    std            7.0       3.0       3.0       6.0       7.0       3.0   
    min          977.0       0.0       0.0      -7.0      -5.0    2007.0   
    25%         1010.0       1.0       2.0      12.0      17.0    2011.0   
    50%         1015.0       5.0       5.0      17.0      21.0    2013.0   
    75%         1020.0       7.0       7.0      22.0      26.0    2015.0   
    max         1040.0       9.0       9.0      40.0      47.0    2017.0   

              Month       Day  
    count  142193.0  142193.0  
    mean        6.0      16.0  
    std         3.0       9.0  
    min         1.0       1.0  
    25%         3.0       8.0  
    50%         6.0      16.0  
    75%         9.0      23.0  
    max        12.0      31.0   2
:::
:::

::: {.cell .markdown}
On closer inspection, we can see that the `Rainfall`, `Evaporation`,
`WindSpeed9am` and `WindSpeed3pm` columns may contain outliers.

I will draw boxplots to visualise outliers in the above variables.
:::

::: {.cell .code execution_count="54"}
``` python
# draw boxplots to visualize outliers

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')


plt.subplot(2, 2, 2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')


plt.subplot(2, 2, 3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')


plt.subplot(2, 2, 4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')
```

::: {.output .execute_result execution_count="54"}
    Text(0, 0.5, 'WindSpeed3pm')
:::

::: {.output .display_data}
![](vertopal_ca93cbc8c7524834a21e37d8a1959b04/f412b74f4c6ce7e448b15a686bf1712b54c9fb05.png)
:::
:::

::: {.cell .markdown}
The above boxplots confirm that there are lot of outliers in these
variables.
:::

::: {.cell .markdown}
### Check the distribution of variables

Now, I will plot the histograms to check distributions to find out if
they are normal or skewed. If the variable follows normal distribution,
then I will do `Extreme Value Analysis` otherwise if they are skewed, I
will find IQR (Interquantile range).
:::

::: {.cell .code execution_count="55"}
``` python
# plot histogram to check distribution

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')
```

::: {.output .execute_result execution_count="55"}
    Text(0, 0.5, 'RainTomorrow')
:::

::: {.output .display_data}
![](vertopal_ca93cbc8c7524834a21e37d8a1959b04/a34fe420d90754196e14e07c936a43ecd3cf4fd4.png)
:::
:::

::: {.cell .markdown}
We can see that all the four variables are skewed. So, I will use
interquantile range to find outliers.
:::

::: {.cell .code execution_count="56"}
``` python
# find outliers for Rainfall variable

IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

::: {.output .stream .stdout}
    Rainfall outliers are values < -2.4000000000000004 or > 3.2
:::
:::

::: {.cell .markdown}
For `Rainfall`, the minimum and maximum values are 0.0 and 371.0. So,
the outliers are values \> 3.2.
:::

::: {.cell .code execution_count="57"}
``` python
# find outliers for Evaporation variable

IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

::: {.output .stream .stdout}
    Evaporation outliers are values < -11.800000000000002 or > 21.800000000000004
:::
:::

::: {.cell .markdown}
For `Evaporation`, the minimum and maximum values are 0.0 and 145.0. So,
the outliers are values \> 21.8.
:::

::: {.cell .code execution_count="58"}
``` python
# find outliers for WindSpeed9am variable

IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

::: {.output .stream .stdout}
    WindSpeed9am outliers are values < -29.0 or > 55.0
:::
:::

::: {.cell .markdown}
For `WindSpeed9am`, the minimum and maximum values are 0.0 and 130.0.
So, the outliers are values \> 55.0.
:::

::: {.cell .code execution_count="59"}
``` python
# find outliers for WindSpeed3pm variable

IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

::: {.output .stream .stdout}
    WindSpeed3pm outliers are values < -20.0 or > 57.0
:::
:::

::: {.cell .markdown}
For `WindSpeed3pm`, the minimum and maximum values are 0.0 and 87.0. So,
the outliers are values \> 57.0.
:::

::: {.cell .markdown}
# **8. Declare feature vector and target variable** `<a class="anchor" id="8">`{=html}`</a>`{=html} {#8-declare-feature-vector-and-target-variable-}

[Table of Contents](#0.1)
:::

::: {.cell .code execution_count="60"}
``` python
X = df.drop(['RainTomorrow'], axis=1)

y = df['RainTomorrow']
```
:::

::: {.cell .markdown}
# **9. Split data into separate training and test set** `<a class="anchor" id="9">`{=html}`</a>`{=html} {#9-split-data-into-separate-training-and-test-set-}

[Table of Contents](#0.1)
:::

::: {.cell .code execution_count="61"}
``` python
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```
:::

::: {.cell .code execution_count="62"}
``` python
# check the shape of X_train and X_test

X_train.shape, X_test.shape
```

::: {.output .execute_result execution_count="62"}
    ((113754, 24), (28439, 24))
:::
:::

::: {.cell .markdown}
# **10. Feature Engineering** `<a class="anchor" id="10">`{=html}`</a>`{=html} {#10-feature-engineering-}

[Table of Contents](#0.1)

**Feature Engineering** is the process of transforming raw data into
useful features that help us to understand our model better and increase
its predictive power. I will carry out feature engineering on different
types of variables.

First, I will display the categorical and numerical variables again
separately.
:::

::: {.cell .code execution_count="63"}
``` python
# check data types in X_train

X_train.dtypes
```

::: {.output .execute_result execution_count="63"}
    Location          object
    MinTemp          float64
    MaxTemp          float64
    Rainfall         float64
    Evaporation      float64
    Sunshine         float64
    WindGustDir       object
    WindGustSpeed    float64
    WindDir9am        object
    WindDir3pm        object
    WindSpeed9am     float64
    WindSpeed3pm     float64
    Humidity9am      float64
    Humidity3pm      float64
    Pressure9am      float64
    Pressure3pm      float64
    Cloud9am         float64
    Cloud3pm         float64
    Temp9am          float64
    Temp3pm          float64
    RainToday         object
    Year               int64
    Month              int64
    Day                int64
    dtype: object
:::
:::

::: {.cell .code execution_count="64"}
``` python
# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical
```

::: {.output .execute_result execution_count="64"}
    ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
:::
:::

::: {.cell .code execution_count="65"}
``` python
# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical
```

::: {.output .execute_result execution_count="65"}
    ['MinTemp',
     'MaxTemp',
     'Rainfall',
     'Evaporation',
     'Sunshine',
     'WindGustSpeed',
     'WindSpeed9am',
     'WindSpeed3pm',
     'Humidity9am',
     'Humidity3pm',
     'Pressure9am',
     'Pressure3pm',
     'Cloud9am',
     'Cloud3pm',
     'Temp9am',
     'Temp3pm',
     'Year',
     'Month',
     'Day']
:::
:::

::: {.cell .markdown}
### Engineering missing values in numerical variables
:::

::: {.cell .code execution_count="66"}
``` python
# check missing values in numerical variables in X_train

X_train[numerical].isnull().sum()
```

::: {.output .execute_result execution_count="66"}
    MinTemp            495
    MaxTemp            264
    Rainfall          1139
    Evaporation      48718
    Sunshine         54314
    WindGustSpeed     7367
    WindSpeed9am      1086
    WindSpeed3pm      2094
    Humidity9am       1449
    Humidity3pm       2890
    Pressure9am      11212
    Pressure3pm      11186
    Cloud9am         43137
    Cloud3pm         45768
    Temp9am            740
    Temp3pm           2171
    Year                 0
    Month                0
    Day                  0
    dtype: int64
:::
:::

::: {.cell .code execution_count="67"}
``` python
# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()
```

::: {.output .execute_result execution_count="67"}
    MinTemp            142
    MaxTemp             58
    Rainfall           267
    Evaporation      12125
    Sunshine         13502
    WindGustSpeed     1903
    WindSpeed9am       262
    WindSpeed3pm       536
    Humidity9am        325
    Humidity3pm        720
    Pressure9am       2802
    Pressure3pm       2795
    Cloud9am         10520
    Cloud3pm         11326
    Temp9am            164
    Temp3pm            555
    Year                 0
    Month                0
    Day                  0
    dtype: int64
:::
:::

::: {.cell .code execution_count="68"}
``` python
# print percentage of missing values in the numerical variables in training set

for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, round(X_train[col].isnull().mean(),4))
```

::: {.output .stream .stdout}
    MinTemp 0.0044
    MaxTemp 0.0023
    Rainfall 0.01
    Evaporation 0.4283
    Sunshine 0.4775
    WindGustSpeed 0.0648
    WindSpeed9am 0.0095
    WindSpeed3pm 0.0184
    Humidity9am 0.0127
    Humidity3pm 0.0254
    Pressure9am 0.0986
    Pressure3pm 0.0983
    Cloud9am 0.3792
    Cloud3pm 0.4023
    Temp9am 0.0065
    Temp3pm 0.0191
:::
:::

::: {.cell .markdown}
### Assumption

I assume that the data are missing completely at random (MCAR). There
are two methods which can be used to impute missing values. One is mean
or median imputation and other one is random sample imputation. When
there are outliers in the dataset, we should use median imputation. So,
I will use median imputation because median imputation is robust to
outliers.

I will impute missing values with the appropriate statistical measures
of the data, in this case median. Imputation should be done over the
training set, and then propagated to the test set. It means that the
statistical measures to be used to fill missing values both in train and
test set, should be extracted from the train set only. This is to avoid
overfitting.
:::

::: {.cell .code execution_count="69"}
``` python
# impute missing values in X_train and X_test with respective column median in X_train

for df1 in [X_train, X_test]:
    for col in numerical:
        col_median=X_train[col].median()
        df1[col].fillna(col_median, inplace=True)           
      
```
:::

::: {.cell .code execution_count="70"}
``` python
# check again missing values in numerical variables in X_train

X_train[numerical].isnull().sum()
```

::: {.output .execute_result execution_count="70"}
    MinTemp          0
    MaxTemp          0
    Rainfall         0
    Evaporation      0
    Sunshine         0
    WindGustSpeed    0
    WindSpeed9am     0
    WindSpeed3pm     0
    Humidity9am      0
    Humidity3pm      0
    Pressure9am      0
    Pressure3pm      0
    Cloud9am         0
    Cloud3pm         0
    Temp9am          0
    Temp3pm          0
    Year             0
    Month            0
    Day              0
    dtype: int64
:::
:::

::: {.cell .code execution_count="71"}
``` python
# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()
```

::: {.output .execute_result execution_count="71"}
    MinTemp          0
    MaxTemp          0
    Rainfall         0
    Evaporation      0
    Sunshine         0
    WindGustSpeed    0
    WindSpeed9am     0
    WindSpeed3pm     0
    Humidity9am      0
    Humidity3pm      0
    Pressure9am      0
    Pressure3pm      0
    Cloud9am         0
    Cloud3pm         0
    Temp9am          0
    Temp3pm          0
    Year             0
    Month            0
    Day              0
    dtype: int64
:::
:::

::: {.cell .markdown}
Now, we can see that there are no missing values in the numerical
columns of training and test set.
:::

::: {.cell .markdown}
### Engineering missing values in categorical variables
:::

::: {.cell .code execution_count="72"}
``` python
# print percentage of missing values in the categorical variables in training set

X_train[categorical].isnull().mean()
```

::: {.output .execute_result execution_count="72"}
    Location       0.000000
    WindGustDir    0.065114
    WindDir9am     0.070134
    WindDir3pm     0.026443
    RainToday      0.010013
    dtype: float64
:::
:::

::: {.cell .code execution_count="73"}
``` python
# print categorical variables with missing data

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))
```

::: {.output .stream .stdout}
    WindGustDir 0.06511419378659213
    WindDir9am 0.07013379749283542
    WindDir3pm 0.026443026179299188
    RainToday 0.01001283471350458
:::
:::

::: {.cell .code execution_count="74"}
``` python
# impute missing categorical variables with most frequent value

for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)
```
:::

::: {.cell .code execution_count="75"}
``` python
# check missing values in categorical variables in X_train

X_train[categorical].isnull().sum()
```

::: {.output .execute_result execution_count="75"}
    Location       0
    WindGustDir    0
    WindDir9am     0
    WindDir3pm     0
    RainToday      0
    dtype: int64
:::
:::

::: {.cell .code execution_count="76"}
``` python
# check missing values in categorical variables in X_test

X_test[categorical].isnull().sum()
```

::: {.output .execute_result execution_count="76"}
    Location       0
    WindGustDir    0
    WindDir9am     0
    WindDir3pm     0
    RainToday      0
    dtype: int64
:::
:::

::: {.cell .markdown}
As a final check, I will check for missing values in X_train and X_test.
:::

::: {.cell .code execution_count="77"}
``` python
# check missing values in X_train

X_train.isnull().sum()
```

::: {.output .execute_result execution_count="77"}
    Location         0
    MinTemp          0
    MaxTemp          0
    Rainfall         0
    Evaporation      0
    Sunshine         0
    WindGustDir      0
    WindGustSpeed    0
    WindDir9am       0
    WindDir3pm       0
    WindSpeed9am     0
    WindSpeed3pm     0
    Humidity9am      0
    Humidity3pm      0
    Pressure9am      0
    Pressure3pm      0
    Cloud9am         0
    Cloud3pm         0
    Temp9am          0
    Temp3pm          0
    RainToday        0
    Year             0
    Month            0
    Day              0
    dtype: int64
:::
:::

::: {.cell .code execution_count="78"}
``` python
# check missing values in X_test

X_test.isnull().sum()
```

::: {.output .execute_result execution_count="78"}
    Location         0
    MinTemp          0
    MaxTemp          0
    Rainfall         0
    Evaporation      0
    Sunshine         0
    WindGustDir      0
    WindGustSpeed    0
    WindDir9am       0
    WindDir3pm       0
    WindSpeed9am     0
    WindSpeed3pm     0
    Humidity9am      0
    Humidity3pm      0
    Pressure9am      0
    Pressure3pm      0
    Cloud9am         0
    Cloud3pm         0
    Temp9am          0
    Temp3pm          0
    RainToday        0
    Year             0
    Month            0
    Day              0
    dtype: int64
:::
:::

::: {.cell .markdown}
We can see that there are no missing values in X_train and X_test.
:::

::: {.cell .markdown}
### Engineering outliers in numerical variables

We have seen that the `Rainfall`, `Evaporation`, `WindSpeed9am` and
`WindSpeed3pm` columns contain outliers. I will use top-coding approach
to cap maximum values and remove outliers from the above variables.
:::

::: {.cell .code execution_count="79"}
``` python
def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)
```
:::

::: {.cell .code execution_count="80"}
``` python
X_train.Rainfall.max(), X_test.Rainfall.max()
```

::: {.output .execute_result execution_count="80"}
    (3.2, 3.2)
:::
:::

::: {.cell .code execution_count="81"}
``` python
X_train.Evaporation.max(), X_test.Evaporation.max()
```

::: {.output .execute_result execution_count="81"}
    (21.8, 21.8)
:::
:::

::: {.cell .code execution_count="82"}
``` python
X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()
```

::: {.output .execute_result execution_count="82"}
    (55.0, 55.0)
:::
:::

::: {.cell .code execution_count="83"}
``` python
X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()
```

::: {.output .execute_result execution_count="83"}
    (57.0, 57.0)
:::
:::

::: {.cell .code execution_count="84"}
``` python
X_train[numerical].describe()
```

::: {.output .execute_result execution_count="84"}
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
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.193497</td>
      <td>23.237216</td>
      <td>0.675080</td>
      <td>5.151606</td>
      <td>8.041154</td>
      <td>39.884074</td>
      <td>13.978155</td>
      <td>18.614756</td>
      <td>68.867486</td>
      <td>51.509547</td>
      <td>1017.640649</td>
      <td>1015.241101</td>
      <td>4.651801</td>
      <td>4.703588</td>
      <td>16.995062</td>
      <td>21.688643</td>
      <td>2012.759727</td>
      <td>6.404021</td>
      <td>15.710419</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.388279</td>
      <td>7.094149</td>
      <td>1.183837</td>
      <td>2.823707</td>
      <td>2.769480</td>
      <td>13.116959</td>
      <td>8.806558</td>
      <td>8.685862</td>
      <td>18.935587</td>
      <td>20.530723</td>
      <td>6.738680</td>
      <td>6.675168</td>
      <td>2.292726</td>
      <td>2.117847</td>
      <td>6.463772</td>
      <td>6.855649</td>
      <td>2.540419</td>
      <td>3.427798</td>
      <td>8.796821</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-8.200000</td>
      <td>-4.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>980.500000</td>
      <td>977.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-7.200000</td>
      <td>-5.400000</td>
      <td>2007.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.600000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.200000</td>
      <td>31.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>57.000000</td>
      <td>37.000000</td>
      <td>1013.500000</td>
      <td>1011.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>12.300000</td>
      <td>16.700000</td>
      <td>2011.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>12.000000</td>
      <td>22.600000</td>
      <td>0.000000</td>
      <td>4.800000</td>
      <td>8.500000</td>
      <td>39.000000</td>
      <td>13.000000</td>
      <td>19.000000</td>
      <td>70.000000</td>
      <td>52.000000</td>
      <td>1017.600000</td>
      <td>1015.200000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>16.700000</td>
      <td>21.100000</td>
      <td>2013.000000</td>
      <td>6.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16.800000</td>
      <td>28.200000</td>
      <td>0.600000</td>
      <td>5.400000</td>
      <td>8.700000</td>
      <td>46.000000</td>
      <td>19.000000</td>
      <td>24.000000</td>
      <td>83.000000</td>
      <td>65.000000</td>
      <td>1021.800000</td>
      <td>1019.400000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>21.500000</td>
      <td>26.300000</td>
      <td>2015.000000</td>
      <td>9.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>33.900000</td>
      <td>48.100000</td>
      <td>3.200000</td>
      <td>21.800000</td>
      <td>14.500000</td>
      <td>135.000000</td>
      <td>55.000000</td>
      <td>57.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>1041.000000</td>
      <td>1039.600000</td>
      <td>9.000000</td>
      <td>8.000000</td>
      <td>40.200000</td>
      <td>46.700000</td>
      <td>2017.000000</td>
      <td>12.000000</td>
      <td>31.000000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
We can now see that the outliers in `Rainfall`, `Evaporation`,
`WindSpeed9am` and `WindSpeed3pm` columns are capped.
:::

::: {.cell .markdown}
### Encode categorical variables
:::

::: {.cell .code execution_count="85"}
``` python
categorical
```

::: {.output .execute_result execution_count="85"}
    ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
:::
:::

::: {.cell .code execution_count="86"}
``` python
X_train[categorical].head()
```

::: {.output .execute_result execution_count="86"}
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
      <th>Location</th>
      <th>WindGustDir</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>RainToday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>110803</th>
      <td>Witchcliffe</td>
      <td>S</td>
      <td>SSE</td>
      <td>S</td>
      <td>No</td>
    </tr>
    <tr>
      <th>87289</th>
      <td>Cairns</td>
      <td>ENE</td>
      <td>SSE</td>
      <td>SE</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>134949</th>
      <td>AliceSprings</td>
      <td>E</td>
      <td>NE</td>
      <td>N</td>
      <td>No</td>
    </tr>
    <tr>
      <th>85553</th>
      <td>Cairns</td>
      <td>ESE</td>
      <td>SSE</td>
      <td>E</td>
      <td>No</td>
    </tr>
    <tr>
      <th>16110</th>
      <td>Newcastle</td>
      <td>W</td>
      <td>N</td>
      <td>SE</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="87"}
``` python
# encode RainToday variable

import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)
```
:::

::: {.cell .code execution_count="88"}
``` python
X_train.head()
```

::: {.output .execute_result execution_count="88"}
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
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>...</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday_0</th>
      <th>RainToday_1</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>110803</th>
      <td>Witchcliffe</td>
      <td>13.9</td>
      <td>22.6</td>
      <td>0.2</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>S</td>
      <td>41.0</td>
      <td>SSE</td>
      <td>S</td>
      <td>...</td>
      <td>1013.4</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>18.8</td>
      <td>20.4</td>
      <td>0</td>
      <td>1</td>
      <td>2014</td>
      <td>4</td>
      <td>25</td>
    </tr>
    <tr>
      <th>87289</th>
      <td>Cairns</td>
      <td>22.4</td>
      <td>29.4</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>6.3</td>
      <td>ENE</td>
      <td>33.0</td>
      <td>SSE</td>
      <td>SE</td>
      <td>...</td>
      <td>1013.1</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>26.4</td>
      <td>27.5</td>
      <td>1</td>
      <td>0</td>
      <td>2015</td>
      <td>11</td>
      <td>2</td>
    </tr>
    <tr>
      <th>134949</th>
      <td>AliceSprings</td>
      <td>9.7</td>
      <td>36.2</td>
      <td>0.0</td>
      <td>11.4</td>
      <td>12.3</td>
      <td>E</td>
      <td>31.0</td>
      <td>NE</td>
      <td>N</td>
      <td>...</td>
      <td>1013.6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>28.5</td>
      <td>35.0</td>
      <td>0</td>
      <td>1</td>
      <td>2014</td>
      <td>10</td>
      <td>19</td>
    </tr>
    <tr>
      <th>85553</th>
      <td>Cairns</td>
      <td>20.5</td>
      <td>30.1</td>
      <td>0.0</td>
      <td>8.8</td>
      <td>11.1</td>
      <td>ESE</td>
      <td>37.0</td>
      <td>SSE</td>
      <td>E</td>
      <td>...</td>
      <td>1010.8</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>27.3</td>
      <td>29.4</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>10</td>
      <td>30</td>
    </tr>
    <tr>
      <th>16110</th>
      <td>Newcastle</td>
      <td>16.8</td>
      <td>29.2</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>W</td>
      <td>39.0</td>
      <td>N</td>
      <td>SE</td>
      <td>...</td>
      <td>1015.2</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>22.2</td>
      <td>27.0</td>
      <td>0</td>
      <td>1</td>
      <td>2012</td>
      <td>11</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>
```
:::
:::

::: {.cell .markdown}
We can see that two additional variables `RainToday_0` and `RainToday_1`
are created from `RainToday` variable.

Now, I will create the `X_train` training set.
:::

::: {.cell .code execution_count="89"}
``` python
X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location), 
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)
```
:::

::: {.cell .code execution_count="90"}
``` python
X_train.head()
```

::: {.output .execute_result execution_count="90"}
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
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>110803</th>
      <td>13.9</td>
      <td>22.6</td>
      <td>0.2</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>41.0</td>
      <td>20.0</td>
      <td>28.0</td>
      <td>65.0</td>
      <td>55.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87289</th>
      <td>22.4</td>
      <td>29.4</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>6.3</td>
      <td>33.0</td>
      <td>7.0</td>
      <td>19.0</td>
      <td>71.0</td>
      <td>59.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>134949</th>
      <td>9.7</td>
      <td>36.2</td>
      <td>0.0</td>
      <td>11.4</td>
      <td>12.3</td>
      <td>31.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>85553</th>
      <td>20.5</td>
      <td>30.1</td>
      <td>0.0</td>
      <td>8.8</td>
      <td>11.1</td>
      <td>37.0</td>
      <td>22.0</td>
      <td>19.0</td>
      <td>59.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16110</th>
      <td>16.8</td>
      <td>29.2</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>39.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>72.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 118 columns</p>
</div>
```
:::
:::

::: {.cell .markdown}
Similarly, I will create the `X_test` testing set.
:::

::: {.cell .code execution_count="91"}
``` python
X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location), 
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)
```
:::

::: {.cell .code execution_count="92"}
``` python
X_test.head()
```

::: {.output .execute_result execution_count="92"}
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
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>86232</th>
      <td>17.4</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>3.6</td>
      <td>11.1</td>
      <td>33.0</td>
      <td>11.0</td>
      <td>19.0</td>
      <td>63.0</td>
      <td>61.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>57576</th>
      <td>6.8</td>
      <td>14.4</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>8.5</td>
      <td>46.0</td>
      <td>17.0</td>
      <td>22.0</td>
      <td>80.0</td>
      <td>55.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>124071</th>
      <td>10.1</td>
      <td>15.4</td>
      <td>3.2</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>31.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>70.0</td>
      <td>61.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>117955</th>
      <td>14.4</td>
      <td>33.4</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>11.6</td>
      <td>41.0</td>
      <td>9.0</td>
      <td>17.0</td>
      <td>40.0</td>
      <td>23.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>133468</th>
      <td>6.8</td>
      <td>14.3</td>
      <td>3.2</td>
      <td>0.2</td>
      <td>7.3</td>
      <td>28.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>92.0</td>
      <td>47.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 118 columns</p>
</div>
```
:::
:::

::: {.cell .markdown}
We now have training and testing set ready for model building. Before
that, we should map all the feature variables onto the same scale. It is
called `feature scaling`. I will do it as follows.
:::

::: {.cell .markdown}
# **11. Feature Scaling** `<a class="anchor" id="11">`{=html}`</a>`{=html} {#11-feature-scaling-}

[Table of Contents](#0.1)
:::

::: {.cell .code execution_count="93"}
``` python
X_train.describe()
```

::: {.output .execute_result execution_count="93"}
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
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>...</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.193497</td>
      <td>23.237216</td>
      <td>0.675080</td>
      <td>5.151606</td>
      <td>8.041154</td>
      <td>39.884074</td>
      <td>13.978155</td>
      <td>18.614756</td>
      <td>68.867486</td>
      <td>51.509547</td>
      <td>...</td>
      <td>0.054530</td>
      <td>0.060288</td>
      <td>0.067259</td>
      <td>0.101605</td>
      <td>0.064059</td>
      <td>0.056402</td>
      <td>0.064464</td>
      <td>0.069334</td>
      <td>0.060798</td>
      <td>0.065483</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.388279</td>
      <td>7.094149</td>
      <td>1.183837</td>
      <td>2.823707</td>
      <td>2.769480</td>
      <td>13.116959</td>
      <td>8.806558</td>
      <td>8.685862</td>
      <td>18.935587</td>
      <td>20.530723</td>
      <td>...</td>
      <td>0.227061</td>
      <td>0.238021</td>
      <td>0.250471</td>
      <td>0.302130</td>
      <td>0.244860</td>
      <td>0.230698</td>
      <td>0.245578</td>
      <td>0.254022</td>
      <td>0.238960</td>
      <td>0.247378</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-8.200000</td>
      <td>-4.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.600000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.200000</td>
      <td>31.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>57.000000</td>
      <td>37.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>12.000000</td>
      <td>22.600000</td>
      <td>0.000000</td>
      <td>4.800000</td>
      <td>8.500000</td>
      <td>39.000000</td>
      <td>13.000000</td>
      <td>19.000000</td>
      <td>70.000000</td>
      <td>52.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16.800000</td>
      <td>28.200000</td>
      <td>0.600000</td>
      <td>5.400000</td>
      <td>8.700000</td>
      <td>46.000000</td>
      <td>19.000000</td>
      <td>24.000000</td>
      <td>83.000000</td>
      <td>65.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>33.900000</td>
      <td>48.100000</td>
      <td>3.200000</td>
      <td>21.800000</td>
      <td>14.500000</td>
      <td>135.000000</td>
      <td>55.000000</td>
      <td>57.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 118 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="94"}
``` python
cols = X_train.columns
```
:::

::: {.cell .code execution_count="95"}
``` python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
```
:::

::: {.cell .code execution_count="96"}
``` python
X_train = pd.DataFrame(X_train, columns=[cols])
```
:::

::: {.cell .code execution_count="97"}
``` python
X_test = pd.DataFrame(X_test, columns=[cols])
```
:::

::: {.cell .code execution_count="98"}
``` python
X_train.describe()
```

::: {.output .execute_result execution_count="98"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>...</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.484406</td>
      <td>0.530004</td>
      <td>0.210962</td>
      <td>0.236312</td>
      <td>0.554562</td>
      <td>0.262667</td>
      <td>0.254148</td>
      <td>0.326575</td>
      <td>0.688675</td>
      <td>0.515095</td>
      <td>...</td>
      <td>0.054530</td>
      <td>0.060288</td>
      <td>0.067259</td>
      <td>0.101605</td>
      <td>0.064059</td>
      <td>0.056402</td>
      <td>0.064464</td>
      <td>0.069334</td>
      <td>0.060798</td>
      <td>0.065483</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.151741</td>
      <td>0.134105</td>
      <td>0.369949</td>
      <td>0.129528</td>
      <td>0.190999</td>
      <td>0.101682</td>
      <td>0.160119</td>
      <td>0.152384</td>
      <td>0.189356</td>
      <td>0.205307</td>
      <td>...</td>
      <td>0.227061</td>
      <td>0.238021</td>
      <td>0.250471</td>
      <td>0.302130</td>
      <td>0.244860</td>
      <td>0.230698</td>
      <td>0.245578</td>
      <td>0.254022</td>
      <td>0.238960</td>
      <td>0.247378</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.375297</td>
      <td>0.431002</td>
      <td>0.000000</td>
      <td>0.183486</td>
      <td>0.565517</td>
      <td>0.193798</td>
      <td>0.127273</td>
      <td>0.228070</td>
      <td>0.570000</td>
      <td>0.370000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.479810</td>
      <td>0.517958</td>
      <td>0.000000</td>
      <td>0.220183</td>
      <td>0.586207</td>
      <td>0.255814</td>
      <td>0.236364</td>
      <td>0.333333</td>
      <td>0.700000</td>
      <td>0.520000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.593824</td>
      <td>0.623819</td>
      <td>0.187500</td>
      <td>0.247706</td>
      <td>0.600000</td>
      <td>0.310078</td>
      <td>0.345455</td>
      <td>0.421053</td>
      <td>0.830000</td>
      <td>0.650000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 118 columns</p>
</div>
```
:::
:::

::: {.cell .markdown}
We now have `X_train` dataset ready to be fed into the Logistic
Regression classifier. I will do it as follows.
:::

::: {.cell .markdown}
# **12. Model training** `<a class="anchor" id="12">`{=html}`</a>`{=html} {#12-model-training-}

[Table of Contents](#0.1)
:::

::: {.cell .code execution_count="99"}
``` python
# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression


# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)
```

::: {.output .execute_result execution_count="99"}
    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)
:::
:::

::: {.cell .markdown}
# **13. Predict results** `<a class="anchor" id="13">`{=html}`</a>`{=html} {#13-predict-results-}

[Table of Contents](#0.1)
:::

::: {.cell .code execution_count="100"}
``` python
y_pred_test = logreg.predict(X_test)

y_pred_test
```

::: {.output .execute_result execution_count="100"}
    array(['No', 'No', 'No', ..., 'No', 'No', 'Yes'], dtype=object)
:::
:::

::: {.cell .markdown}
### predict_proba method

**predict_proba** method gives the probabilities for the target
variable(0 and 1) in this case, in array form.

`0 is for probability of no rain` and `1 is for probability of rain.`
:::

::: {.cell .code execution_count="101"}
``` python
# probability of getting output as 0 - no rain

logreg.predict_proba(X_test)[:,0]
```

::: {.output .execute_result execution_count="101"}
    array([0.91382428, 0.83565645, 0.82033915, ..., 0.97674285, 0.79855098,
           0.30734161])
:::
:::

::: {.cell .code execution_count="102"}
``` python
# probability of getting output as 1 - rain

logreg.predict_proba(X_test)[:,1]
```

::: {.output .execute_result execution_count="102"}
    array([0.08617572, 0.16434355, 0.17966085, ..., 0.02325715, 0.20144902,
           0.69265839])
:::
:::

::: {.cell .markdown}
# **14. Check accuracy score** `<a class="anchor" id="14">`{=html}`</a>`{=html} {#14-check-accuracy-score-}

[Table of Contents](#0.1)
:::

::: {.cell .code execution_count="103"}
``` python
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
```

::: {.output .stream .stdout}
    Model accuracy score: 0.8502
:::
:::

::: {.cell .markdown}
Here, **y_test** are the true class labels and **y_pred_test** are the
predicted class labels in the test-set.
:::

::: {.cell .markdown}
### Compare the train-set and test-set accuracy

Now, I will compare the train-set and test-set accuracy to check for
overfitting.
:::

::: {.cell .code execution_count="104"}
``` python
y_pred_train = logreg.predict(X_train)

y_pred_train
```

::: {.output .execute_result execution_count="104"}
    array(['No', 'No', 'No', ..., 'No', 'No', 'No'], dtype=object)
:::
:::

::: {.cell .code execution_count="105"}
``` python
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
```

::: {.output .stream .stdout}
    Training-set accuracy score: 0.8476
:::
:::

::: {.cell .markdown}
### Check for overfitting and underfitting
:::

::: {.cell .code execution_count="106"}
``` python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))
```

::: {.output .stream .stdout}
    Training set score: 0.8476
    Test set score: 0.8502
:::
:::

::: {.cell .markdown}
The training-set accuracy score is 0.8476 while the test-set accuracy to
be 0.8501. These two values are quite comparable. So, there is no
question of overfitting.
:::

::: {.cell .markdown}
In Logistic Regression, we use default value of C = 1. It provides good
performance with approximately 85% accuracy on both the training and the
test set. But the model performance on both the training and test set
are very comparable. It is likely the case of underfitting.

I will increase C and fit a more flexible model.
:::

::: {.cell .code execution_count="107"}
``` python
# fit the Logsitic Regression model with C=100

# instantiate the model
logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)


# fit the model
logreg100.fit(X_train, y_train)
```

::: {.output .execute_result execution_count="107"}
    LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)
:::
:::

::: {.cell .code execution_count="108"}
``` python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))
```

::: {.output .stream .stdout}
    Training set score: 0.8478
    Test set score: 0.8505
:::
:::

::: {.cell .markdown}
We can see that, C=100 results in higher test set accuracy and also a
slightly increased training set accuracy. So, we can conclude that a
more complex model should perform better.
:::

::: {.cell .markdown}
Now, I will investigate, what happens if we use more regularized model
than the default value of C=1, by setting C=0.01.
:::

::: {.cell .code execution_count="109"}
``` python
# fit the Logsitic Regression model with C=001

# instantiate the model
logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)


# fit the model
logreg001.fit(X_train, y_train)
```

::: {.output .execute_result execution_count="109"}
    LogisticRegression(C=0.01, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)
:::
:::

::: {.cell .code execution_count="110"}
``` python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))
```

::: {.output .stream .stdout}
    Training set score: 0.8409
    Test set score: 0.8448
:::
:::

::: {.cell .markdown}
So, if we use more regularized model by setting C=0.01, then both the
training and test set accuracy decrease relatiev to the default
parameters.
:::

::: {.cell .markdown}
### Compare model accuracy with null accuracy

So, the model accuracy is 0.8501. But, we cannot say that our model is
very good based on the above accuracy. We must compare it with the
**null accuracy**. Null accuracy is the accuracy that could be achieved
by always predicting the most frequent class.

So, we should first check the class distribution in the test set.
:::

::: {.cell .code execution_count="111"}
``` python
# check class distribution in test set

y_test.value_counts()
```

::: {.output .execute_result execution_count="111"}
    No     22067
    Yes     6372
    Name: RainTomorrow, dtype: int64
:::
:::

::: {.cell .markdown}
We can see that the occurences of most frequent class is 22067. So, we
can calculate null accuracy by dividing 22067 by total number of
occurences.
:::

::: {.cell .code execution_count="112"}
``` python
# check null accuracy score

null_accuracy = (22067/(22067+6372))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
```

::: {.output .stream .stdout}
    Null accuracy score: 0.7759
:::
:::

::: {.cell .markdown}
We can see that our model accuracy score is 0.8501 but null accuracy
score is 0.7759. So, we can conclude that our Logistic Regression model
is doing a very good job in predicting the class labels.
:::

::: {.cell .markdown}
Now, based on the above analysis we can conclude that our classification
model accuracy is very good. Our model is doing a very good job in terms
of predicting the class labels.

But, it does not give the underlying distribution of values. Also, it
does not tell anything about the type of errors our classifer is making.

We have another tool called `Confusion matrix` that comes to our rescue.
:::

::: {.cell .markdown}
# **15. Confusion matrix** `<a class="anchor" id="15">`{=html}`</a>`{=html} {#15-confusion-matrix-}

[Table of Contents](#0.1)

A confusion matrix is a tool for summarizing the performance of a
classification algorithm. A confusion matrix will give us a clear
picture of classification model performance and the types of errors
produced by the model. It gives us a summary of correct and incorrect
predictions broken down by each category. The summary is represented in
a tabular form.

Four types of outcomes are possible while evaluating a classification
model performance. These four outcomes are described below:-

**True Positives (TP)** -- True Positives occur when we predict an
observation belongs to a certain class and the observation actually
belongs to that class.

**True Negatives (TN)** -- True Negatives occur when we predict an
observation does not belong to a certain class and the observation
actually does not belong to that class.

**False Positives (FP)** -- False Positives occur when we predict an
observation belongs to a certain class but the observation actually does
not belong to that class. This type of error is called **Type I error.**

**False Negatives (FN)** -- False Negatives occur when we predict an
observation does not belong to a certain class but the observation
actually belongs to that class. This is a very serious error and it is
called **Type II error.**

These four outcomes are summarized in a confusion matrix given below.
:::

::: {.cell .code execution_count="113"}
``` python
# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
```

::: {.output .stream .stdout}
    Confusion matrix

     [[20892  1175]
     [ 3086  3286]]

    True Positives(TP) =  20892

    True Negatives(TN) =  3286

    False Positives(FP) =  1175

    False Negatives(FN) =  3086
:::
:::

::: {.cell .markdown}
The confusion matrix shows `20892 + 3285 = 24177 correct predictions`
and `3087 + 1175 = 4262 incorrect predictions`.

In this case, we have

-   `True Positives` (Actual Positive:1 and Predict Positive:1) - 20892

-   `True Negatives` (Actual Negative:0 and Predict Negative:0) - 3285

-   `False Positives` (Actual Negative:0 but Predict Positive:1) - 1175
    `(Type I error)`

-   `False Negatives` (Actual Positive:1 but Predict Negative:0) - 3087
    `(Type II error)`
:::

::: {.cell .code execution_count="114"}
``` python
# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
```

::: {.output .execute_result execution_count="114"}
    <matplotlib.axes._subplots.AxesSubplot at 0x7f28b1306208>
:::

::: {.output .display_data}
![](vertopal_ca93cbc8c7524834a21e37d8a1959b04/a4a53ceadee5563daa36435d81e8dadcf13c3f59.png)
:::
:::

::: {.cell .markdown}
# **16. Classification metrices** `<a class="anchor" id="16">`{=html}`</a>`{=html} {#16-classification-metrices-}

[Table of Contents](#0.1)
:::

::: {.cell .markdown}
## Classification Report

**Classification report** is another way to evaluate the classification
model performance. It displays the **precision**, **recall**, **f1** and
**support** scores for the model. I have described these terms in later.

We can print a classification report as follows:-
:::

::: {.cell .code execution_count="115"}
``` python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))
```

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

              No       0.87      0.95      0.91     22067
             Yes       0.74      0.52      0.61      6372

        accuracy                           0.85     28439
       macro avg       0.80      0.73      0.76     28439
    weighted avg       0.84      0.85      0.84     28439
:::
:::

::: {.cell .markdown}
## Classification accuracy
:::

::: {.cell .code execution_count="116"}
``` python
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
```
:::

::: {.cell .code execution_count="117"}
``` python
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
```

::: {.output .stream .stdout}
    Classification accuracy : 0.8502
:::
:::

::: {.cell .markdown}
## Classification error
:::

::: {.cell .code execution_count="118"}
``` python
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))
```

::: {.output .stream .stdout}
    Classification error : 0.1498
:::
:::

::: {.cell .markdown}
## Precision

**Precision** can be defined as the percentage of correctly predicted
positive outcomes out of all the predicted positive outcomes. It can be
given as the ratio of true positives (TP) to the sum of true and false
positives (TP + FP).

So, **Precision** identifies the proportion of correctly predicted
positive outcome. It is more concerned with the positive class than the
negative class.

Mathematically, precision can be defined as the ratio of
`TP to (TP + FP).`
:::

::: {.cell .code execution_count="119"}
``` python
# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))
```

::: {.output .stream .stdout}
    Precision : 0.9468
:::
:::

::: {.cell .markdown}
## Recall

Recall can be defined as the percentage of correctly predicted positive
outcomes out of all the actual positive outcomes. It can be given as the
ratio of true positives (TP) to the sum of true positives and false
negatives (TP + FN). **Recall** is also called **Sensitivity**.

**Recall** identifies the proportion of correctly predicted actual
positives.

Mathematically, recall can be given as the ratio of `TP to (TP + FN).`
:::

::: {.cell .code execution_count="120"}
``` python
recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))
```

::: {.output .stream .stdout}
    Recall or Sensitivity : 0.8713
:::
:::

::: {.cell .markdown}
## True Positive Rate

**True Positive Rate** is synonymous with **Recall**.
:::

::: {.cell .code execution_count="121"}
``` python
true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
```

::: {.output .stream .stdout}
    True Positive Rate : 0.8713
:::
:::

::: {.cell .markdown}
## False Positive Rate
:::

::: {.cell .code execution_count="122"}
``` python
false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
```

::: {.output .stream .stdout}
    False Positive Rate : 0.2634
:::
:::

::: {.cell .markdown}
## Specificity
:::

::: {.cell .code execution_count="123"}
``` python
specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))
```

::: {.output .stream .stdout}
    Specificity : 0.7366
:::
:::

::: {.cell .markdown}
## f1-score

**f1-score** is the weighted harmonic mean of precision and recall. The
best possible **f1-score** would be 1.0 and the worst would be 0.0.
**f1-score** is the harmonic mean of precision and recall. So,
**f1-score** is always lower than accuracy measures as they embed
precision and recall into their computation. The weighted average of
`f1-score` should be used to compare classifier models, not global
accuracy.
:::

::: {.cell .markdown}
## Support

**Support** is the actual number of occurrences of the class in our
dataset.
:::

::: {.cell .markdown}
# **17. Adjusting the threshold level** `<a class="anchor" id="17">`{=html}`</a>`{=html} {#17-adjusting-the-threshold-level-}

[Table of Contents](#0.1)
:::

::: {.cell .code execution_count="124"}
``` python
# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = logreg.predict_proba(X_test)[0:10]

y_pred_prob
```

::: {.output .execute_result execution_count="124"}
    array([[0.91382428, 0.08617572],
           [0.83565645, 0.16434355],
           [0.82033915, 0.17966085],
           [0.99025322, 0.00974678],
           [0.95726711, 0.04273289],
           [0.97993908, 0.02006092],
           [0.17833011, 0.82166989],
           [0.23480918, 0.76519082],
           [0.90048436, 0.09951564],
           [0.85485267, 0.14514733]])
:::
:::

::: {.cell .markdown}
### Observations

-   In each row, the numbers sum to 1.

-   There are 2 columns which correspond to 2 classes - 0 and 1.

    -   Class 0 - predicted probability that there is no rain tomorrow.

    -   Class 1 - predicted probability that there is rain tomorrow.

-   Importance of predicted probabilities

    -   We can rank the observations by probability of rain or no rain.

-   predict_proba process

    -   Predicts the probabilities

    -   Choose the class with the highest probability

-   Classification threshold level

    -   There is a classification threshold level of 0.5.

    -   Class 1 - probability of rain is predicted if probability \>
        0.5.

    -   Class 0 - probability of no rain is predicted if probability \<
        0.5.
:::

::: {.cell .code execution_count="125"}
``` python
# store the probabilities in dataframe

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - No rain tomorrow (0)', 'Prob of - Rain tomorrow (1)'])

y_pred_prob_df
```

::: {.output .execute_result execution_count="125"}
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
      <th>Prob of - No rain tomorrow (0)</th>
      <th>Prob of - Rain tomorrow (1)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.913824</td>
      <td>0.086176</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.835656</td>
      <td>0.164344</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.820339</td>
      <td>0.179661</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.990253</td>
      <td>0.009747</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.957267</td>
      <td>0.042733</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.979939</td>
      <td>0.020061</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.178330</td>
      <td>0.821670</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.234809</td>
      <td>0.765191</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.900484</td>
      <td>0.099516</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.854853</td>
      <td>0.145147</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="126"}
``` python
# print the first 10 predicted probabilities for class 1 - Probability of rain

logreg.predict_proba(X_test)[0:10, 1]
```

::: {.output .execute_result execution_count="126"}
    array([0.08617572, 0.16434355, 0.17966085, 0.00974678, 0.04273289,
           0.02006092, 0.82166989, 0.76519082, 0.09951564, 0.14514733])
:::
:::

::: {.cell .code execution_count="127"}
``` python
# store the predicted probabilities for class 1 - Probability of rain

y_pred1 = logreg.predict_proba(X_test)[:, 1]
```
:::

::: {.cell .code execution_count="128"}
``` python
# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of rain')


# set the x-axis limit
plt.xlim(0,1)


# set the title
plt.xlabel('Predicted probabilities of rain')
plt.ylabel('Frequency')
```

::: {.output .execute_result execution_count="128"}
    Text(0, 0.5, 'Frequency')
:::

::: {.output .display_data}
![](vertopal_ca93cbc8c7524834a21e37d8a1959b04/0418dfa93fdc461b31142c470273f3eb5e4c9d25.png)
:::
:::

::: {.cell .markdown}
### Observations {#observations}

-   We can see that the above histogram is highly positive skewed.

-   The first column tell us that there are approximately 15000
    observations with probability between 0.0 and 0.1.

-   There are small number of observations with probability \> 0.5.

-   So, these small number of observations predict that there will be
    rain tomorrow.

-   Majority of observations predict that there will be no rain
    tomorrow.
:::

::: {.cell .markdown}
### Lower the threshold
:::

::: {.cell .code execution_count="129"}
``` python
from sklearn.preprocessing import binarize

for i in range(1,5):
    
    cm1=0
    
    y_pred1 = logreg.predict_proba(X_test)[:,1]
    
    y_pred1 = y_pred1.reshape(-1,1)
    
    y_pred2 = binarize(y_pred1, i/10)
    
    y_pred2 = np.where(y_pred2 == 1, 'Yes', 'No')
    
    cm1 = confusion_matrix(y_test, y_pred2)
        
    print ('With',i/10,'threshold the Confusion Matrix is ','\n\n',cm1,'\n\n',
           
            'with',cm1[0,0]+cm1[1,1],'correct predictions, ', '\n\n', 
           
            cm1[0,1],'Type I errors( False Positives), ','\n\n',
           
            cm1[1,0],'Type II errors( False Negatives), ','\n\n',
           
           'Accuracy score: ', (accuracy_score(y_test, y_pred2)), '\n\n',
           
           'Sensitivity: ',cm1[1,1]/(float(cm1[1,1]+cm1[1,0])), '\n\n',
           
           'Specificity: ',cm1[0,0]/(float(cm1[0,0]+cm1[0,1])),'\n\n',
          
            '====================================================', '\n\n')
```

::: {.output .stream .stdout}
    With 0.1 threshold the Confusion Matrix is  

     [[12726  9341]
     [  547  5825]] 

     with 18551 correct predictions,  

     9341 Type I errors( False Positives),  

     547 Type II errors( False Negatives),  

     Accuracy score:  0.6523084496641935 

     Sensitivity:  0.9141556811048337 

     Specificity:  0.5766982371867494 

     ==================================================== 


    With 0.2 threshold the Confusion Matrix is  

     [[17066  5001]
     [ 1234  5138]] 

     with 22204 correct predictions,  

     5001 Type I errors( False Positives),  

     1234 Type II errors( False Negatives),  

     Accuracy score:  0.7807588171173389 

     Sensitivity:  0.8063402385436284 

     Specificity:  0.7733720034440568 

     ==================================================== 


    With 0.3 threshold the Confusion Matrix is  

     [[19080  2987]
     [ 1872  4500]] 

     with 23580 correct predictions,  

     2987 Type I errors( False Positives),  

     1872 Type II errors( False Negatives),  

     Accuracy score:  0.8291430781673055 

     Sensitivity:  0.7062146892655368 

     Specificity:  0.8646395069560883 

     ==================================================== 


    With 0.4 threshold the Confusion Matrix is  

     [[20191  1876]
     [ 2517  3855]] 

     with 24046 correct predictions,  

     1876 Type I errors( False Positives),  

     2517 Type II errors( False Negatives),  

     Accuracy score:  0.845529027040332 

     Sensitivity:  0.6049905838041432 

     Specificity:  0.9149861784565188 

     ==================================================== 
:::
:::

::: {.cell .markdown}
### Comments

-   In binary problems, the threshold of 0.5 is used by default to
    convert predicted probabilities into class predictions.

-   Threshold can be adjusted to increase sensitivity or specificity.

-   Sensitivity and specificity have an inverse relationship. Increasing
    one would always decrease the other and vice versa.

-   We can see that increasing the threshold level results in increased
    accuracy.

-   Adjusting the threshold level should be one of the last step you do
    in the model-building process.
:::

::: {.cell .markdown}
# **18. ROC - AUC** `<a class="anchor" id="18">`{=html}`</a>`{=html} {#18-roc---auc-}

[Table of Contents](#0.1)

## ROC Curve

Another tool to measure the classification model performance visually is
**ROC Curve**. ROC Curve stands for **Receiver Operating Characteristic
Curve**. An **ROC Curve** is a plot which shows the performance of a
classification model at various classification threshold levels.

The **ROC Curve** plots the **True Positive Rate (TPR)** against the
**False Positive Rate (FPR)** at various threshold levels.

**True Positive Rate (TPR)** is also called **Recall**. It is defined as
the ratio of `TP to (TP + FN).`

**False Positive Rate (FPR)** is defined as the ratio of
`FP to (FP + TN).`

In the ROC Curve, we will focus on the TPR (True Positive Rate) and FPR
(False Positive Rate) of a single point. This will give us the general
performance of the ROC curve which consists of the TPR and FPR at
various threshold levels. So, an ROC Curve plots TPR vs FPR at different
classification threshold levels. If we lower the threshold levels, it
may result in more items being classified as positve. It will increase
both True Positives (TP) and False Positives (FP).
:::

::: {.cell .code execution_count="130"}
``` python
# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = 'Yes')

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
```

::: {.output .display_data}
![](vertopal_ca93cbc8c7524834a21e37d8a1959b04/c8c3ad2359f5da11898f9fc99da27a2ab50f75d5.png)
:::
:::

::: {.cell .markdown}
ROC curve help us to choose a threshold level that balances sensitivity
and specificity for a particular context.
:::

::: {.cell .markdown}
## ROC-AUC

**ROC AUC** stands for **Receiver Operating Characteristic - Area Under
Curve**. It is a technique to compare classifier performance. In this
technique, we measure the `area under the curve (AUC)`. A perfect
classifier will have a ROC AUC equal to 1, whereas a purely random
classifier will have a ROC AUC equal to 0.5.

So, **ROC AUC** is the percentage of the ROC plot that is underneath the
curve.
:::

::: {.cell .code execution_count="131"}
``` python
# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))
```

::: {.output .stream .stdout}
    ROC AUC : 0.8729
:::
:::

::: {.cell .markdown}
### Comments {#comments}

-   ROC AUC is a single number summary of classifier performance. The
    higher the value, the better the classifier.

-   ROC AUC of our model approaches towards 1. So, we can conclude that
    our classifier does a good job in predicting whether it will rain
    tomorrow or not.
:::

::: {.cell .code execution_count="132"}
``` python
# calculate cross-validated ROC AUC 

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(logreg, X_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))
```

::: {.output .stream .stdout}
    Cross validated ROC AUC : 0.8695
:::
:::

::: {.cell .markdown}
# **19. k-Fold Cross Validation** `<a class="anchor" id="19">`{=html}`</a>`{=html} {#19-k-fold-cross-validation-}

[Table of Contents](#0.1)
:::

::: {.cell .code execution_count="133"}
``` python
# Applying 5-Fold Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))
```

::: {.output .stream .stdout}
    Cross-validation scores:[0.84686387 0.84624852 0.84633642 0.84963298 0.84773626]
:::
:::

::: {.cell .markdown}
We can summarize the cross-validation accuracy by calculating its mean.
:::

::: {.cell .code execution_count="134"}
``` python
# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))
```

::: {.output .stream .stdout}
    Average cross-validation score: 0.8474
:::
:::

::: {.cell .markdown}
Our, original model score is found to be 0.8476. The average
cross-validation score is 0.8474. So, we can conclude that
cross-validation does not result in performance improvement.
:::

::: {.cell .markdown}
# **20. Hyperparameter Optimization using GridSearch CV** `<a class="anchor" id="20">`{=html}`</a>`{=html} {#20-hyperparameter-optimization-using-gridsearch-cv-}

[Table of Contents](#0.1)
:::

::: {.cell .code execution_count="135" scrolled="true"}
``` python
from sklearn.model_selection import GridSearchCV


parameters = [{'penalty':['l1','l2']}, 
              {'C':[1, 10, 100, 1000]}]



grid_search = GridSearchCV(estimator = logreg,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)


grid_search.fit(X_train, y_train)
```

::: {.output .execute_result execution_count="135"}
    GridSearchCV(cv=5, error_score='raise-deprecating',
                 estimator=LogisticRegression(C=1.0, class_weight=None, dual=False,
                                              fit_intercept=True,
                                              intercept_scaling=1, l1_ratio=None,
                                              max_iter=100, multi_class='warn',
                                              n_jobs=None, penalty='l2',
                                              random_state=0, solver='liblinear',
                                              tol=0.0001, verbose=0,
                                              warm_start=False),
                 iid='warn', n_jobs=None,
                 param_grid=[{'penalty': ['l1', 'l2']}, {'C': [1, 10, 100, 1000]}],
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=0)
:::
:::

::: {.cell .code execution_count="136"}
``` python
# examine the best model

# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))

# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))

# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))
```

::: {.output .stream .stdout}
    GridSearch CV best score : 0.8474


    Parameters that give the best results : 

     {'penalty': 'l1'}


    Estimator that was chosen by the search : 

     LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l1',
                       random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)
:::
:::

::: {.cell .code execution_count="137"}
``` python
# calculate GridSearch CV score on test set

print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))
```

::: {.output .stream .stdout}
    GridSearch CV score on test set: 0.8507
:::
:::

::: {.cell .markdown}
### Comments {#comments}

-   Our original model test accuracy is 0.8501 while GridSearch CV
    accuracy is 0.8507.

-   We can see that GridSearch CV improve the performance for this
    particular model.
:::

::: {.cell .markdown}
# **21. Results and conclusion** `<a class="anchor" id="21">`{=html}`</a>`{=html} {#21-results-and-conclusion-}

[Table of Contents](#0.1)
:::

::: {.cell .markdown}
1.  The logistic regression model accuracy score is 0.8501. So, the
    model does a very good job in predicting whether or not it will rain
    tomorrow in Australia.

2.  Small number of observations predict that there will be rain
    tomorrow. Majority of observations predict that there will be no
    rain tomorrow.

3.  The model shows no signs of overfitting.

4.  Increasing the value of C results in higher test set accuracy and
    also a slightly increased training set accuracy. So, we can conclude
    that a more complex model should perform better.

5.  Increasing the threshold level results in increased accuracy.

6.  ROC AUC of our model approaches towards 1. So, we can conclude that
    our classifier does a good job in predicting whether it will rain
    tomorrow or not.

7.  Our original model accuracy score is 0.8501 whereas accuracy score
    after RFECV is 0.8500. So, we can obtain approximately similar
    accuracy but with reduced set of features.

8.  In the original model, we have FP = 1175 whereas FP1 = 1174. So, we
    get approximately same number of false positives. Also, FN = 3087
    whereas FN1 = 3091. So, we get slighly higher false negatives.

9.  Our, original model score is found to be 0.8476. The average
    cross-validation score is 0.8474. So, we can conclude that
    cross-validation does not result in performance improvement.

10. Our original model test accuracy is 0.8501 while GridSearch CV
    accuracy is 0.8507. We can see that GridSearch CV improve the
    performance for this particular model.
:::

::: {.cell .markdown}
# **22. References** `<a class="anchor" id="22">`{=html}`</a>`{=html} {#22-references-}

[Table of Contents](#0.1)

The work done in this project is inspired from following books and
websites:-

1.  Hands on Machine Learning with Scikit-Learn and Tensorflow by
    Aurélién Géron

2.  Introduction to Machine Learning with Python by Andreas C. Müller
    and Sarah Guido

3.  Udemy course -- Machine Learning -- A Z by Kirill Eremenko and
    Hadelin de Ponteves

4.  Udemy course -- Feature Engineering for Machine Learning by Soledad
    Galli

5.  Udemy course -- Feature Selection for Machine Learning by Soledad
    Galli

6.  <https://en.wikipedia.org/wiki/Logistic_regression>

7.  <https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html>

8.  <https://en.wikipedia.org/wiki/Sigmoid_function>

9.  <https://www.statisticssolutions.com/assumptions-of-logistic-regression/>

10. <https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python>

11. <https://www.kaggle.com/neisha/heart-disease-prediction-using-logistic-regression>

12. <https://www.ritchieng.com/machine-learning-evaluate-classification-model/>
:::

::: {.cell .markdown}
So, now we will come to the end of this kernel.

I hope you find this kernel useful and enjoyable.

Your comments and feedback are most welcome.

Thank you
:::

::: {.cell .markdown}
[Go to Top](#0)
:::
