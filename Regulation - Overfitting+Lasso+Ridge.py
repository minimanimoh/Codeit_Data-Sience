#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Regulation(정규화): 머신러닝모델을 학습시킬 때 세타 값들이 너무 커지는 것을 방지해주는 기법 
그렇게 하기 위해 손실함수에 정규화 항을 더해줌 = 가설함수 평가 기준을 변경함
종류에는
- Lasso Model (L1 Regulation)
- Ridge Model (L2 Regulation)
정규화 항을 추가하면 평균제곱오차와 세타 값 둘 다 최소화 할 수 있음. 어떤 것을 더 줄이는 것이 중요한지는 상수 람다가 결정
- 람다 클수록: 세타 값 줄이는 것이 중요, 람다 작을수록 평균제곱오차 줄이는 것이 중요


# In[1]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from math import sqrt

import pandas as pd
import numpy as np


# In[2]:


ADMISSION_FILE_PATH = 'Downloads/admission_data.csv'


# In[3]:


admission_df = pd.read_csv(ADMISSION_FILE_PATH)


# In[4]:


admission_df.head()


# In[5]:


admission_df = pd.read_csv(ADMISSION_FILE_PATH).drop('Serial No.', axis=1)


# In[6]:


admission_df.head()


# In[9]:


#input variable
X = admission_df.drop(['Chance of Admit '], axis=1)


# In[10]:


#육차항 변형기
Polynomial_transformer = PolynomialFeatures(6)


# In[11]:


Polynomial_features = Polynomial_transformer.fit_transform(X.values)


# In[12]:


features = Polynomial_transformer.get_feature_names(X.columns)


# In[13]:


X = pd.DataFrame(Polynomial_features, columns=features)


# In[16]:


X.head()


# In[17]:


# target variables
y = admission_df[['Chance of Admit ']]


# In[18]:


y.head()


# In[19]:


# 1. divide training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)


# In[21]:


# 2. data learning
model = LinearRegression()
model.fit(X_train, y_train)


# In[28]:


# 3. compare results of train/test sets
# 3-1. calculate the estimate by using the training set
y_train_predict = model.predict(X_train)
# 3-2. calculate the estimate by using the test set
y_test_predict = model.predict(X_test)


# In[38]:


# 4. evaluate the result of training/test set by using MSE
mse_1 = mean_squared_error(y_train, y_train_predict)
print("training set performance: ")
print(sqrt(mse_1))
mse_2 = mean_squared_error(y_test, y_test_predict)
print("test set performance: ")
print(sqrt(mse_2))


# In[43]:


#Lasso Model (L1 Regulation)
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=0.001, max_iter=1000, normalize=True)
lasso_model.fit(X_train, y_train)
#alpha=lambda


# In[44]:


y_train_predict_2 = lasso_model.predict(X_train)
y_test_predict_2 = lasso_model.predict(X_test)


# In[45]:


mse_3 = mean_squared_error(y_train, y_train_predict_2)
print("training set performance: ")
print(sqrt(mse_1))
mse_4 = mean_squared_error(y_test, y_test_predict_2)
print("test set performance: ")
print(sqrt(mse_2))


# In[ ]:


#Summary
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt

import numpy as np
import pandas as pd

# 데이터 파일 경로 정의
INSURANCE_FILE_PATH = './datasets/insurance.csv'

insurance_df = pd.read_csv(INSURANCE_FILE_PATH)  # 데이터를 pandas dataframe으로 갖고 온다 (insurance_df.head()를 사용해서 데이터를 한 번 살펴보세요!)
insurance_df = pd.get_dummies(data=insurance_df, columns=['sex', 'smoker', 'region'])  # 필요한 열들에 One-hot Encoding을 해준다

# 입력 변수 데이터를 따로 새로운 dataframe에 저장
X = insurance_df.drop(['charges'], axis=1)

polynomial_transformer = PolynomialFeatures(4)  # 4 차항 변형기를 정의
polynomial_features = polynomial_transformer.fit_transform(X.values)  #  4차 항 변수로 변환

features = polynomial_transformer.get_feature_names(X.columns)  # 새로운 변수 이름들 생성

X = pd.DataFrame(polynomial_features, columns=features)  # 다항 입력 변수를 dataframe으로 만들어 준다
y = insurance_df[['charges']]  # 목표 변수 정의

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

model = Lasso(alpha=1, max_iter=2000, normalize=True)
model.fit(X_train, y_train)

y_test_predict = model.predict(X_test)
y_train_predict = model.predict(X_train)

mse = mean_squared_error(y_train, y_train_predict)

print("training set에서 성능")
print("-----------------------")
print(f'오차: {sqrt(mse)}')

mse = mean_squared_error(y_test, y_test_predict)

print("testing set에서 성능")
print("-----------------------")
print(f'오차: {sqrt(mse)}')


# In[ ]:


#Ridge Model (L2 Regulation)
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt

import numpy as np
import pandas as pd

# 데이터 파일 경로 정의
INSURANCE_FILE_PATH = './datasets/insurance.csv'

insurance_df = pd.read_csv(INSURANCE_FILE_PATH)  # 데이터를 pandas dataframe으로 갖고 온다 (insurance_df.head()를 사용해서 데이터를 한 번 살펴보세요!)
insurance_df = pd.get_dummies(data=insurance_df, columns=['sex', 'smoker', 'region'])  # 필요한 열들에 One-hot Encoding을 해준다

# 입력 변수 데이터를 따로 새로운 dataframe에 저장
X = insurance_df.drop(['charges'], axis=1)

polynomial_transformer = PolynomialFeatures(4)  # 4 차항 변형기를 정의
polynomial_features = polynomial_transformer.fit_transform(X.values)  #  4차 항 변수로 변환

features = polynomial_transformer.get_feature_names(X.columns)  # 새로운 변수 이름들 생성

X = pd.DataFrame(polynomial_features, columns=features)  # 다항 입력 변수를 dataframe으로 만들어 준다
y = insurance_df[['charges']]  # 목표 변수 정의

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

model = Ridge(alpha=0.01, max_iter=2000, normalize=True)
model.fit(X_train, y_train)

y_test_predict = model.predict(X_test)
y_train_predict = model.predict(X_train)

mse = mean_squared_error(y_train, y_train_predict)

print("training set에서 성능")
print("-----------------------")
print(f'오차: {sqrt(mse)}')

mse = mean_squared_error(y_test, y_test_predict)

print("testing set에서 성능")
print("-----------------------")
print(f'오차: {sqrt(mse)}')

