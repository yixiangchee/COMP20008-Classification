# COMP20008-Classification

COMP20008 Project 2 Part 2: Classification

Marks: 12/13

An exploration of different classification algorithms and feature engineering.

Each year, the World Bank publishes the World Development Indicators which provide high quality and international comparable statistics about global development and the fight against poverty. 

As data scientists, we wish to understand how the information can be used to predict average lifespan in different countries.

All files and source codes are located in the /src folder.

[Project Specification](/proj2_spec.pdf)  

[world.csv (World Development Indicators)](/src/world.csv)  

[life.csv (Life Expectancy)](/src/life.csv)  


## Comparing Classification Algorithms

[Script](/src/task2a.py)

Compare the performance of the following 3 classification algorithms: 

1. k-NN (k=3) 
2. k-NN (k=7) 
3. Decision tree (with a maximum depth of 3)

**Libraries Used**

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
```

**Load CSV's**

```python
life=pd.read_csv('life.csv',encoding = 'ISO-8859-1',na_values='..')
world=pd.read_csv('world.csv',encoding = 'ISO-8859-1',na_values='..')

result = pd.merge(life, world, on=['Country Code'])
result = result.sort_values(by='Country Code', ascending = True)
```

**Extract Features and Class Labels**

```python
data=result[['Access to electricity (% of population) [EG.ELC.ACCS.ZS]',
 'Adjusted net national income per capita (current US$) [NY.ADJ.NNTY.PC.CD]',
 'Age dependency ratio (% of working-age population) [SP.POP.DPND]',
 'Cause of death, by communicable diseases and maternal, prenatal and nutrition conditions (% of total) [SH.DTH.COMM.ZS]',
 'Current health expenditure per capita (current US$) [SH.XPD.CHEX.PC.CD]',
 'Fertility rate, total (births per woman) [SP.DYN.TFRT.IN]',
 'Fixed broadband subscriptions (per 100 people) [IT.NET.BBND.P2]',
 'Fixed telephone subscriptions (per 100 people) [IT.MLT.MAIN.P2]',
 'GDP per capita (constant 2010 US$) [NY.GDP.PCAP.KD]',
 'GNI per capita, Atlas method (current US$) [NY.GNP.PCAP.CD]',
 'Individuals using the Internet (% of population) [IT.NET.USER.ZS]',
 'Lifetime risk of maternal death (%) [SH.MMR.RISK.ZS]',
 'People using at least basic drinking water services (% of population) [SH.H2O.BASW.ZS]',
 'People using at least basic drinking water services, rural (% of rural population) [SH.H2O.BASW.RU.ZS]',
 'People using at least basic drinking water services, urban (% of urban population) [SH.H2O.BASW.UR.ZS]',
 'People using at least basic sanitation services, urban (% of urban population) [SH.STA.BASS.UR.ZS]',
 'Prevalence of anemia among children (% of children under 5) [SH.ANM.CHLD.ZS]',
 'Secure Internet servers (per 1 million people) [IT.NET.SECR.P6]',
 'Self-employed, female (% of female employment) (modeled ILO estimate) [SL.EMP.SELF.FE.ZS]',
 'Wage and salaried workers, female (% of female employment) (modeled ILO estimate) [SL.EMP.WORK.FE.ZS]']].astype(float)

classlabel=result['Life expectancy at birth (years)']
```

**Fit and Normalise Models**

For each of the algorithms, a model is fit with the following processing steps:

* Split the dataset into a training set comprising 70% of the data and a test set comprising the remaining 30% using the train test split function with a random state of 200.
* Perform the same imputation and scaling to the training set:

⋅⋅⋅For each feature, perform median imputation to impute missing values. 

⋅⋅⋅Scale each feature by removing the mean and scaling to unit variance.

* Train the classifiers using the training set.
* Test the classifiers by applying them to the test set.

```python
X_train, X_test, y_train, y_test = train_test_split(data, classlabel, train_size=0.7, test_size=0.3, random_state=200)
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

med = X_train.median()

#normalise the data to have 0 mean and unit variance using the library functions. This will help for later computation of distances between instances
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
```

**Create CSV**

Create "task2a.csv", describing: 
* median used for imputation for each feature
* mean and variance used for scaling

```python
writerdf=pd.DataFrame({'feature': data.columns, 'median':med,'mean': scaler.mean_,'variance': scaler.var_})
writerdf=writerdf.round(decimals=3)
writerdf.to_csv(r'task2a.csv', index = False)
```

**Outputs Accuracy**

**Decision Tree**

```python
dt = DecisionTreeClassifier(random_state=200, max_depth=3)
dt.fit(X_train, y_train)
y_pred=dt.predict(X_test)
print(f"Accuracy of decision tree: {accuracy_score(y_test, y_pred):.3f}")
```

**K-NN (N=3)**

```python
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
print(f"Accuracy of k-nn (k=3): {accuracy_score(y_test, y_pred):.3f}")
```

**K-NN (N=7)**

```python
knn = neighbors.KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
print(f"Accuracy of k-nn (k=7): {accuracy_score(y_test, y_pred):.3f}")
```

**Observation**

For k-nn, k=7 (accuracy of 0.727) performed best, and k=3 (accuracy of 0.673) weaker than k=7. The Decision Tree algorithms performance was weaker than knn=7 but better than knn=3 on this dataset (accuracy of 0.709), hence k-nn performed better, with k=7.

Higher k of k-nn performed better because it is less sensitive to the noises present in the dataset, also k-nn of k=7 performed better than the decision tree since the maximum depth allowed for the decision tree was only 3, thus making it less accurate.

---

## Feature Engineering and Selection

[Script](/src/task2b.py)

