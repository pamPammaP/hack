

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import numpy as np
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from category_encoders.binary import BinaryEncoder

from datetime import datetime

import re

def isday(s):
  if int(s) >= 0 and int(s) <= 31: return(True)
  else: return(False)
def ismonth(s):
  if int(s) >= 0 and int(s) <= 12: return(True)
  else: return(False)
def isyear(s):
  if int(s) >= 0 and int(s) <= 24: return(True)
  else: return(False)
def isfullyear(s):
  if int(s) >= 1900 and int(s) <= 2024: return(True)
  else: return(False)

def instruction1(sugdate):
  a = [0,0,0]
  a[2] = int(sugdate[0:2])
  a[1] = int(sugdate[2:4])
  a[0] = 2024
  return(a)
def instruction2(sugdate):
  a = [0,0,0]
  a[1] = int(sugdate[0:2])
  a[2] = int(sugdate[2:4])
  a[0] = 2024
  return(a)
def instruction3(sugdate):
  a = [0,0,0]
  a[1] = int(sugdate[0:2])
  a[0] = 2000 + int(sugdate[2:4])
  a[2] = 1
  return(a)
def instruction4(sugdate):
  a = [0,0,0]
  a[0] = 2000 + int(sugdate[0:2])
  a[1] = int(sugdate[2:4])
  a[2] = 1
  return(a)
def instruction5(sugdate):
  a = [0,0,0]
  a[2] = int(sugdate[0:2])
  a[1] = int(sugdate[2:4])
  a[0] = 2000 + int(sugdate[4:6])
  return(a)
def instruction6(sugdate):
  a = [0,0,0]
  a[0] = 2000 + int(sugdate[0:2])
  a[1] = int(sugdate[2:4])
  a[2] = int(sugdate[4:6])
  return(a)
def instruction7(sugdate):
  a = [0,0,0]
  a[2] = 1
  a[1] = int(sugdate[4:6])
  a[0] = int(sugdate[0:4])
  return(a)
def instruction8(sugdate):
  a = [0,0,0]
  a[2] = 1
  a[0] = int(sugdate[2:6])
  a[1] = int(sugdate[0:2])
  return(a)
def instruction9(sugdate):
  a = [0,0,0]
  a[2] = int(sugdate[0:2])
  a[1] = int(sugdate[2:4])
  a[0] = int(sugdate[4:8])
  return(a)
def instruction10(sugdate):
  a = [0,0,0]
  a[0] = int(sugdate[0:4])
  a[1] = int(sugdate[4:6])
  a[2] = int(sugdate[6:8])
  return(a)

def func(a):
  current_date = datetime.now()
  start_date = datetime(a[0], a[1], a[2])
  delta = current_date - start_date
  return delta.days

def find_month_in_string2(input_string):
    # Список названий месяцев
    months = [
        "январ", "феврал", "март", "апрел", "ма", "июн",
        "июл", "август", "сентябр", "октябр", "ноябр", "декабр"
    ]
    for ps in months:
      if ps in input_string.lower():
        return (months.index(ps) + 1)
    return(0)

def preproc(df):
  instruction = [instruction1, instruction2,instruction3,instruction4,instruction5,instruction6,instruction7,instruction8,instruction9,instruction10]
  numeric_df = df.select_dtypes(include=[np.number])
  num_cols = numeric_df.columns
  for i in num_cols:
    isdate = False
    intype = 0
    sugdate = str(df[i].iloc[0])
    if sugdate.isdigit() == True:
      if len(str(df[i].iloc[0])) == 4:
        if (isday(sugdate[0:2]) == True and ismonth(sugdate[2:4]) == True):
          intype = 1
          isdate = True
        if (ismonth(sugdate[0:2]) == True and isday(sugdate[2:4]) == True):
          intype = 2
          isdate = True
        if (ismonth(sugdate[0:2]) == True and isyear(sugdate[2:4]) == True):
          intype = 3
          isdate = True
        if (isyear(sugdate[0:2]) == True and ismonth(sugdate[2:4]) == True):
          intype = 4
          isdate = True
      if len(str(df[i].iloc[0])) == 6:
        if (isday(sugdate[0:2]) == True and ismonth(sugdate[2:4]) == True and isyear(sugdate[4:6]) == True):
          intype = 5
          isdate = True
        if (isyear(sugdate[0:2]) == True and ismonth(sugdate[2:4]) == True and isday(sugdate[4:6]) == True):
          intype = 6
          isdate = True
        if (isfullyear(sugdate[0:4]) == True and ismonth(sugdate[4:6]) == True):
          intype = 7
          isdate = True
        if (ismonth(sugdate[0:2]) == True and isfullyear(sugdate[2:6]) == True):
          intype = 8
          isdate = True
      if len(str(df[i].iloc[0])) == 8:
        if (isday(sugdate[0:2]) == True and ismonth(sugdate[2:4]) == True and isfullyear(sugdate[4:8]) == True):
          intype = 9
          isdate = True
        if (isfullyear(sugdate[0:4]) == True and ismonth(sugdate[4:6]) == True and isday(sugdate[6:8]) == True):
          intype = 10
          isdate = True

  if isdate == True:
    for j in range(len(df)): a = instruction[intype]
    df[i].iloc[j] = func(a)

  for i in df.columns:
    if df[i].dtype == 'datetime64[ns]':
      for j in range(len(df)):
        a[2] = df[i][j].day
        a[1] = df[i][j].month
        a[0] = df[i][j].year
        start_date = datetime(a[0], a[1], a[2])
        delta = current_date - start_date
        df[i].iloc[j] = delta.days
      df[i] = pd.to_numeric(df[i], downcast='float')

  object_cols = df.select_dtypes('object').columns
  for i in object_cols:
    intype = 0
    isdate = False
    m = np.array(re.findall(r'\b\d+\b', df[i][0]), dtype = int)
    if len(m) == 1:
      if len(str(m[0])) == 8:
        if (isday(str(m[0])[0:2]) == True and ismonth(str(m[0])[2:4]) == True and isfullyear(str(m[0])[4:8]) == True):
          intype = 95
          isdate = True
        if (isfullyear(str(m[0])[0:4]) == True and ismonth(str(m[0])[4:6]) == True and isday(str(m[0])[6:8]) == True):
          intype = 105
          isdate = True
    if len(m) == 2:
      if not find_month_in_string2(df[i][0]) == 0:
        if isday(m[0]) == True and isfullyear(m[1]) == True:
          intype = 21
          isdate = True
        if isday(m[1]) == True and isfullyear(m[0]) == True:
          intype = 22
          isdate = True
    if len(m) == 3:
      if isday(m[0]) == True and ismonth(m[1]) == True and (isfullyear(m[2]) == True):
        intype = 9
        isdate = True
      if isday(m[0]) == True and ismonth(m[1]) == True and isyear(m[2]):
        intype = 92
      if ((isfullyear(m[0]) == True or isyear(m[0])) and ismonth(m[1]) == True and isday(m[2]) == True):
        intype = 10
        isdate = True
      if ((isyear(m[0]) == True or isyear(m[0])) and ismonth(m[1]) == True and isday(m[2]) == True):
        intype = 102
        isdate = True

  if isdate == True:
    current_date = datetime.now()
    match intype:
      case 21:
        for j in range(len(df)):
          m = np.array(re.findall(r'\b\d+\b', df[i][j]), dtype = int)
          a[2] = m[0]
          a[1] = find_month_in_string2(df[i][j])
          a[0] = m[1]
          df[i].iloc[j] = func(a)
      case 22:
        for j in range(len(df)):
          m = np.array(re.findall(r'\b\d+\b', df[i][j]), dtype = int)
          a[2] = m[1]
          a[1] = find_month_in_string2(df[i][j])
          a[0] = m[0]
          df[i].iloc[j] = func(a)
      case 9:
        for j in range(len(df)):
          m = np.array(re.findall(r'\b\d+\b', df[i][j]), dtype = int)
          a[2] = m[0]
          a[1] = m[1]
          a[0] = m[2]
          df[i].iloc[j] = func(a)
      case 92:
        for j in range(len(df)):
          m = np.array(re.findall(r'\b\d+\b', df[i][j]), dtype = int)
          a[2] = m[0]
          a[1] = m[1]
          a[0] = 2000 + m[2]
          df[i].iloc[j] = func(a)
      case 95:
        for j in range(len(df)):
          m = np.array(re.findall(r'\b\d+\b', df[i][j]), dtype = int)
          a[2] = int(str(m[0])[0:2])
          a[1] = int(str(m[0])[2:4])
          a[0] = int(str(m[0])[4:8])
          df[i].iloc[j] = func(a)
      case 102:
        for j in range(len(df)):
          m = np.array(re.findall(r'\b\d+\b', df[i][j]), dtype = int)
          a[2] = m[2]
          a[1] = m[1]
          a[0] = 2000 + m[0]
          df[i].iloc[j] = func(a)
      case 105:
        for j in range(len(df)):
          m = np.array(re.findall(r'\b\d+\b', df[i][j]), dtype = int)
          a[0] = int(str(m[0])[0:4])
          a[1] = int(str(m[0])[4:6])
          a[2] = int(str(m[0])[6:8])
          df[i].iloc[j] = func(a)
    df[i] = pd.to_numeric(df[i], downcast='float')

  object_cols = df.select_dtypes('object').columns
  bn = BinaryEncoder()
  for col in object_cols:
    nans = df[col].isna()
    t = df[col].copy()
    for i in range(len(df[col])):
      if (not nans[i]):
        m = np.array(re.findall(r'\b\d+\b', t[i]), dtype = float).mean()
        t.loc[i] = m
    if(t.isna().value_counts().loc[True] < len(t)/2):
        df[col] = pd.to_numeric(t, downcast='float')
    if len(df[col].unique()) > len(df) / 2:
      df = df.drop(columns = col)
    else:
      df = pd.concat([df,bn.fit_transform(df[col])],axis = 1)
      df = df.drop(columns = col)

  for i in df.columns:
    Y=pd.DataFrame(df[i].values.reshape(-1,1))
    Y = Y.dropna()
    clf = KNN()
    clf.fit(Y)
    an1 = clf.labels_
    clf=LOF()
    clf.fit(Y)
    an2 = clf.labels_
    an3 = an1 + an2
    id = [Y.index[x] for x in range(len(an3)) if an3[x] == 2]
    Y=pd.DataFrame(df[i].values.reshape(-1,1))
    Y = Y.drop(id)
    mean = Y.mean()
    df[i] = df[i].fillna(mean.iloc[0])
    for x in id:
      df[i][x] = mean.iloc[0]

  df = preprocessing.StandardScaler().fit_transform(df)

  pca = PCA()
  pca.fit(df)
  explained_variance_ratio = pca.explained_variance_ratio_
  s = 0
  a = 0
  j = 0
  for i in range(1,len(explained_variance_ratio),1):
    a = explained_variance_ratio[i-1] - explained_variance_ratio[i]
    if a > s:
      s = a
      j = i
  pca.n_components = j + 1
  df = pd.DataFrame(pca.fit_transform(df))
  return(df)