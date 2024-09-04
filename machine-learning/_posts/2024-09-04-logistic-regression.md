---
layout: post
title: Logistic Regression
categories: [machine-learning]
tags: [classification]
image: /assets/img/machine-learning/logistic_regression.gif
description: |
  Logistic regression, basit yapısı ve açıklanabilirliği nedeniyle makine öğrenmesi uygulamalarında sıkça tercih edilen bir algoritmadır. Temel sınıflandırma problemlerinde etkili bir çözümdür ve daha karmaşık modellerin anlaşılmasında temel bir yapı taşı olarak görev yapar.
slug: logistic-regression
last_modified_at: 05.09.2024
keywords:
  - Artificial Intelligence
  - Machine Learning
  - Data Science
  - Data Analysis
  - Logistic Regression
  - Yapay Zeka
  - Makine Öğrenmesi
  - Lojistik Regresyon
  - Veri Bilimi
  - Veri Analizi
---
* 1. Temel Kavramlar
{:toc}
Logistic regression, makine öğrenmesinde kullanılan en temel sınıflandırma algoritmalarından biridir. Lineer regresyon ile benzer temellere dayansa da, logistic regression sınıflandırma problemleri için uygundur. Özellikle iki sınıf (binary) arasındaki ayrımı yapmak için kullanılır, ancak birden fazla sınıf için de genişletilebilir (multinomial logistic regression).

# 1. Temel Kavramlar

Logistic regression, doğrusal bir modeldir, ancak çıkış değeri sürekli bir sayı yerine bir olasılık olarak ifade edilir. Bu, modelin çıkışını 0 ile 1 arasında bir değere dönüştürmek için bir sigmoid fonksiyonu kullanması sayesinde olur.

# 2. Matematiksel Temel

## a. Lineer Model

Logistic regression, bir doğrusal model ile başlar:

$$z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n = \beta_0 + \sum_{i=1}^{n}{\beta_i x_i}$$

Burada 
$$z$$, bağımsız değişkenlerin bir lineer kombinasyonu olarak hesaplanır. Bu, aslında lineer regresyonun hesaplamasından farklı değildir.

## b. Sigmoid Fonksiyonu

Logistic regression'ın temel farkı, yukarıdaki doğrusal modelin sonucunu sigmoid fonksiyonu denilen bir fonksiyondan geçirerek olasılıkları hesaplamasıdır:

$$h(z) = \frac{1}{1 + {e}^{-z}} = \frac{1}{1 + {e}^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n)}}$$

<img src="/assets/img/machine-learning/sigmoid_func.png" alt="Sigmoid Function" style="width: 700px;"/>

Sigmoid fonksiyonu, her zaman 0 ile 1 arasında bir değer döndürür ve bu da, sınıf 1'e ait olma olasılığı olarak yorumlanır. Eğer olasılık 0.5'ten büyükse, model sınıfı 1 olarak tahmin eder; aksi takdirde 0 olarak tahmin eder.

# 3. Çoklu Sınıf Problemleri (Multinomial Logistic Regression)

Eğer problemde iki yerine birden fazla sınıf varsa, logistic regression birden fazla sınıfı ayırmak için genişletilebilir. Bu, "one-vs-all" (birine karşı diğerleri) veya "softmax regression" gibi yöntemlerle yapılabilir.

# 4. Örnek Uygulama

Örneğin, bir kişinin cinsiyetini tahmin etmek için logistic regression kullanılabilir. Girdi olarak kişinin boy, kilo, yaş bilgileri gibi özellikler kullanılır ve model, bu özelliklere dayalı olarak kişinin erkek olup olmadığını (sınıf 1) veya kadın olduğunu (sınıf 0) tahmin eder.

# Dataset

~~~terminal
// file: "data.csv"
ulke,boy,kilo,yas,cinsiyet
tr,180,90,30,e
tr,190,80,25,e
tr,175,90,35,e
tr,177,60,22,k
us,185,105,33,e
us,165,55,27,k
us,155,50,44,k
us,160,58,39,k
us,162,59,41,k
us,167,62,55,k
fr,174,70,47,e
fr,193,90,23,e
fr,187,80,27,e
fr,183,88,28,e
fr,159,40,29,k
fr,164,66,32,k
fr,166,56,42,k
~~~

# Import the necessary modules and libraries

~~~python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
~~~

# Data Preprocessing

## Read Data

~~~python
data = pd.read_csv('data.csv')

print(data.head(10))

print(data.describe())
~~~

>~~~
  ulke  boy  kilo  yas cinsiyet
0   tr  180    90   30        e
1   tr  190    80   25        e
2   tr  175    90   35        e
3   tr  177    60   22        k
4   us  185   105   33        e
5   us  165    55   27        k
6   us  155    50   44        k
7   us  160    58   39        k
8   us  162    59   41        k
9   us  167    62   55        k
              boy        kilo        yas
count   17.000000   17.000000  17.000000
mean   173.058824   70.529412  34.058824
std     11.829363   17.871477   9.270050
min    155.000000   40.000000  22.000000
25%    164.000000   58.000000  27.000000
50%    174.000000   66.000000  32.000000
75%    183.000000   88.000000  41.000000
max    193.000000  105.000000  55.000000
~~~

## Split Data

~~~python
# boy
boy = data.iloc[:, 1:2].values

# kilo
kilo = data.iloc[:, 2:3].values

# yas
yas = data.iloc[:, 3:4].values

# cinsiyet
cinsiyet = data.iloc[:, -1:].values
~~~

## Concatenate Data

Şimdi bu boy, kilo ve yas bağımsız değişkenlerini (x) ve cinsiyet (y) bağımlı değişkenini birleştiriyorum.

~~~python
# Concatenate Data
x = pd.DataFrame(np.concatenate((boy, kilo, yas), axis= 1), columns=['boy', 'kilo', 'yas'])
y = pd.DataFrame(cinsiyet, columns=['cinsiyet'])

print(x.head(5))
print(y.head(5))
~~~
>~~~
   boy  kilo  yas
0  180    90   30
1  190    80   25
2  175    90   35
3  177    60   22
4  185   105   33
  cinsiyet
0        e
1        e
2        e
3        k
4        e
~~~

## Encoding

Eğitim sürecinde kullanılacak cinsiyet (y) bağımlı değişkenini LabelEncoder sınıfı ile sayısal değerlere dönüştürüyorum.

~~~python
le = LabelEncoder()

y = le.fit_transform(y)

print(y)
~~~
>~~~
[0 0 0 1 0 1 1 1 1 1 0 0 0 0 1 1 1]
~~~

## Train - Test Split

~~~python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
~~~

## Scale Data

~~~python
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

print(X_train.shape)
print(y_train.shape)
~~~
>~~~
(11, 3)
(6, 3)
~~~

# Building and training of the model

~~~python
log_reg = LogisticRegression(random_state=0)

log_reg.fit(X_train, y_train)
~~~

<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(random_state=0)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(random_state=0)</pre></div></div></div></div></div>

# Predict

~~~python
y_pred = log_reg.predict(X_test)

for i in range(len(y_pred)):
    print(f"Actual Class: {y_test[i]} - Predicted Class: {y_pred[i]}")
~~~
>~~~
Actual Class: 0 - Predicted Class: 0
Actual Class: 1 - Predicted Class: 1
Actual Class: 1 - Predicted Class: 1
Actual Class: 1 - Predicted Class: 1
Actual Class: 0 - Predicted Class: 0
Actual Class: 0 - Predicted Class: 0
~~~

# Evaluation

~~~python
# Confusion Matrix
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

print(cm)

sbn.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Erkek', 'Kadın'], yticklabels=['Erkek', 'Kadın'])

plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix')
plt.show()
~~~
~~~
[[3 0]
 [0 3]]
~~~
![Confusion Matrix](/assets/img/machine-learning/confusion_matrix.png)

~~~python
# Accuracy
accuracy = (cm[0][0] + cm[1][1]) / len(y_test)
print(f'Accuracy: {accuracy}')
~~~
~~~
Accuracy: 1.0
~~~

# 5. Avantajlar ve Dezavantajlar

## Avantajlar:
* Basit ve uygulanması kolaydır.
* Sınıflandırma problemleri için oldukça etkilidir.
* Modelin çıktısı bir olasılık verdiği için, karar verici sistemlerde kolayca entegre edilebilir.

## Dezavantajlar:
* Non-lineer problemlerde sınırlıdır.
* Veriler arasındaki ilişki doğrusal değilse, performansı düşer.
* Dengesiz veri setlerinde (bir sınıfın diğerinden çok daha fazla olduğu durumlar) performansı düşebilir.