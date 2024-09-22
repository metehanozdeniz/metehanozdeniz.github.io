---
layout: post
title: Naive Bayes
categories: [machine-learning]
tags: [classification]
image: /assets/img/machine-learning/naive_bayes_classifier.png
description: |
  Naive Bayes algoritması, olasılık teorisine dayalı, basit ve hızlı bir makine öğrenme algoritmasıdır. Özellikle sınıflandırma problemlerinde sıkça kullanılır. Algoritmanın temel prensibi, Bayes Teoremi'ne dayanarak sınıf olasılıklarını tahmin etmektir.
slug: naive-bayes
last_modified_at: 24.09.2024
keywords:
  - Artificial Intelligence
  - Machine Learning
  - Data Science
  - Data Analysis
  - Naive Bayes
  - Yapay Zeka
  - Makine Öğrenmesi
  - Veri Bilimi
  - Veri Analizi
---
Naive Bayes, bağımsızlık varsayımı (naive assumption) yapar; yani bir veri kümesindeki özelliklerin birbirinden bağımsız olduğunu kabul eder. Bu, gerçek dünyada nadiren doğru olsa da, birçok problemde etkili sonuçlar verir.

Naive Bayes, adını iki temel varsayımdan alır:

## 1. Bayes Teoremi

Naive Bayes algoritması Bayes Teoremi'ni temel alır. Bayes Teoremi, bir olayın olasılığını, o olaya ait gözlemlere ve diğer ilgili olayların olasılıklarına dayanarak hesaplamamıza olanak tanır. Formülü şu şekildedir:

$$P(C|X) = \frac{P(X|C) * P(C)}{P(X)}$$

* $$P(C\|X)$$ : $$X$$ olayının gerçekleştiği bilindiğinde $$C$$ olayının gerçekleşme olasılığıdır. Yani $$X$$ verisinin sınıfı $$C$$ olma olasılığıdır.
* $$P(X\|C)$$ : $$C$$ olayının gerçekleştiği bilindiğinde $$X$$ olayının gerçekleşme olasılığıdır. Yani sınıf $$C$$ verildiğinde $$X$$ verisinin gözlenme olasılığıdır.
* $$P(C)$$ : Sınıf $$C$$ olasılığı.
* $$P(X)$$ : $$X$$ verisinin gözlenme olasılığı.

## 2. Naive Varsayım (Bağımsızlık Varsayımı):

Naive Bayes'in en önemli varsayımı, tüm özelliklerin (feature) birbirinden bağımsız olduğudur. Yani bir özelliğin değeri, diğer özelliklerin değeriyle bağımsızdır. Bu varsayım genellikle gerçek dünyada nadiren geçerlidir, ancak yine de çoğu durumda algoritmanın iyi performans göstermesine engel olmaz.

# Naive Bayes Algoritması Türleri

Naive Bayes algoritmasının birkaç çeşidi vardır ve farklı veri dağılımlarına göre uygulanır:

## Gaussian Naive Bayes
Özellikler sürekli olduğunda ve normal dağılıma uygun olduğunda kullanılır.

## Multinomial Naive Bayes
Belirli sayıda kategoriye sahip veriler için kullanılır. Özellikle metin sınıflandırma problemlerinde yaygın olarak kullanılır. Kelime sayımı gibi verilerle çalışır.

## Bernoulli Naive Bayes
İkili (binary) veriler için uygundur.

# Avantajları
* **Hızlı ve Verimli:** Hesaplama açısından çok hızlıdır ve büyük veri setleriyle bile iyi ölçeklenir.
* **Az Veriyle İyi Sonuçlar:** Özellikle veri seti küçükken dahi iyi sonuçlar verebilir.
* **Kolay Uygulama:** Scikit-learn gibi kütüphaneler sayesinde kolayca uygulanabilir.
* Yüksek boyutlu verilerle iyi başa çıkar.

# Dezavantajları
* **Bağımsızlık Varsayımı:** Özelliklerin birbirinden bağımsız olması varsayımı gerçek dünyada genellikle geçerli olmadığından bazen yanıltıcı olabilir.
* **Sürekli Veriler İçin Zayıflık:** Gaussian Naive Bayes dışında, genellikle sürekli verilerle çalışmada zayıf performans gösterebilir.

# Sample App

Şimdi Naive Bayes algoritması ile sınıflandırma yaparak diabet hastalığını tahmin eden bir model geliştirelim.

# Dataset
* [diabetes.csv](/assets/datasets/diabetes.csv)

## About Dataset

Bu veriseti 768 gözlem ve 8 sayısal bağımsız değişkenden oluşmaktadır.

Bağımlu ve hedef değişken OUTCOME (sonuç)'dur.
* 1 diabet testi sonucunun pozitif,
* 0 ise diabet testi sonucunun negtif olduğunu gösterir.

* **Pregnancies:** Hastanın kaç kere hamile kaldığını gösterir.
* **Glucose:** 2 saatlik oral glukoz tolerans testinde plazma glukoz konsantrasyonu.
* **BloodPressure:** Diastolic kan basıncı (mm Hg)
* **SkinThickness:** Triceps cilt kıvrımı kalınlığı (mm).
* **Insulin:** 2 saatlik serum insülin düzeyi (mu U/ml).
* **BMI:** Vücut kitle indeksi (ağırlık kg / boy m²).
* **DiabetesPedigreeFunction:** Aile geçmişine dayalı olarak diyabet olasılığını hesaplayan bir fonksiyon.
* **Age:** Hastanın yaşı.
* **Outcome:** Diyabet durumu (0 = diyabet yok, 1 = diyabet var).

# Import the necessary modules and libraries

~~~python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Models
from sklearn.naive_bayes import GaussianNB

#Performance metrices
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

#For ignoring warnings
import warnings
warnings.filterwarnings("ignore")
~~~

# Load the Data

~~~python
data = pd.read_csv('diabetes.csv')

print(data.head())
~~~
~~~
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
0            6      148             72             35        0  33.6   
1            1       85             66             29        0  26.6   
2            8      183             64              0        0  23.3   
3            1       89             66             23       94  28.1   
4            0      137             40             35      168  43.1   

   DiabetesPedigreeFunction  Age  Outcome  
0                     0.627   50        1  
1                     0.351   31        0  
2                     0.672   32        1  
3                     0.167   21        0  
4                     2.288   33        1  
~~~

# checking the number of rows and columns

~~~python
print(data.shape)
~~~
~~~
(768, 9)
~~~

~~~python
# getting the some information about the data

print(data.info())
~~~
~~~
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Pregnancies               768 non-null    int64  
 1   Glucose                   768 non-null    int64  
 2   BloodPressure             768 non-null    int64  
 3   SkinThickness             768 non-null    int64  
 4   Insulin                   768 non-null    int64  
 5   BMI                       768 non-null    float64
 6   DiabetesPedigreeFunction  768 non-null    float64
 7   Age                       768 non-null    int64  
 8   Outcome                   768 non-null    int64  
dtypes: float64(2), int64(7)
memory usage: 54.1 KB
None
~~~

~~~python
# checking for the missing values
print(data.isnull().sum())
~~~
~~~
Pregnancies                 0
Glucose                     0
BloodPressure               0
SkinThickness               0
Insulin                     0
BMI                         0
DiabetesPedigreeFunction    0
Age                         0
Outcome                     0
dtype: int64
~~~

~~~python
# Check number of unique values in each column
for i in data.columns:
    print(i, len(data[i].unique()))
~~~
~~~
Pregnancies 17
Glucose 136
BloodPressure 47
SkinThickness 51
Insulin 186
BMI 248
DiabetesPedigreeFunction 517
Age 52
Outcome 2
~~~

~~~python
# Descriptive statistics
print(data.describe())
~~~
~~~
       Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin  \
count   768.000000  768.000000     768.000000     768.000000  768.000000   
mean      3.845052  120.894531      69.105469      20.536458   79.799479   
std       3.369578   31.972618      19.355807      15.952218  115.244002   
min       0.000000    0.000000       0.000000       0.000000    0.000000   
25%       1.000000   99.000000      62.000000       0.000000    0.000000   
50%       3.000000  117.000000      72.000000      23.000000   30.500000   
75%       6.000000  140.250000      80.000000      32.000000  127.250000   
max      17.000000  199.000000     122.000000      99.000000  846.000000   

              BMI  DiabetesPedigreeFunction         Age     Outcome  
count  768.000000                768.000000  768.000000  768.000000  
mean    31.992578                  0.471876   33.240885    0.348958  
std      7.884160                  0.331329   11.760232    0.476951  
min      0.000000                  0.078000   21.000000    0.000000  
25%     27.300000                  0.243750   24.000000    0.000000  
50%     32.000000                  0.372500   29.000000    0.000000  
75%     36.600000                  0.626250   41.000000    1.000000  
max     67.100000                  2.420000   81.000000    1.000000  
~~~

~~~python
# Check the distribution of the target variable
print(data['Outcome'].value_counts())
~~~
~~~
Outcome
0    500
1    268
Name: count, dtype: int64
~~~

# Graphical Analysis of Data

~~~python
# correlation matrix
corr = data.corr()

plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True, cmap='BuPu')
plt.show()
~~~
![correlation matrix for diabetes dataset](/assets/img/machine-learning/correlation_matrix_for_diabetes_dataset.png)

Korelasyon ısı haritasına göre;
* Glukoz ve Outcome (diyabet durumu) arasında pozitif bir korelasyon bulunuyor, yani yüksek glukoz seviyeleri diyabetle daha çok ilişkili.
* BMI ve Outcome arasında da bir ilişki mevcut, ancak daha zayıf.
* Pregnancies ve outcome arasındada zayıf bir ilişki var ancak bu sütunu çıkartıyorum.

~~~python
data.drop('Pregnancies', axis=1, inplace=True)
~~~

~~~python
plt.figure(figsize=(8,8))
plt.scatter(data['Glucose'],data['Outcome'])
plt.xlabel('Glucose')
plt.ylabel('Diabetes')
plt.title('Glucose vs Diabetes')
plt.grid()
~~~
![glucos vs diabetes](/assets/img/machine-learning/glucose_vs_diabetes_scatter_plot.png)

~~~python
plt.figure(figsize=(8,8))
plt.scatter(data['Insulin'],data['Outcome'])
plt.xlabel('Insulin')
plt.ylabel('Diabetes')
plt.title('Insulin vs Diabetes')
plt.grid()
~~~
![insulin vs diabetes](/assets/img/machine-learning/insulin_vs_diabetes_scatter_plot.png)

## Count of Glucose

~~~python
plt.figure(figsize=(40,20),dpi=90)
ax=sns.countplot(x='Glucose',data=data)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Glucose',fontsize=20)
plt.ylabel('Diabetes',fontsize=20)
plt.title('count of Glucose',fontsize=30)
plt.grid()
~~~
![count of glucose](/assets/img/machine-learning/count_of_glucose_countplot.png)

## Count of BloodPressure

~~~python
plt.figure(figsize=(40,20),dpi=90)
ax=sns.countplot(x='BloodPressure',data=data)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('BloodPressure',fontsize=20)
plt.ylabel('Diabetes',fontsize=20)
plt.title('count of BloodPressure',fontsize=30)
plt.grid()
~~~
![count of BloodPressure](/assets/img/machine-learning/count_of_BloodPressure_countplot.png)

## Count of Age

~~~python
plt.figure(figsize=(40,20),dpi=90)
ax=sns.countplot(x='Age',data=data)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Age',fontsize=20)
plt.ylabel('Diabetes',fontsize=20)
plt.title('count of Age',fontsize=30)
plt.grid()
~~~
![count of age](/assets/img/machine-learning/count_of_age_countplot.png)

## Count of SkinThickness

~~~python
plt.figure(figsize=(40,20),dpi=90)
ax=sns.countplot(x='SkinThickness',data=data)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('SkinThickness',fontsize=20)
plt.ylabel('Diabetes',fontsize=20)
plt.title('count of SkinThickness',fontsize=30)
plt.grid()
~~~
![count of SkinThickness](/assets/img/machine-learning/count_of_SkinThickness.png)

Grafikdende anlaşılacağı üzere cilt kalınlığı 0 olan aykırı değerleri çıkartıyorum.

~~~python
# drop the outliers
data=data[data['SkinThickness']>0]
data=data[data['SkinThickness']<90]
~~~

~~~python
plt.figure(figsize=(40,20),dpi=90)
ax=sns.countplot(x='SkinThickness',data=data)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('SkinThickness',fontsize=20)
plt.ylabel('Diabetes',fontsize=20)
plt.title('count of SkinThickness',fontsize=30)
plt.grid()
~~~
![count of SkinThickness](/assets/img/machine-learning/count_of_SkinThickness_2.png)

~~~python
data.dropna(how='any', inplace=True) # drop the missing values

data.reset_index(drop=True, inplace=True) # reset the index
~~~

~~~python
# last shape of the data
print(data.shape)
~~~
~~~
(540, 8)
~~~

## pairplot

~~~python
sns.pairplot(data=data, diag_kind='kde', hue='Outcome',palette='copper')
plt.show()
~~~
![pairplot of diabetes dataset](/assets/img/machine-learning/pairplot_of_diabetes_dataset.png)

# Normalization

~~~python
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
~~~

# Separating the independent and dependent variables

~~~python
x = data_scaled.drop('Outcome', axis=1)
y = data_scaled['Outcome']
~~~

# Splitting the data

~~~python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=0)
~~~

# Create a Gaussian Classifier

~~~python
model = GaussianNB()
model.fit(x_train, y_train)
~~~
<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GaussianNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">GaussianNB</label><div class="sk-toggleable__content"><pre>GaussianNB()</pre></div></div></div></div></div>

# Prediction

~~~python
# Predict the response for test dataset
y_pred = model.predict(x_test)
~~~

# Model Accuracy

~~~python
print("Accuracy:", accuracy_score(y_test, y_pred))
~~~
~~~
Accuracy: 0.7671957671957672
~~~

# confusion matrix

~~~python
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
~~~
![confusion matrix](/assets/img/machine-learning/confusion_matrix_of_naive_bayes_model.png)

# classification report

~~~python
print(classification_report(y_test, y_pred))
~~~
~~~
              precision    recall  f1-score   support

         0.0       0.84      0.81      0.83       128
         1.0       0.63      0.67      0.65        61

    accuracy                           0.77       189
   macro avg       0.73      0.74      0.74       189
weighted avg       0.77      0.77      0.77       189
~~~

# Example patient

~~~python
patient = [[0.6, 0.8, 0.56, 0.4, 0.5, 0.6, 0.7]]

# Making prediction
pred = model.predict(patient)

if pred[0] == 0:
    print('The patient is not diabetic')
else:
    print('The patient is diabetic')
~~~
~~~
The patient is diabetic
~~~

0.77 accuracy değerine sahip kabul edilebilir ölçüde bir model oluşturduk.

Son olarak örnek hastanın değerleri verildğinde, modelimiz bu hastanın diabet hastası olduğunu öngördü.