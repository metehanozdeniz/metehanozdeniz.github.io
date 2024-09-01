---
layout: post
title: Random Forest
categories: [machine-learning]
tags: [prediction]
image: /assets/img/machine-learning/random_forest.gif
description: |
  Random Forest, güçlü bir algoritma olup, genellikle yüksek doğruluk oranları sağlar ve birçok uygulamada tercih edilen bir yöntemdir.
slug: random-forest
last_modified_at: 02.09.2024
keywords:
  - Artificial Intelligence
  - Machine Learning
  - Data Science
  - Data Analysis
  - Decision Trees
  - Random Forest
  - Yapay Zeka
  - Makine Öğrenmesi
  - Karar Ağaçları
  - Rassal Ağaçlar
  - Veri Bilimi
  - Veri Analizi
---
* Nasıl Çalışır?
{:toc}
Random Forest, makine öğrenmesinde kullanılan popüler bir ensemble (topluluk) öğrenme yöntemidir. Bu algoritma, birden fazla karar ağacı modelinin bir araya getirilmesiyle oluşturulur ve genellikle sınıflandırma ve regresyon problemlerinde kullanılır.

# Nasıl Çalışır?

## 1. Karar Ağaçları
Random Forest, birden fazla karar ağacı oluşturur. Her bir karar ağacı, eğitim verilerinin rastgele bir alt kümesiyle ve özelliklerin rastgele bir alt kümesiyle eğitilir. Bu süreç, ağaçların birbirinden bağımsız olmasını ve farklı kararlar alabilmesini sağlar.

>Karar ağaçları hakkında bilgi edinmek için [Decision Trees]({{ '/machine-learning/2024-08-31-decision-trees/' | relative_url }}) konusunu okuyunuz.

## 2. Ensemble (Topluluk) Learning Yaklaşımı
Ensemble learning, makine öğrenmesinde birden fazla modelin bir araya getirilerek daha güçlü ve daha doğru bir model oluşturma yöntemidir. Bu yaklaşım, her bir modelin tek başına elde edebileceği performanstan daha iyi sonuçlar elde etmek amacıyla kullanılır.

Her bir karar ağacı, tahminini yapar. Sınıflandırma problemlerinde, her ağaç bir sınıf tahmini yapar ve en fazla oy alan sınıf, modelin nihai tahmini olur. Regresyon problemlerinde ise, ağaçların tahminlerinin ortalaması alınarak nihai sonuç elde edilir.

### Temel Ensemble Learning Yaklaşımları

#### 1. Bagging (Bootstrap Aggregating)

Bagging, eğitim verisinin rastgele alt kümeleri kullanılarak birden fazla model (genellikle aynı türde) eğitilmesi ve bu modellerin sonuçlarının birleştirilmesi işlemidir.

![Ensemble Learning Bagging Techniques](/assets/img/machine-learning/ensemble_learning_bagging.gif)

Bu yöntem, her bir modelin farklı veri alt kümeleri üzerinde eğitilmesini sağlar, bu da modelin genelleme yeteneğini artırır. Örneğin, Random Forest algoritması, karar ağaçlarının bagging yöntemini kullanarak bir araya getirilmesidir.

**Avantajları:**
Bagging, modeli daha kararlı hale getirir ve varyansı azaltır.

#### 2. Boosting

Boosting, zayıf modellerin (genellikle basit modellerin) ardışık olarak eğitilmesi ve her yeni modelin önceki modellerin hatalarını düzeltmeye çalışması prensibine dayanır.

![Ensemble Learning Boosting Techniques](/assets/img/machine-learning/ensemble_learning_boosting.gif)

Her adımda, yanlış sınıflandırılmış örneklere daha fazla ağırlık verilir, böylece sonraki model bu örnekleri daha iyi öğrenmeye çalışır. Örneğin, AdaBoost ve Gradient Boosting, boosting yöntemine dayanan popüler algoritmalardır.

**Avantajları:** Boosting, genellikle çok yüksek doğruluk oranları sağlar ve özellikle zorlayıcı veri setlerinde etkilidir.

#### 3. Stacking

Stacking, farklı türdeki modellerin çıktılarının bir araya getirilip, bu çıktılardan yeni bir model (meta-model) eğitilmesi yöntemidir.

![Ensemble Learning Boosting Techniques](/assets/img/machine-learning/ensemble_learning_stacking.png)

Stacking, çeşitli modellerin güçlü yönlerinden yararlanarak daha iyi performans elde etmeyi amaçlar. Bu yöntem, genellikle farklı türdeki modellerin kombinasyonunu içerir (örneğin, karar ağaçları, lojistik regresyon, SVM gibi).

**Avantajları:** Stacking, model çeşitliliği sayesinde esneklik sağlar ve genellikle daha yüksek bir doğruluk elde edilir.

## 3. Overfitting Azaltma
Karar ağaçları genellikle aşırı uyum (overfitting) yapma eğilimindedir. Ancak, birden fazla karar ağacının bir araya getirilmesi ve her ağacın farklı bir veri alt kümesiyle eğitilmesi, bu sorunu azaltır ve modelin genelleme yeteneğini artırır.

# Avantajları
* **Genelleme Yeteneği:** Birden fazla karar ağacı kullanıldığı için model, veri setindeki gürültüden etkilenmez ve daha iyi genelleme yapar.
* **Overfitting'e Dayanıklılık:** Karar ağaçlarının bir araya getirilmesi, aşırı uyum riskini azaltır.
* **Esneklik:** Hem sınıflandırma hem de regresyon problemlerinde kullanılabilir.

# Dezavantajları
* **Hesaplama Maliyeti:** Birden fazla karar ağacı oluşturulması gerektiğinden, özellikle büyük veri setlerinde eğitim süreci uzun sürebilir.
* **Yorumlanabilirlik:** Birden fazla ağacın bir araya getirilmesi, modelin karmaşıklığını artırır ve kararların yorumlanmasını zorlaştırır.

Şimdi [salaries.csv](https://gist.githubusercontent.com/metehanozdeniz/be330879f29b0a9c49521c6ffc48aa16/raw/18b58e460944f24326df6ff639f035fca1540464/salaries.csv) verisetindeki eğitim seviyesi (X) bağımsız değişkeni ile, maaş (Y) bağımlı değişkenini tahmin eden bir **Random Forest** modeli oluşturalım.

# Dataset
* [salaries.csv](https://gist.githubusercontent.com/metehanozdeniz/be330879f29b0a9c49521c6ffc48aa16/raw/18b58e460944f24326df6ff639f035fca1540464/salaries.csv)

# Import the necessary modules and libraries

~~~python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
~~~

# Data Preprocessing

## Read Data

~~~python
data = pd.read_csv('salaries.csv')

print(data.head(10))
~~~
~~~
              unvan  Egitim Seviyesi   maas
0             Cayci                1   2250
1          Sekreter                2   2500
2  Uzman Yardimcisi                3   3000
3             Uzman                4   4000
4  Proje Yoneticisi                5   5500
5               Sef                6   7500
6             Mudur                7  10000
7          Direktor                8  15000
8           C-level                9  25000
9               CEO               10  50000
~~~

## Graphical Analysis of Data

~~~python
sbn.pairplot(data)
~~~
![seaborn graph](/assets/img/machine-learning/support_vector_regression_seaborn_outout.png)

## Split Data

~~~python
egitim_seviyesi_X = data.iloc[:,1:2].values
maas_Y = data.iloc[:,2:].values

for i in range(len(egitim_seviyesi_X)):
    print("Egitim Seviyesi: ",egitim_seviyesi_X[i],"Maas: ",maas_Y[i])
~~~
~~~
Egitim Seviyesi:  [1] Maas:  [2250]
Egitim Seviyesi:  [2] Maas:  [2500]
Egitim Seviyesi:  [3] Maas:  [3000]
Egitim Seviyesi:  [4] Maas:  [4000]
Egitim Seviyesi:  [5] Maas:  [5500]
Egitim Seviyesi:  [6] Maas:  [7500]
Egitim Seviyesi:  [7] Maas:  [10000]
Egitim Seviyesi:  [8] Maas:  [15000]
Egitim Seviyesi:  [9] Maas:  [25000]
Egitim Seviyesi:  [10] Maas:  [50000]
~~~

# Creating the Model

Şimdi ``RandomForestRegressor`` sınıfını kullanarak bir random forest modeli oluşturalım. Bu sınıfı kullanırken aldığı parametreler modelin nasıl inşa edileceğini belirler.

~~~python
rf_reg = RandomForestRegressor(n_estimators=10,  
                            criterion='squared_error',  
                            max_depth=None,  
                            min_samples_split=2,  
                            min_samples_leaf=1,  
                            min_weight_fraction_leaf=0.0,  
                            max_features=1.0,  
                            max_leaf_nodes=None,  
                            min_impurity_decrease=0.0,  
                            bootstrap=True,  
                            oob_score=False,  
                            n_jobs=None,  
                            random_state=None,  
                            verbose=0,  
                            warm_start=False,  
                            ccp_alpha=0.0,  
                            max_samples=None, )
~~~

## Parameters

Modeli oluştururken kullanılan parametrelerin aldığı değerler default değerlerdir. Bu parametrelerin bağzılarının açıklamaları şu şekildedir:

### 1. ``n_estimators=10`` :
Random Forest modelindeki karar ağaçlarının sayısını belirtir. Bu örnekte, model 10 adet karar ağacından oluşacaktır. Daha fazla ağaç genellikle daha iyi performans sağlar, ancak işlem süresi de artar.

### 2. ``criterion='squared_error'`` :
Her bir karar ağacında dallanma kararlarını verirken kullanılan kriterdir. ``squared_error``, varyansı en aza indirmeye çalışır ve regresyon problemleri için uygun bir kriterdir.

### 3. ``max_depth=None`` :
Karar ağaçlarının maksimum derinliğini belirler. ``None`` değeri, ağaçların maksimum derinliğe kadar büyümesine izin verir, bu da her bir yaprağın saf olana kadar (örneğin, her bir yaprakta tek bir örnek kalana kadar) büyüyeceği anlamına gelir.

### 4. ``min_samples_leaf=1`` :
Her yaprakta (son düğümde) bulunması gereken minimum örnek sayısıdır. 1 değeri, yaprakta en az bir örnek bulunmasını sağlar.

### 5. ``max_leaf_nodes=None`` :
Karar ağaçlarındaki maksimum yaprak düğüm sayısını sınırlar. ``None``, ağaçlarda sınırsız sayıda yaprak düğümü olabileceğini belirtir.

### 6. ``bootstrap=True`` :
Modeli eğitirken örneklerin rastgele seçilip seçilmeyeceğini belirtir. ``True`` olduğunda, her bir ağacın eğitim verileri, bootstrap (yeniden örnekleme) yöntemiyle seçilir.

### 7. ``oob_score=False`` :
Out-of-Bag (OOB) hatasını hesaplamak isteyip istemediğinizi belirler. ``True`` olarak ayarlandığında, OOB verileri üzerinde model performansını değerlendiren bir hata skoru hesaplanır. Bu, çapraz doğrulamaya benzer bir yöntemdir.

### 8. ``n_jobs=None`` :
Modelin eğitimi sırasında kullanılacak işlemci çekirdeği sayısını belirtir. None, varsayılan olarak tek bir çekirdek kullanır. -1 olarak ayarlanırsa, tüm mevcut çekirdekler kullanılır. Eğer büyük bir veri seti üzerinde çalışılıyorsa tüm çekirdekleri kullanmak daha hızlı bir model eğitimi sağlar.

### 9. ``random_state=None`` :
Rastgele sayı üreteci için başlangıç değeri (seed) sağlar. Bu, modelin çıktılarının tekrarlanabilir olmasını sağlar. ``None`` olarak bırakıldığında, rastgelelik kontrol edilmez.

### 10. ``warm_start=False`` :
``True`` olarak ayarlandığında, modelin yeniden eğitilmesi sırasında önceki fit sonuçları kullanılır ve model üzerine ek ağaçlar eklenir. False, her eğitimde modelin sıfırdan başlaması anlamına gelir.

Bu parametreler, ``RandomForestRegressor`` modelinin esnekliğini ve performansını çeşitli şekillerde kontrol etmenizi sağlar. Modeli özelleştirerek veri setinize ve probleminize uygun hale getirebilirsiniz.

# Fit Regression Model

~~~python
rf_reg.fit(egitim_seviyesi_X, maas_Y)
~~~
<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(n_estimators=10)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(n_estimators=10)</pre></div></div></div></div></div>

# Predict

Şimdi oluşturmuş ve eğitmiş olduğumuz modelimizde tahminlerde bulunalım.

~~~python
predict_1 = rf_reg.predict([[11]]) # Eğitim seviyesi 11 olan kişinin maaşının tahmini
predict_2 = rf_reg.predict([[6.6]]) # Eğitim seviyesi 6.6 olan kişinin maaşının tahmini

print("Eğitim Seviyesi 11 olan kişinin tahmini maaşı: ",predict_1)
print("Eğitim Seviyesi 6.6 olan kişinin tahmini maaşı: ",predict_2)
~~~
~~~
Eğitim Seviyesi 11 olan kişinin tahmini maaşı:  [42500.]
Eğitim Seviyesi 6.6 olan kişinin tahmini maaşı:  [10300.]
~~~

Eğer tek bir decision tree kullanılmış olsaydı sonuçlar farklı olacaktı. Çünkü decision tree tek bir ağaç üzerinden tahmin yapar ve bu ağaç üzerindeki verilere göre tahmin yapar. Ama random forest algoritması birden fazla decision tree üzerinden tahmin yapar ve bu ağaçların ortalamasını alarak tahmin yapar. Bu sayede daha doğru tahminler yapabilir.

# Visualize the Random Forest Regression results

~~~python
plt.scatter(egitim_seviyesi_X, maas_Y, color='red', label='Dataset')
plt.scatter(11, predict_1, color='green', label='Eğitim Seviyesi 11 olan kişinin tahmini maaşı')
plt.scatter(6.6, predict_2, color='purple', label='Eğitim Seviyesi 6.6 olan kişinin tahmini maaşı')
plt.plot(egitim_seviyesi_X, rf_reg.predict(egitim_seviyesi_X), color='blue', label='Random Forest Regression Prediction')
plt.title('Random Forest Regression')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaş')
plt.legend()
plt.show()
~~~
![Random Forest Regression Plot](/assets/img/machine-learning/random_forest_regression_plot.png)

# Evaluation of Model

~~~python
print("Random Forest R2 Değeri: ",r2_score(maas_Y, rf_reg.predict(egitim_seviyesi_X)))
~~~
~~~
Random Forest R2 Değeri:  0.969218252652263
~~~

Grafiktende anlaşılacağı üzere 0.969218252652263 -> 0.97 R2 değerine sahip bir başarı oranı elde ettik.