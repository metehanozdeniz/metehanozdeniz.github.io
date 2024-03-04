---
layout: post
title: NumPy Arrays
categories: [software]
tags: [Python, Data Science, Data Analysis, NumPy]
image: /assets/img/software/numpy.webp
description: |
  Bu yazımda NumPy Dizilerini inceliyoruz.
slug: numpy-arrays
last_modified_at: 03.03.2024
keywords:
  - Python
  - Data Science
  - Data Analysis
  - NumPy
  - NumPy Arrays
---

NumPy, tek boyutlu dizilerden başlayarak çok boyutlu dizilere kadar geniş bir yelpazede dizileri destekler.

NumPy dizileri, Python listelerinden daha hızlı ve daha verimli bir şekilde işlenir. Bu diziler, aynı türden (genellikle sayısal) elemanlar içerir ve boyutları sabittir.

NumPy dizilerine giriş yapmadan önce `numpy` kütüphanesini ekleyelim.

~~~python
import numpy as np
~~~

# NumPy Array Oluşturma

![Full-width image](/assets/img/software/numpy_array_module.webp){:.lead width="800" height="150" loading="lazy"}

NumPy da array oluşturmak için `array()` fonksiyonunu kullanırız.

`array()` fonksiyonu içine bir **liste** yada **tuple** alır ve bu liste yada tuple'ı NumPy dizisine dönüştürür.

Tek boyutlu bir NumPy dizisi oluşturalım.

~~~python
arr = np.array([1, 2, 3, 4, 5])
print(arr)
~~~
~~~
[1 2 3 4 5]
~~~

İki boyutlu bir NumPy dizisi oluşturalım.

~~~python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr)
~~~
~~~
[[1 2 3]
 [4 5 6]]
~~~

Üç boyutlu bir NumPy dizisi oluşturalım.

~~~python
matrixListesi = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
arr = np.array(matrixListesi)
print(arr)
~~~
~~~
[[1 2 3]
 [4 5 6]
 [7 8 9]]
~~~

# arange

![Full-width image](/assets/img/software/np.arange.webp){:.lead width="800" height="100" loading="lazy"}

`arange()` fonksiyonu, belirli bir aralıktaki sayıları içeren bir dizi oluşturmak için kullanılır.
Oluşturulan bu diziyi **numpy.ndarray** veri tipinde geri döndürür.

~~~python
arr = np.arange(0,10)
print(arr)
~~~
~~~
[0 1 2 3 4 5 6 7 8 9]
~~~

Eğer istersek bir stepsize değeride verebiliriz.

~~~python
# 0'dan 10'a kadar 2'şer artan bir dizi oluşturur.
arr = np.arange(0,10,2)
print(arr)
~~~
~~~
[0 2 4 6 8]
~~~

## zeros

`zeros()` fonksiyonu, içine aldığı parametre kadar **sıfırlardan** oluşan bir dizi oluşturur. Oluşan bu diziyi **numpy.ndarray** veri tipinde geri döndürür.

~~~python
arr = np.zeros(5)
print(arr)
~~~
~~~
[0. 0. 0. 0. 0.]
~~~

İstersek matrix şeklinde çok boyutlu olarak da oluşturabiliriz.

~~~python
arr = np.zeros((3,3))
print(arr)
~~~
~~~
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]
~~~

## ones

`ones()` fonksiyonu, içine aldığı parametre kadar **birlerden** oluşan bir dizi oluşturur. `zeros()` ile kullanımı aynıdır. Dilersek bunda da çok boyutlu bir matrix şeklinde oluşturabiliriz.

~~~python
arr = np.ones(5)
print(arr)
~~~
~~~
[1. 1. 1. 1. 1.]
~~~

# linspace

![Full-width image](/assets/img/software/numpy_linspace.webp){:.lead width="800" height="150" loading="lazy"}

`linspace()` fonksiyonu başlangıç ve bitiş değerleri arasında eşit aralıklarla belirtilen sayıda eleman içeren bir dizi oluşturur. Oluşan bu diziyi **numpy.ndarray** veri tipinde geri döndürür.

~~~python
#0 ile 10 arasında 20 adet sayı üretir ve bu sayıların arasındaki farklar eşit olur.
arr = np.linspace(0,10, 20)
print(arr)
~~~
~~~
[ 0.          0.52631579  1.05263158  1.57894737  2.10526316  2.63157895
  3.15789474  3.68421053  4.21052632  4.73684211  5.26315789  5.78947368
  6.31578947  6.84210526  7.36842105  7.89473684  8.42105263  8.94736842
  9.47368421 10.        ]
~~~

# eye

`eye()` fonksiyonu ile birim matris oluşturulur ve **numpy.ndarray** veri tipinde geri döndürür.

~~~python
# 3x3'lük birim matris oluşturur.
arr = np.eye(3)
print(arr)
~~~
~~~
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
~~~

# random

![Full-width image](/assets/img/software/numpy_random.webp){:.lead width="800" height="150" loading="lazy"}

~~~python
# Bu modül random fonksiyonunu daha açıklayıcı bir şekilde kullanmak ve görselleştirmek için eklenmiştir.
# matplotlib modülü için ayrı bir bölüm oluşturulacaktır.
import matplotlib.pyplot as plt
~~~

## rand

`random.rand` fonksiyonu 0 ile 1 arasında yarı açık aralıkta **[0.0, 1.0)**  aldığı parametre kadar rastgele sayılar üretir ve bu sayıları **numpy.ndarray** veritipinde geri döndürür.
Bu üretilen rastgele sayılar **düzgün dağılıma** sahip olur.

~~~python
duzgun_dagilim = np.random.rand(100)
print(duzgun_dagilim)

pdf, bins, patches = plt.hist(duzgun_dagilim, bins=20, range=(0, 1), density=True)
plt.title('rand: düzgün dağılım')
plt.show()
~~~
~~~
[0.77579866 0.84455141 0.46788593 0.99109672 0.65992829 0.62707084
 0.47456743 0.5425032  0.44449771 0.73976107 0.4897272  0.78820953
 0.91183784 0.47451638 0.70560843 0.50194818 0.76850847 0.54574648
 ...
 0.57782342 0.30487575 0.8094421  0.25995016 0.03612558 0.92562029
 0.56278298 0.32444382 0.48555378 0.00850049]
~~~

![Full-width image](/assets/img/software/duzgun_dagilim_output.png){:.lead loading="lazy"}
matplotlib output
{:.figcaption}

## randn
![Full-width image](/assets/img/software/numpy_randn.webp){:.lead width="800" height="150" loading="lazy"}

`random.randn` fonksiyonu 0 ile 1 arasında aldığı argüman değeri kadar rastgele sayılar oluşturur.
Oluşan bu rastgele sayılar **standart normal dağılıma** sahip olur.

~~~python
normal_dagilim = np.random.randn(1000)
print(normal_dagilim)

pdf, bins, patches = plt.hist(normal_dagilim, bins=50, range=(-4, 4), density=True)
plt.title('randn: normal dağılım')
plt.show()
~~~
~~~
[ 9.59068260e-01  1.83699230e-01 -1.54856530e+00 -5.84017694e-01
  1.13925103e+00  1.37592729e+00  6.13220180e-01 -6.50900489e-02
  1.45157276e+00  1.11531011e+00 -4.28057906e-01 -8.84610866e-01
 -8.43905081e-01 -8.37270301e-01  1.99786062e+00 -1.16864809e+00
 -3.89113802e-01  5.72274539e-01  1.26108808e-01 -2.22543856e+00
  7.15577403e-01  8.34596730e-02  6.10824774e-01  1.43124166e+00
  ...
  -4.30282619e-01  4.72681203e-01 -1.95738820e+00  5.04030998e-01
  1.59918012e+00  6.62418308e-01 -9.86402745e-01 -1.91581532e+00]
~~~

![Full-width image](/assets/img/software/normal_dagilim_output.png){:.lead loading="lazy"}
matplotlib output
{:.figcaption}

## randint

`randint` fonksiyonu, belirtilen aralıkta rastgele integer veri tipinde bir sayı üretir.

~~~python
arr = np.random.randint(1,100)
print(arr)
~~~
~~~
52
~~~

Eğer randint fonksiyonuna 3. bir argüman girilirse, bu sefer girilen argüman kadar int veri tipinde rastgele sayı üretip, üretilen bu sayıları diziye atar. Bu diziyi ndarray veri tipinde geri döndürür.

~~~python
#1 ile 100 arasında rastgele 20 adet sayı üretir.
arr = np.random.randint(1,100,20)
print(arr)
~~~
~~~
[ 3 31 28 55 45 49 27 24 67 22 97 45 73 59 36 30 59 10 42 55]
~~~

# reshape

![Full-width image](/assets/img/software/numpy_reshape.webp){:.lead width="800" height="150" loading="lazy"}

`reshape` fonksiyonunu daha iyi anlamak için numpy dizisini yeniden oluşturalım.

~~~python
# 0 dan 30 a kadar bir dizi oluşturur.
arr = np.arange(30)
print(arr)
~~~
~~~
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29]
~~~

`reshape` fonksiyonu numpy dizisini yeniden şekillendirir.

~~~python
# 30 adet sayıyı 5x6'lık bir matrise dönüştürür.
matris = arr.reshape(5,6)
print(matris)
~~~
~~~
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]
 [24 25 26 27 28 29]]
~~~

>Burada dikkat edilmesi gereken nokta, reshape fonksiyonuna ilgili parametreleri verirken, dizinin elaman sayısının istenilen formata uygun olması gerektiğidir. `(5x6=30)`

# shape

numpy dizisi ile oluşturulan matrisin boyutlarını verir. **(nxm)**

~~~python
print(matris.shape)
~~~
~~~
(5, 6)
~~~

# max

![Full-width image](/assets/img/software/numpy_max.webp){:.lead width="800" height="150" loading="lazy"}

`max` fonksiyonunu daha iyi anlayabilmek için random bir numpy dizisi oluşturalım.

~~~python
# 0 dan 100 e kadar 20 adet rastgele sayı üretir.
randomArr = np.random.randint(0,100,20)
print(randomArr)
~~~
~~~
[85 35 89 81 46 98  3 61 17  7 23 90 41 33 88 35 34 63 94  4]
~~~

`max` fonksiyonu, dizideki en büyük değeri döndürür.

~~~python
maxValue = randomArr.max()
print(maxValue)
~~~
~~~
98
~~~

## argmax

`argmax` fonksiyonu dizideki en büyük değerin index numarasını geri döndürür.

~~~python
maxValueIndex = randomArr.argmax()
print(maxValueIndex)
~~~
~~~
5
~~~

# min

`min` fonksiyonu, dizideki en küçük değeri döndürür.

~~~python
minValue = randomArr.min()
print(minValue)
~~~
~~~
3
~~~

## argmin

`argmin` fonksiyonu dizideki en küçük değerin index numarasını geri döndürür.

~~~python
minValueIndex = randomArr.argmin()
print(minValueIndex)
~~~
~~~
6
~~~