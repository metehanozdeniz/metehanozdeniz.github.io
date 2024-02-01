---
layout: post
title: Stack (Yığın) Yapısı Kullanarak Infix-Postfix Dönüşümü
categories:
  - projects
tags:
  - C
  - Stack
  - Data Structures
image: /assets/img/projects/infix-to-postfix-using-stack-data-structure.jpg
description: |
  Stack (Yığın) Kullanarak Infix-Postfix Dönüşümü
slug: infix-to-postfix-conversion-using-stack-data-structure
last_modified_at: 29.10.2023
keywords:
  - C
  - Stack Data Structure
  - Data Structures
  - Yığın
  - Veri Yapıları
---

## Stack (Yığın) Yapısı Kullanılarak Infix-Postfix Dönüşümü

Bu örnekte **Stack Yapısı** kullanılarak infix bir ifade postfix ifadeye dönüştürülmüştür. Ardından elde edilen postfix ifade yine stack yapısı kullanılarak ilgili hesaplamalar yapılmıştır. Yapılan her adım ekrana yazdırılmıştır. Ayrıca ilgili infix ifade bir txt dosyasından **(infix.txt)** okunarak alınmıştır.

# Code :

{% gist 9573e272bfde9b9b75fb14f5c9920012 %}

~~~txt
3+4*2/(1-5)^2
~~~
Örnek **infix.txt** dosyası
{:.figcaption}

~~~terminal
// file: "OUTPUT"
Infix: 3+4*2/(1-5)^2
Infix: +4*2/(1-5)^2        | Stack :                     | Postfix: 3
Infix: 4*2/(1-5)^2         | Stack : +                   | Postfix: 3
Infix: *2/(1-5)^2          | Stack : +                   | Postfix: 34
Infix: 2/(1-5)^2           | Stack : *+                  | Postfix: 34
Infix: /(1-5)^2            | Stack : *+                  | Postfix: 342
Infix: (1-5)^2             | Stack : /+                  | Postfix: 342*
Infix: 1-5)^2              | Stack : (/+                 | Postfix: 342*
Infix: -5)^2               | Stack : (/+                 | Postfix: 342*1
Infix: 5)^2                | Stack : -(/+                | Postfix: 342*1
Infix: )^2                 | Stack : -(/+                | Postfix: 342*15
Infix: ^2                  | Stack : /+                  | Postfix: 342*15-
Infix: 2                   | Stack : ^/+                 | Postfix: 342*15-
Infix:                     | Stack : ^/+                 | Postfix: 342*15-2
Infix:                     | Stack:                      | Postfix: 342*15-2^/+
Infix:                     | Stack: 3                    | Postfix: 42*15-2^/+
Infix:                     | Stack: 43                   | Postfix: 2*15-2^/+
Infix:                     | Stack: 243                  | Postfix: *15-2^/+
Infix:                     | Stack: 83                   | Postfix: 15-2^/+
Infix:                     | Stack: 183                  | Postfix: 5-2^/+
Infix:                     | Stack: 5183                 | Postfix: -2^/+
Infix:                     | Stack: -483                 | Postfix: 2^/+
Infix:                     | Stack: 2-483                | Postfix: ^/+
Infix:                     | Stack: 1683                 | Postfix: /+
Infix:                     | Stack: 03                   | Postfix: +
Infix:                     | Stack: 3                    | Postfix:
Sonuc: 3
~~~
**Output**
{:.figcaption}