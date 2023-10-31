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

~~~c
// file: "main.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h> // ifadenin sayı olup olmadığını kontrol etmek için bu kütüphaneyi dahil ettim
#include <math.h> // Üs alma işlemi için bu kütüphaneyi dahil ettim

// Stack yapısını temsil eden struct veri tipi
struct stack {
    char eleman; // Stack elemanının verisi
    struct stack *next; // Stack elemanının bir sonraki elemanına işaret eden pointer
};

// Stack yapısına yeni bir eleman ekleyen fonksiyon
void push(struct stack **ust, char eleman) {
    // Yeni bir stack elemanı için bellekten yer ayırma
    struct stack *yeni_liste = (struct stack *)malloc(sizeof(struct stack));
    // Yeni elemanın verisini ve bir sonraki elemanını belirleme
    yeni_liste->eleman = eleman;
    yeni_liste->next = *ust;
    // Stack'in en üstündeki elemanı yeni eleman olarak güncelleme
    *ust = yeni_liste;
}

// Stack yapısından en üstteki elemanı silip geri döndüren fonksiyon
char pop(struct stack **ust) {
    // Stack boş ise -1 döndürme
    if (*ust == NULL) {
        return -1;
    }
    // Stack'in en üstündeki elemanın verisini ve bir sonraki elemanını saklama
    char eleman = (*ust)->eleman;
    struct stack *next = (*ust)->next;
    // Stack'in en üstündeki elemanı bellekten silme
    free(*ust);
    // Stack'in en üstündeki elemanı bir sonraki eleman olarak güncelleme
    *ust = next;
    // Silinen elemanın verisini geri döndürme
    return eleman;
}

// Stack yapısının en üstündeki elemanın verisini geri döndüren fonksiyon
char en_ust_eleman(struct stack *ust) {
    // Stack boş ise -1 döndürme
    if (ust == NULL) {
        return -1;
    }
    // Stack'in en üstündeki elemanın verisini geri döndürme
    return ust->eleman;
}

// Stack yapısının boş olup olmadığını kontrol eden fonksiyon
int bos_mu(struct stack *ust) {
    // Stack boş ise 1, değilse 0 döndürme
    return ust == NULL;
}

// Bir karakterin operatör olup olmadığını kontrol eden fonksiyon
int operator_mu(char c) {
    // Karakter +, -, *, / veya ^ ise 1, değilse 0 döndürme
    return c == '+' || c == '-' || c == '*' || c == '/' || c == '^';
}

// İki operatörün önceliklerini karşılaştıran fonksiyon
int oncelik_karsilastir(char op1, char op2) {
    // Operatörlerin öncelik sıralarını belirleyen diziler
    char operatorler[] = "+-*/^";
    int oncelikler[] = {0, 0, 1, 1, 2};
    
    // Operatörlerin dizideki indislerini bulma
    int index1 = -1, index2 = -1;
    for (int i = 0; i < 5; i++) {
        if (op1 == operatorler[i]) {
            index1 = i;
        }
        if (op2 == operatorler[i]) {
            index2 = i;
        }
    }
    
    // Operatörlerin önceliklerini karşılaştırma ve sonucu döndürme
    if (index1 == -1 || index2 == -1) {
        return 0; // Geçersiz operatörler için 0 döndürme
    }
    if (oncelikler[index1] > oncelikler[index2]) {
        return 1; // op1'in önceliği op2'den büyükse 1 döndürme
    }
    if (oncelikler[index1] < oncelikler[index2]) {
        return -1; // op1'in önceliği op2'den küçükse -1 döndürme
    }
    return 0; // op1'in önceliği op2'ye eşitse 0 döndürme
}

// Bir infix ifadeyi postfixe çeviren fonksiyon
char *infixden_postfixe(char *infix) {
    // Postfix ifadeyi saklamak için bellekten yer ayırma
    char *postfix = (char *)malloc(strlen(infix) + 1);
    // Postfix ifadeyi oluşturmak için kullanılacak stack yapısı
    struct stack *stack = NULL;
    // Postfix ifadeyi oluşturmak için kullanılacak indis
    int index = 0;
    int counter_infix = 0;
    // Infix ifadeyi soldan sağa doğru gezme
    for (int i = 0; i < strlen(infix); i++) {
        char c = infix[i];
        
        // Karakter bir sayı veya harf ise postfix ifadeye ekleme
        if (isdigit(c) || isalpha(c)) {
            postfix[index++] = c;
        }
        
        // Karakter bir açma parantezi ise stack'e ekleme
        else if (c == '(') {
            push(&stack, c);
        }
        
        // Karakter bir operatör ise stack'teki öncelikli operatörleri postfix ifadeye ekleme ve son olarak kendisini stack'e ekleme
        else if (operator_mu(c)) {
            while (!bos_mu(stack) && en_ust_eleman(stack) != '(' && oncelik_karsilastir(en_ust_eleman(stack), c) >= 0) {
                postfix[index++] = pop(&stack);
            }
            push(&stack, c);
        }
        
        // Karakter bir kapama parantezi ise stack'teki açma parantezine kadar olan operatörleri postfix ifadeye ekleme ve açma parantezini silme
        else if (c == ')') {
            while (!bos_mu(stack) && en_ust_eleman(stack) != '(') {
                postfix[index++] = pop(&stack);
            }
            pop(&stack); // Açma parantezini silme
        }
        int counter_stack = 0;
        
        // Ekrana adım adım işlem sonuçlarını yazdırma
        printf("\nInfix: ");
        for (int k = 1 + counter_infix; k < strlen(infix); k++)
        {
            printf("%c", infix[k]);
        }
        counter_infix++;

        printf("%*s| Stack : ",20 - (strlen(infix) - counter_infix),"");
        struct stack *temp = stack;
        while (temp != NULL) {
            printf("%c", temp->eleman);
            temp = temp->next;
            counter_stack++;
        }
        printf("%*s| Postfix: ",20 - counter_stack,"");

        // Burada, postfix ifadenin her bir karakterini kontrol ederek ekrana yazdırıyorum. Çünkü en başta postfix için bellekten yer ayırdığım için, postfix ifadeyi doğrudan ekrana yazdırırken postfix ifadenin devamında bellekte kalan verilerde string ifadeye dönüşttürülüp ekrana yazılıyor. 
        // Bu sayede postfix ifade ekrana daha temiz ve okunabilir bir şekilde yazılıyor.
        for (int j = 0; j < strlen(postfix); j++)
        {
            if (isalnum(postfix[j]) || operator_mu(postfix[j]))
            {
                printf("%c", postfix[j]);
            }
        }
    }
    
    // Stack'te kalan operatörleri postfix ifadeye ekleme
    while (!bos_mu(stack)) {
        postfix[index++] = pop(&stack);
    }
    
    // Postfix ifadenin sonuna null karakteri ekleme ve geri döndürme
    postfix[index] = '\0';

    // Ekrana adım adım işlem sonuçlarını yazdırma (son durum)
    printf("\nInfix: %-20s| Stack: ","");
    struct stack *temp = stack;
    while (temp != NULL) {
        printf("%c", temp->eleman);
        temp = temp->next;
    }
    printf("%*s | Postfix: %s\n",20,"", postfix);
    return postfix;
}

// Bir postfix ifadenin sonucunu hesaplayan fonksiyon
int postfix_hesapla(char *postfix) {
    // Hesaplama işlemleri için kullanılacak stack yapısı
    struct stack *stack = NULL;
    int counter_postfix = 0;
    
    // Postfix ifadeyi soldan sağa doğru gezme
    for (int i = 0; i < strlen(postfix); i++) {
        char c = postfix[i];
        
        // Karakter bir sayı ise stack'e ekleme
        if (isdigit(c)) {
            push(&stack, c - '0'); // Karakteri sayıya çevirip ekleme
        }
        
        // Karakter bir operatör ise stack'ten iki sayı çıkarıp işlemi yapma ve sonucu stack'e ekleme
        else if (operator_mu(c)) {
            int x = pop(&stack); // Stack'ten ilk sayıyı çıkarma
            int y = pop(&stack); // Stack'ten ikinci sayıyı çıkarma
            
            // Operatöre göre işlemi yapma ve sonucu stack'e ekleme
            switch (c) {
                case '+':
                    push(&stack, y + x);
                    break;
                case '-':
                    push(&stack, y - x);
                    break;
                case '*':
                    push(&stack, y * x);
                    break;
                case '/':
                    push(&stack, y / x);
                    break;
                case '^':
                    push(&stack, pow(y, x));
                    break;
            }
        }

        // Ekrana adım adım işlem sonuçlarını yazdırma
        int stack_counter = 0;
        printf("\nInfix: %*s| Stack: ",20,"");
        struct stack *temp = stack;
        while (temp != NULL) {
            printf("%d", temp->eleman);
            temp = temp->next;
            stack_counter++;
        }
        printf("%*s | Postfix: ",20 - stack_counter,"");
        for (int j = 1 + counter_postfix; j < strlen(postfix); j++)
        {
            printf("%c", postfix[j]);
        }
        counter_postfix++;
        
    }
    
    // Stack'te kalan tek sayının sonuç olduğunu varsayma ve geri döndürme
    return pop(&stack);
}

int main() {
    FILE *dosya;
    char infix[200]; // Dosyadan okunan infix ifadenin saklanacağı karakter dizisi
    dosya = fopen("infix.txt", "r"); // infix.txt adlı dosyayı okuma modunda açıyoruz

    if (dosya == NULL) {
        printf("Dosya bulunamadi!\n");
        return 1;
    }

    fscanf(dosya, "%s", infix); // Dosyadan verinin okunması ve infix karakter dizisinin içine atılması
    printf("Infix: %s", infix);
    
    // Infix ifadeyi postfixe çevirme
    char *postfix = infixden_postfixe(infix);
    
    // Postfix ifadeyi hesaplama
    int result = postfix_hesapla(postfix);
    printf("\e[0;32m\nSonuc: %d\e[0m", result);
    
     free(postfix);

     fclose(dosya);
    
    return 0;
}
~~~

~~~
3+4*2/(1-5)^2
~~~
Örnek **infix.txt** dosyası
{:.figcaption}