---
layout: post
title: Çift Yönlü Bağlı Liste - (Doubly Circular Linked List)
categories:
  - projects
tags:
  - C
  - Doubly Circular Linked List
  - Data Structures
image: /assets/img/projects/doubly-circular-linked-list.png
description: |
  C programlama dili ile, çift yönlü dairesel bağlı liste yapısı kullanılarak örnek uygulama
slug: doubly-circular-linked-list
last_modified_at: 22.10.2023
keywords:
  - C
  - Doubly Circular Linked List
  - Data Structures
  - Çift Yönlü Dairesel Bağlı Liste
  - Veri Yapıları
---

## Çift Yönlü Bağlı Liste (Doubly Circular Linked List)

Bu örnekte **çift yönlü dairesel bağlı liste** yapısı kullanışmıştır.

sunucu, yük ve max. kapasite bilgileri içeren struct yapısı, çift yönlü dairesel bağlı listede tutularak aşağıdaki üç fonksiyon yazılmıştır.
* **`üretici`**: Fonksiyon ile yük üretilmektedir ve yükü en az sunucudan başlamak üzere üretilen yükü sunuculara dağıtılmaktadır. Her yük dağılımından sonra sunucular boş kapasite miktarına göre yeniden sıralanmaktadır. (veriler değil, adresleri değişmektedir)
* **`tüketici`**: Belirtilen yük yükü en fazla olan sunucudan başlanmak üzere silinmektedir. Bir sunucudaki yük yetersiz geldiğinde yükü en fazla olan sıradaki sunucudan devam edilmektedir.
* **`listele`**: Sunucuların id, max kapasite ve yük bilgileri bağlı listede bulunduğu sırayla listelenmektedir. 

# Code :

~~~c
// file: "main.c"
#include <stdio.h>
#include <stdlib.h>
#include <conio.h> //getch fonksiyonunu kullanabilmek için dahil ettim

//Renk kodları tanımlamaları
#define KIRMIZI "\e[0;31m"
#define CYN "\e[0;36m"
#define YESIL "\e[0;32m"
#define RENK_SIFIRLA "\e[0m"

// Sunucu bilgilerini tutan struct yapısı
struct sunucu {
    int id;
    float yuk;
    float max_kapasite;
    struct sunucu *prev;
    struct sunucu *next;
};

// Çift yönlü dairesel bağlı listeyi oluşturan ilk ve son pointer'lar
struct sunucu *bas = NULL;
struct sunucu *son = NULL;

// konsol menüsünde seçim yaptıktan sonra, programın akışının devam edebilmesi için kullanıcının klavyede herhangi bir tuşa bastıktan sonra, klavyede bastığı tuşu kaydetmek için bir değişken.
char ch; 

// Listenin boş olup olmadığını kontrol eden yardımcı fonksiyon
int bos_mu(struct sunucu *bas, struct sunucu *son) {
    if (bas == NULL && son == NULL) { // Eğer liste boş ise
        return 1; // 1 değerini döndürür
    }
    else { // Eğer liste boş değil ise
        return 0; // 0 değerini döndürür
    }
}

// Çift yönlü dairesel bağlı listeye yeni bir sunucu ekleyen fonksiyon
void ekle(struct sunucu **bas, struct sunucu **son, int id, float yuk, float max_kapasite) {
    struct sunucu *yeni = (struct sunucu *)malloc(sizeof(struct sunucu)); // Yeni bir sunucu için bellekten yer ayırır

    // Eğer bellekten yer ayırma başarısız olursa fonksiyondan çıkar
    if (yeni == NULL) { 
        printf(KIRMIZI"Bellek hatasi.\n"RENK_SIFIRLA);
        printf("\nDevam etmek icin bir tusa basin.\n");
        ch = getch();
        return;
    }

    // Yeni sunucunun verilerini parametre olarak alınan değerlere atar
    yeni->id = id;
    yeni->yuk = yuk;
    yeni->max_kapasite = max_kapasite;

    if (bos_mu(*bas, *son)) { // Eğer liste boş ise
        yeni->prev = yeni; // Yeni sunucunun prev pointer'ı kendisini işaret eder
        yeni->next = yeni; // Yeni sunucunun next pointer'ı kendisini işaret eder
        *bas = yeni; // Liste başını yeni sunucuya atar
        *son = yeni; // Liste sonunu yeni sunucuya atar
    }
    else { // Eğer liste boş değil ise
        yeni->prev = *son; // Yeni sunucunun prev pointer'ı liste sonunu işaret eder
        yeni->next = *bas; // Yeni sunucunun next pointer'ı liste başını işaret eder
        (*son)->next = yeni; // Liste sonunun next pointer'ı yeni sunucuyu işaret eder
        (*bas)->prev = yeni; // Liste başının prev pointer'ı yeni sunucuyu işaret eder
        *son = yeni; // Liste sonunu yeni sunucuya atar
    }
}

// Kullanıcıdan sunucu bilgilerini alıp listeye ekleyen fonksiyon
void sunucu_ekle(struct sunucu **bas, struct sunucu **son) {
    int n; // Toplam sunucu sayısını tutan değişken. Bu değişken sayesinde kullanıcıdan alınacak olan 'eklenecek sunucu sayısına göre' bir döngü oluştararak sunucu ekleme işlemini gerçekleştireceğim
    int id;
    float yuk;
    float max_kapasite;
    int hata; // Hata kontrolü için kullanılan değişken. Bu sayede kullanıcıdan alınacak olan sunucu id'sinin daha önce başka bir sunucuya atanmış olup olmadığını kontrol edeceğim
    struct sunucu *gecici;

    printf("Toplam kac tane sunucu eklenecek? : ");
    scanf("%d", &n);

    // Kullanıcıyı doğru değer (pozitif) girmeye zorluyorum.
    while (n <= 0) {
        printf(KIRMIZI"Sunucu sayisi pozitif olmali.\n"RENK_SIFIRLA);
        printf("Toplam kac tane sunucu eklenecek? : ");
        scanf("%d", &n);
    }

    // Kullanıcıdan alınan sunucu sayısı kadar dön ve her bir sunucunun bilgilerini kullanıcıdan al
    for (int i = 0; i < n; i++) {
        printf("%d. sunucunun id numarasini giriniz: ", i + 1);
        scanf("%d", &id);

        while (id <= 0) {
            printf(KIRMIZI"id numarasi pozitif olmali.\n"RENK_SIFIRLA);
            printf("%d. sunucunun id numarasini giriniz: ", i + 1);
            scanf("%d", &id);
        }

        hata = 0;
        gecici = *bas;
        if (!bos_mu(*bas, *son)) { // Eğer liste boş değilse
            do {
                if (gecici->id == id) { // Listenin üzerinde gezerek kullanıcının girdiği id değerini listedeki id değerleri ile karşılaştırıyorum
                    hata = 1;
                    break;
                }

                gecici = gecici->next; // Eğer kullanıcının girdiği id değeri, gecici pointerinin gösterdiği id değeri ile aynı değilse, sonraki sonraki pointera geç
            } while (gecici != *bas); // Tekrar listenin başına gelene kadar, döngü içinde id değerinin aynı olup olmadığını kontrol et.
            // Burada do while döngüsü kullanmamın sebebi, bağlı listenin "İLK" değerinin id bilgisini, kullanıcının girdiği id bilgisi ile karşılaştırabilmek ve gecici değişkeninde bir sonraki pointera ilerleyebilmek.
            // Yani döngünün koşulu sağlanmasa bile döngü içindeki kodların en az bir defa çalıştırılmasına ihtiyacım var.
            // Eğer do while yerine sadece while kullansaydım, bağlı listenin ilk değerini bile kontrol edemeden döngüye giremeyecektim.
            // Ama do while kullandığım için bağlı listenin ilk elemanına hiç bir şarta bağlı olmadan ulaşabildim.
        }

        while (hata == 1) { // Eğer hata değişkeni 1 ise (id değeri başka bir sunucu için kullanılıyorsa) kullanıcıyı uygun id değerini girene kadar zorluyorum
            printf(KIRMIZI"Bu id numarasi zaten kullanilmis.\n"RENK_SIFIRLA);
            printf("%d. sunucunun id numarasini giriniz: ", i + 1);
            scanf("%d", &id);

            while (id <= 0) {
                printf(KIRMIZI"id numarasi pozitif olmali.\n"RENK_SIFIRLA);
                printf("%d. sunucunun id numarasini giriniz: ", i + 1);
                scanf("%d", &id);
            }

            hata = 0;
            gecici = *bas;

            if (!bos_mu(*bas, *son)) {
                do {
                    if (gecici->id == id) { // Eğer geçici pointer'ın gösterdiği sunucunun id'si kullanıcının girdiği id ile aynıysa döngüden çık ve kullanıcıdan tekrar id değeri iste
                        hata = 1;
                        break;
                    }

                    gecici = gecici->next;
                } while (gecici != *bas);
            }
        }

        printf("%d. sunucunun maksimum kapasitesini giriniz: ", i + 1);
        scanf("%f", &max_kapasite);

        while (max_kapasite <= 0) {
            printf(KIRMIZI"Maksimum kapasite pozitif olmali.\n"RENK_SIFIRLA);
            printf("%d. sunucunun maksimum kapasitesini giriniz: ", i + 1);
            scanf("%f", &max_kapasite);
        }

        yuk = 0;
        ekle(bas, son, id, yuk, max_kapasite); // Uygun değerleri kullanıcıdan aldıktan sonra listenin sonuna sunucuyu ekliyorum.
        printf(YESIL"%d. sunucu basariyla eklendi.\n"RENK_SIFIRLA, i + 1);
    }
    printf("\nDevam etmek icin bir tusa basin.\n");
    ch = getch();
}

void sirala(struct sunucu **bas, struct sunucu **son) {
    if (bos_mu(*bas, *son)) { // Eğer liste boş ise sıralama işlemi yapmadan fonksiyondan çık
        printf(KIRMIZI"Liste bos.\n"RENK_SIFIRLA);
        printf("\nDevam etmek icin bir tusa basin.\n");
        ch = getch();
        return;
    }

    struct sunucu *gecici1 = *bas;
    struct sunucu *gecici2;
    struct sunucu *temp;

    do {
        gecici2 = gecici1->next;
        while (gecici2 != *bas) {
            if (gecici1->max_kapasite - gecici1->yuk > gecici2->max_kapasite - gecici2->yuk) { // Eğer geçici1'in boş kapasite değeri geçici2'nin boş kapasite değerinden küçükse
            // Burada boş kapasiteye göre sıralama yapmamamın sebebi UZEM deki ödev tanımında yüklere göre, sınıfta anlatılana göre ise maksimum kapasiteye göre olduğu için bende böyle bir yaklaşım tercih ettim
                // geçici1 ve geçici2'nin adreslerini değiştir
                temp = gecici1->prev;
                gecici1->prev = gecici2->prev;
                gecici2->prev = temp;

                temp = gecici1->next;
                gecici1->next = gecici2->next;
                gecici2->next = temp;

                if (*bas == gecici1)
                    *bas = gecici2;
                else if (*bas == gecici2)
                    *bas = gecici1;

                if (*son == gecici1)
                    *son = gecici2;
                else if (*son == gecici2)
                    *son = gecici1;

                temp = gecici1;
                gecici1 = gecici2;
                gecici2 = temp;
            }
            gecici2 = gecici2->next; // Adresler değiştikten sonra gecici2 yi ilerlet
        }
        gecici1 = gecici1->next; // Ve geçici biri ilerlet. Bu sayede gecici1 ve gecici2 hep yan yana
    } while (gecici1 != *son);
}

// Kullanıcıdan yük miktarı alıp sunuculara dağıtan fonksiyon
void uretici(struct sunucu **bas, struct sunucu **son) {
    if (bos_mu(*bas, *son)) { // Eğer liste boş ise fonksiyondan çık
        printf(KIRMIZI"Liste bos.\n"RENK_SIFIRLA);
        printf("\nDevam etmek icin bir tusa basin.\n");
        ch = getch();
        return;
    }

    float yuk;
    sirala(bas, son); // ilk başta çift yönlü dairesel bağlı listenin sıralanması gerekiyor.
    printf("Uretilecek yuk miktarini giriniz: ");
    scanf("%f", &yuk);

    while (yuk < 0) {
        printf(KIRMIZI"Yuk miktari negatif olamaz.\n"RENK_SIFIRLA);
        printf("Uretilecek yuk miktarini giriniz: ");
        scanf("%f", &yuk);
    }

    struct sunucu *gecici = *bas;

    do {
        if (gecici->max_kapasite - gecici->yuk >= yuk) { // Eğer geçici sunucunun boş kapasitesi yeterliyse
            gecici->yuk += yuk; // Yuk miktarını hepsini sunucuya ekleyip
            yuk = 0; // Yuk miktarını sıfırlıyorum
        }
        else { // Eğer geçici sunucunun boş kapasitesi yetersizse
            yuk -= gecici->max_kapasite - gecici->yuk; // Yuk miktarından, sunucunun alabileceği maksimum yük miktarını çıkarıp
            gecici->yuk = gecici->max_kapasite; // sunucunun yükünü maksimum seviyeye çıkarıyorum.
        }

        sirala(bas, son); // Bu yük dağıtma işleminden sonra listeyi tekrar sıralıyorum
        gecici = gecici->next;
    } while (yuk > 0 && gecici != (*son)->next); // Yuk miktarı bitene kadar veya gecici pointerı tekrar en başa geldiği zaman döngü bitecek. 
    // Gecici pointerinin listenin başına tekrar gelip gelmemesini kontrol etmemin sebebi, eğer üretilecek yük miktarı, bütün sunucuların boş kapasitelerinin toplamından büyük ise, sunucuların hepsi dolduktan sonra artık yük ekleyemeyecektir ve yük miktarı hep sıfırdan büyük olacağı için sonsuz döngüye girecekti

    if (yuk > 0) { // Eğer hala dağıtılacak yük varsa
        printf("Tum sunucular dolu. "KIRMIZI"Kalan yuk: %.2f\n"RENK_SIFIRLA, yuk);
        printf("\nDevam etmek icin bir tusa basin.\n");
        ch = getch();
    }
}

void tuketici(struct sunucu **bas, struct sunucu **son) {
    if (bos_mu(*bas, *son)) { // Eğer liste boş ise
        printf(KIRMIZI"Liste bos.\n"RENK_SIFIRLA); // Hata mesajı verir
        printf("\nDevam etmek icin bir tusa basin.\n");
        ch = getch();
        return; // Fonksiyondan çıkar
    }

    float yuk;
    sirala(bas, son); // ilk başta çift yönlü dairesel bağlı listenin sıralanması gerekiyor.
    printf("Tuketilecek yuk miktarini giriniz: ");
    scanf("%f", &yuk);

    while (yuk < 0) {
        printf(KIRMIZI"Yuk miktari negatif olamaz.\n"RENK_SIFIRLA);
        printf("Tuketilecek yuk miktarini giriniz: ");
        scanf("%f", &yuk);
    }

    struct sunucu *gecici = *son; // yükleri silmeye listenin en sonundan başladım. çünkü yükü en fazla olan sunucu hep en sonda olacak.

    do {
        if (gecici->yuk >= yuk) { // Eğer geçici sunucunun yükü yeterliyse
            gecici->yuk -= yuk; // Yuk miktarını sunucudan çıkarır
            yuk = 0; // Yuk miktarını sıfırlar
        }
        else { // Eğer geçici sunucunun yükü yetersizse
            yuk -= gecici->yuk; // Yuk miktarından, sunucunun mevcut yük miktarını çıkarır
            gecici->yuk = 0; // Sunucunun yükünü sıfırlar
        }

        sirala(bas, son); // Ödev tanımında, ilgili yükler silindikten sonra yeniden sıralanması istenmemiş. Ama yinede sıralanması gerekiyor. Çünkü yükü en fazla olandan az olana doğru silebilmem için, listenin azalan sıralı bir halde olması gerekiyor.
        gecici = gecici->prev; // Geçici pointer'ını listenin sonuna getirir
    } while (yuk > 0 && gecici != *son); // Yuk miktarı bitene kadar devam eder veya gecici pointerı listenin son elemanının pointerına eşit olana kadar döngü devam edecek
    // Bu fonksiyon yük tüketimini listenin en sonundan başlayarak yaptığı için tekrar en sona gelip gelmediğini kontrol ediyorum.
    // Bu döngüde gecici pointerinin tekrar listenin sonuna gelip gelmediğini kontrol etmemin sebebi, eğer tüketilecek yük miktarı sunuculardaki toplam yük miktarından fazla olsaydı yük hep sıfırdan büyük olacağı için döngü sonsuz döngüye girecekti. Bu sayede ikinci bir şart ekleyerek bu sorunu kontrol altına almış oldum
    
    if (yuk > 0) { // Eğer hala silinecek yük varsa
        printf("Tum sunucular bos. "KIRMIZI"Kalan yuk: %.2f\n"RENK_SIFIRLA, yuk); // Hata mesajı verir
        printf("\nDevam etmek icin bir tusa basin.\n");
        ch = getch();
    }
}

// Listenin içeriğini ekrana yazdıran fonksiyon
void listele(struct sunucu *bas, struct sunucu *son) {
    if (bos_mu(bas, son)) { // Eğer liste boş ise
        printf(KIRMIZI"Liste bos.\n"RENK_SIFIRLA);
    }
    else { // Eğer liste boş değil ise
        struct sunucu *gecici = bas; // Liste başını gösteren geçiçi bir pointer
        printf("Listedeki sunucular:\n");

        do {
            printf("id: %d\t"CYN"yuk: %.2f\t"RENK_SIFIRLA""YESIL"max_kapasite: %.2f\n"RENK_SIFIRLA, gecici->id, gecici->yuk, gecici->max_kapasite);
            gecici = gecici->next; // Geçici pointer'ı bir sonraki sunucuya kaydır
        } while (gecici != bas); // Geçici pointer tekrar liste başına gelene kadar döngü devam eder
    }

    printf("\nDevam etmek icin bir tusa basin.\n");
    ch = getch();
}

int main() {
    int secim;

    while (1) {
        system("cls");
        printf(CYN"\n[1] Sunucu Listele\n");
        printf("[2] Sunucu Ekle\n");
        printf("[3] Uretici\n");
        printf("[4] Tuketici\n"RENK_SIFIRLA);
        printf(KIRMIZI"[5] Cikis Yap\n"RENK_SIFIRLA);

        printf("\nLutfen bir secenek girin: ");
        scanf("%d", &secim);

        switch (secim) {
        case 1:
            listele(bas, son);
            break;
        case 2:
            sunucu_ekle(&bas, &son);
            break;
        case 3:
            uretici(&bas, &son);
            break;
        case 4:
            tuketici(&bas,&son);
            break;
        case 5:
            return 0;
        default:
            printf("Geçersiz seçenek...");
            break;
        }
    }
}
~~~