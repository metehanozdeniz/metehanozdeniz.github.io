---
layout: post
title: Bağlı Liste - (Linked List)
categories:
  - projects
tags:
  - C
image: /assets/img/projects/linked-list.png
description: |
  C programlama dili ile, tek yönlü bağlı liste yapısı kullanılarak öğrenci otomasyonu uygulaması
slug: linked-list-example
last_modified_at: 15.10.2023
keywords:
  - C
  - Linked List
  - Data Structures
  - Bağlı Liste
  - Veri Yapıları
---

## Bağlı Liste Yapısı (Linked List)

Bu ***tek yönlü bağlı liste*** örneği aşağıdaki senaryo göz önüne alınarak tasarlanmıştır.

* 1. struct: ***`dersler`***: içerisinde ders adı, ders kodu ve ders notu bulunmaktadır.
* 2. struct ***`ogrenci`***: içerisinde öğrenci no, öğrenci ad, öğrenci soyad bilgileri bulunmaktadır. Ayrıca öğrencinin aldığı dersler bir önceki maddede anlatılan struct yapısı ile bu struct içerisinde bulunmaktadır.
* Başlangıçta ***`ogrenciler.txt`*** dosyasından veri çekilerek struct yapısına uygun şekilde kaydedildi.
* ***`listele()`*** fonksiyonu ile tüm veriler listelendi.
* ***`silme()`*** fonksiyonu parametre olarak öğrenci soyadını alarak, o soyada sahip tüm öğrenciler ile ilgili bilgileri silmedktedir
* Öğrenciler ve aldıkları derslerin ortalama puanı hesaplanarak bağlı liste yapısına küçükten büyüğe sıralı bir şekilde kaydedilmektedir.
* ***`arama()`*** fonksiyonu, ders kodunu parametre olarak alarak o dersi alan öğrencilerin bilgilerini (ders notu dahil) yazdırmaktadır.

# Code :

~~~c
// file: "main.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <conio.h> //getch fonksiyonunu kullanabilmek için bu kütüphaneyi dahil ettim

//Renk kodları tanımlamaları
#define KIRMIZI "\e[0;31m"
#define MAGENTA "\e[0;35m"
#define YESIL "\e[0;32m"
#define RENK_SIFIRLA "\e[0m"

//ogrenci struct tanımı
struct ogrenci {
    char isim[20];
    char soyisim[20];
    int numara;
    struct dersler *ders; //ogrenci struct'ının içinde dersler struct'ından bir pointer tanımladım
    // dersler struct'ını bu ogrenci struct'ının içinde tanımlamam gerektiğinin farkındayım ama bu seferde tek bir öğrenciye ait birden fazla ders bilgisini nasıl tutacağımı bilemedim.
    // 3 farklı ders bilgisinin tutulması istendi ve bende ders1, ders2 ve ders3 şeklinde bir yapı tercih edebilirdim. Ama böyle yaparsam amatörce olabileceğini düşündüm.
    struct ogrenci *sonraki; //ogrenci struct'ının içinde bir sonraki düğümü gösteren bir pointer tanımladım
    float not_ortalamasi;
};

//dersler struct tanımı
struct dersler {
    char ders_adi[20];
    int ders_notu;
    int ders_kodu;
};

//bağlı liste pointer'ı tanımı. Bu pointer, ogrenci struct'ının ilk adresini tutacak.
struct ogrenci *ilk = NULL;
char ch; // konsol menüsünde seçim yaptıktan sonra, programın akışının devam edebilmesi için kullanıcının klavyede herhangi bir tuşa bastıktan sonra, klavyede bastığı tuşu kaydetmek için bir değişken.

void bilgi_al(){
    //dosya işaretçisi tanımı
    FILE *fp;

    //dosyayı açma
    fp = fopen("ogrenciler.txt", "r");

    //dosya açma kontrolü
    if (fp == NULL) {
        printf("Dosya acilamadi.\n");
        exit(1);
    }
    
    //dosyadan okuma döngüsü
    while (!feof(fp)) { //dosyanın sonuna gelene kadar devam et
        
        //ogrenci struct'ından ogr adında bir pointer
        struct ogrenci *ogr;

        //ogr pointer'ına malloc ile bellek ayır
        ogr = (struct ogrenci *)malloc(sizeof(struct ogrenci));

        //bellek ayırma kontrolü
        if (ogr == NULL) {
            printf("Ogrenci bilgileri için bellek tahsisi basarisiz.\n");
            exit(1);
        }

        //dosyadan isim, soyisim ve numara bilgilerini oku ve ogr pointer'ının gösterdiği yere ata
        fscanf(fp, "%s %s %d", ogr->isim, ogr->soyisim, &ogr->numara);

        //ogr pointer'ının gösterdiği yerdeki ders pointer'ına malloc ile bellek ayır
        ogr->ders = (struct dersler *)malloc(3 * sizeof(struct dersler));

        //bellek ayırma kontrolü
        if (ogr->ders == NULL) {
            printf("Ders bilgileri için bellek tahsisi basarisiz.\n");
            exit(1);
        }

        // dosyadan üç adet ders bilgisi oku ve ogr pointer'ının gösterdiği yerdeki ders pointer'ının gösterdiği yere ata
        // ayrıca her dersin not bilgisini okuyarak ortalamayı hesapla ve ogrenci struct'ının içindeki not_ortalamasi pointerinin gösterdiği yere ata
        float not_ortalamasi = 0;
        for (int i = 0; i < 3; i++) {
            fscanf(fp, "%s %d %d", ogr->ders[i].ders_adi, &ogr->ders[i].ders_notu, &ogr->ders[i].ders_kodu);
            not_ortalamasi += ogr->ders[i].ders_notu;
        }

        not_ortalamasi /= 3;
        ogr->not_ortalamasi = not_ortalamasi;

        //ogr pointer'ının sonraki özelliğinin değerini, sonraki düğümü gösterecek şekilde NULL yap
        ogr->sonraki = NULL;

        //liste boş ise
        if (ilk == NULL) {
            //ilk pointer'ını ogr pointer'ına ata
            ilk = ogr;
        }
        else {
            //gecici bir liste pointer'ı tanımla ve ilk düğüme ata
            struct ogrenci *gecici = ilk;

            //gecici pointer'ını son düğüme kadar ilerlet
            while (gecici->sonraki != NULL) {
                gecici = gecici->sonraki;
            }

            //son düğümün sonraki alanına ogr pointer'ını ata
            gecici->sonraki = ogr;
        }
    }

    //dosyayı kapatma
    fclose(fp);
}

void listele(struct ogrenci *ilk) {
    //liste boş ise
    if (ilk == NULL) {
        printf("Txt dosyasi bos.\n");
    }
    else {
        //gecici bir liste pointer'ı tanımla ve ilk düğüme ata
        struct ogrenci *gecici = ilk;

        //gecici NULL olana kadar (bağlı listenin sonuna gelene kadar) döngüyü devam ettir
        while (gecici != NULL) {
            //gecici düğümün verilerini ekrana yazdır
            printf("Isim: %s\tSoyisim: %s\tNumara: %d\t"KIRMIZI"Not Ortalamasi: %f\n"RENK_SIFIRLA, gecici->isim, gecici->soyisim,gecici->numara, gecici->not_ortalamasi);

            //gecici düğümün ders pointer'ını kullanarak ders bilgilerini ekrana yazdır
            for (int i = 0; i < 3; i++) {
                printf("Ders adi: %s\n", gecici->ders[i].ders_adi);
                printf("Ders notu: %d\n", gecici->ders[i].ders_notu);
                printf("Ders kodu: %d\n", gecici->ders[i].ders_kodu);
            }

            printf("---------------------------------------------------\n");

            //gecici pointer'ını bir sonraki düğüme ilerlet
            gecici = gecici->sonraki;
        }
    }
    printf("\nDevam etmek icin bir tusa basin.");
    ch = getch();
    system("cls");
}

void ortalamaya_gore_listele(struct ogrenci *ilk) {
    struct ogrenci *gecici = ilk;
    struct ogrenci *gecici2 = ilk;

    char temp_isim[20], temp_soyisim[20];
    int temp_numara;
    float temp_not_ort;

    char temp_ders_adi[20];
    int temp_ders_kodu;
    int temp_ders_notu;

    while (gecici != NULL) {
        gecici2 = gecici->sonraki;
        while (gecici2 != NULL) {
            if (gecici->not_ortalamasi > gecici2->not_ortalamasi) {
                //isimleri yer değiştir
                strcpy(temp_isim, gecici->isim);
                strcpy(gecici->isim, gecici2->isim);
                strcpy(gecici2->isim, temp_isim);

                //soyisimleri yer değiştir
                strcpy(temp_soyisim, gecici->soyisim);
                strcpy(gecici->soyisim, gecici2->soyisim);
                strcpy(gecici2->soyisim, temp_soyisim);

                //numaraları yer değiştir
                temp_numara = gecici->numara;
                gecici->numara = gecici2->numara;
                gecici2->numara=temp_numara;

                //not ortalamalarını yer değiştir.
                temp_not_ort = gecici->not_ortalamasi;
                gecici->not_ortalamasi = gecici2->not_ortalamasi;
                gecici2->not_ortalamasi = temp_not_ort;

                //ders bilgilerinin yerini değiştir
                for (int i = 0; i < 3; i++)
                {
                    strcpy(temp_ders_adi, gecici->ders[i].ders_adi);
                    strcpy(gecici->ders[i].ders_adi, gecici2->ders[i].ders_adi);
                    strcpy(gecici2->ders[i].ders_adi, temp_ders_adi);
                    
                    temp_ders_kodu = gecici->ders[i].ders_kodu;
                    gecici->ders[i].ders_kodu = gecici2->ders[i].ders_kodu;
                    gecici2->ders[i].ders_kodu = temp_ders_kodu;

                    temp_ders_notu = gecici->ders[i].ders_notu;
                    gecici->ders[i].ders_notu = gecici2->ders[i].ders_notu;
                    gecici2->ders[i].ders_notu = temp_ders_notu;
                }
            }
            gecici2 = gecici2->sonraki;
        }
        gecici = gecici->sonraki;
    }
    printf(YESIL"Ogrencilerin not ortalamasina gore listesi -->\n"RENK_SIFIRLA);
    listele(ilk);
}

void sil(char *soyisim) {
    int bulundu = 0;

    //liste boş ise
    if (ilk == NULL) {
        printf("Txt dosyasi bos.\n");
    }
    else {
        //gecici bir liste pointer'ı tanımla ve ilk düğüme ata
        struct ogrenci *gecici = ilk;

        //ilk düğümdeki öğrencinin soyisimi aranan soyisme eşitse
        while (gecici != NULL && strcmp(ilk->soyisim, soyisim) == 0) {
            ilk = ilk->sonraki;
            free(gecici);
            gecici = ilk;
            bulundu = 1;
        }

        //diğer düğümlerdeki öğrencilerin soyisimi aranan soyisme eşitse
        while (gecici != NULL && gecici->sonraki != NULL) {
            if (strcmp(gecici->sonraki->soyisim, soyisim) == 0) {
                struct ogrenci *temp = gecici->sonraki;
                gecici->sonraki = temp->sonraki;
                free(temp);
                bulundu = 1;
            }
            else {
                gecici = gecici->sonraki;
            }
        }
        if (bulundu == 1) {
            printf("%s soyismine sahip ogrenciler basariyla silindi...\n", soyisim);
        }
        else {
            printf("%s soyismine sahip bir ogrenci bulunamadi!\n");
        }
    }
    printf("Devam etmek icin bir tusa basin.");
    ch = getch();
    system("cls");
}

void ara(int ders_kodu) {
    struct ogrenci *gecici = ilk;
    int bulundu = 0;
    char ders_adi[20];
    while (gecici != NULL) {
        for (int i = 0; i < 3; i++) {
            // parametre olarak gönderilen ders_kodu, gecici listenin ders pointerinin gösterdiği adresteki değere eşitse
            if (gecici->ders[i].ders_kodu == ders_kodu) {
                if (!bulundu) {
                    strcpy(ders_adi, gecici->ders[i].ders_adi);
                    printf("%s dersini alan ogrenciler:\n", ders_adi);
                    bulundu = 1;
                }
                printf("Isim: %s\tSoyisim: %s\tDers Notu: %d\n", gecici->isim, gecici->soyisim, gecici->ders[i].ders_notu);
            }
        }
        gecici = gecici->sonraki;
    }
    if (!bulundu) {
        printf("Bu ders koduna ait bir ders yok.\n");
    }
    printf("\nDevam etmek icin bir tusa basin.");
    ch = getch();
    system("cls");
}

int main() {
    system("cls");
    bilgi_al();
    int secim;
    char soyad[20];
    int derskodu;

    while (1)
    {
        printf(MAGENTA"\n[1] Listele\n");
        printf("[2] Ortalamaya Gore Listele\n");
        printf("[3] Soyismine Gore Sil\n");
        printf("[4] Ders Koduna Gore Ara\n"RENK_SIFIRLA);
        printf(KIRMIZI"[5] Cikis Yap\n"RENK_SIFIRLA);
        printf(YESIL"\nSeciminizi Yaziniz: "RENK_SIFIRLA);
        scanf("%d",&secim);

        switch (secim) {
            case 1:
                listele(ilk);
                break;
            case 2:
                ortalamaya_gore_listele(ilk);
                break;
            case 3:
                printf(YESIL"\nSilmek istediginiz ogrencinin soyisimini girin: "RENK_SIFIRLA);
                scanf("%s", &soyad);
                sil(soyad);
                break;
            case 4:
                printf(YESIL"\nDers kodunu giriniz: "RENK_SIFIRLA);
                scanf("%d", &derskodu);
                ara(derskodu);
                break;
            case 5:
                printf("Cikis yapildi...");
                return 0;
                break;
            default:
                printf("\nGecersiz secenek...");
                break;
        }
    }
}
~~~

~~~
Metehan Ozdeniz 123456 VeriYapilari 50 98 AVP 80 99 Elektronik 60 100
Mucahit Duman 234567 Matematik 80 101 Fizik 75 102 Kimya 60 103
Zeynep Ozturk 345678 LojikDevreler 90 104 Fizik 84 102 Veritabani 60 106
Mehmet Demir 456789 Istatistik 70 107 Matematik 50 101 AVP 65 99
Ayse Celik 567890 YapayZeka 85 115 VeriYapilari 95 98 Kimya 75 103
Hasan Kaya 678901 BilgiGuvenligi 75 113 Veritabani 50 106 YapayZeka 55 115
~~~

**ogrenciler.txt** dosyası
{:.figcaption}