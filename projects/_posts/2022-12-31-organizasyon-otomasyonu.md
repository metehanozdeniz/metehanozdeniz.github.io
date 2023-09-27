---
layout: post
title: C ile Organizasyon Otomasyonu
categories:
  - projects
tags:
  - C
image: /assets/img/projects/dynamic-memory-allocation.jpg
description: |
  C Programlama dili ile struct veri tipi ve dinamik bellek tahsis fonksiyonları kullanarak bir otomasyon projesi yaptım.
slug: organizasyon-otomasyonu
last_modified_at: 01.08.2023
keywords:
  - C
  - Organizasyon Otomasyonu
  - Dinamik Bellek Tahsis
  - dynamic memory allocation in c
  - organization automation
---

Bir organizasyon şirketi yurt dışında geziye giden kafileye organizasyon hizmeti sağlamaktadır. Müşterilerin farklı seçim alternatifleri bulunmaktadır.

*   Müşteriler 75 TL'ye kahvaltı, 100 TL'ye akşam yemeği veya hiç ödemeyerek yemek hizmeti almayabilirler.
*   Otel hizmet almak zorunlu değildir. Fakat otel hizmeti almak isteyenler için iki otel seçeneği mevcuttur. Müşteriler 2 yıldızlı otelde 200 TL'ye ve 3 yıldızlı otelde 300 TL'ye konaklayabilirler.
*   Müşterilerin yurt dışına çıkışı için iki farklı vasıta seçeneği sunulmuş olup, otobüs ile gidenler 150 TL'ye, uçak ile gidenler ise 250 TL'ye gidebilirler.

Bu verilerin saklanması için iç içe struct veri tipi kullanıldı. _ musteri_bilgileri _ adında bir struct tanımlandı. Struct yapısında müşteri numarası, müşterinin adı, müşterinin soyadı, kahvaltı hizmeti, akşam yemeği hizmeti, otel hizmeti ve otelde konaklayacaksa kaç yıldızlı otelde kalıp kalmayacağı tanımlandı. Ayrıca bu struct yapısının içinde tekrar _ tasit _ isminde bir struct tanımlanarak müşterinin yurt dışına hangi vasıta ile gideceği bilgisi tutuldu. Bu iç içe struct yapısının boş görünmemesi için başlangıçta dinamik bellek fonksiyonları kullanılarak _ ornek_veriler _ isminde bir fonksiyon ile örnek verilerle dolduruldu.

Bu iç içe struct yapısı için _ ekleme, silme, güncelleme ve listeleme _ işlemleri dinamik bellek tahsis fonksiyonları kullanılarak ayrı ayrı fonksiyonlarda yapıldı.

Müşterilerin aldığı hizmetler için ödemesi gereken tutarın hesaplama işlemi, _ fatura_hesapla _ isimli bir fonksiyonda yapıldı.

Müşterilerin hangi vasıta ile yurt dışına gideceğinin listelemesini yapan _ vasita_listele _ isimli fonksiyon oluşturuldu.

Bu yapılan işlemler için aktif olarak kullanılabilir bir menü hazırlandı. Menünün daha anlaşılabilir görünmesi için ilgili menü seçenekleri renklendirildi.

Proje GCC derleyici ile hazırlanmış olup çalışması için sadece GCC derleyicisine gereksinim duymaktadır. Kullanılan GCC versiyonu 6.3.0 dır.
{:.note}

[Download](https://github.com/metehanozdeniz/organizasyon-otomasyonu/archive/refs/heads/main.zip)   [GitHub](https://github.com/metehanozdeniz/organizasyon-otomasyonu)

# Code :

~~~c
// file: "main.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//Renk kodları tanımlamaları
#define KIRMIZI "\e[0;31m"
#define CYN "\e[0;36m"
#define MAGENTA "\e[0;35m"
#define KALIN_MAVI "\e[1;34m"
#define YESIL "\e[0;32m"
#define RENK_SIFIRLA "\e[0m"

struct musteri_bilgileri
{
    int musteri_numarasi; // Her müsteriye ait identity
    char isim[20];
    char soyisim[20];
    int kahvalti_varmi;    // 0=hayir 1=evet
    int aksamyemegi_varmi; // 0=hayir 1=evet
    int hotel;             // 1=otel yok 2=2 yildizli 3=3 yildizli
    struct tasit //Müşterinin yurt dışına hangi vasıta ile gideceğini tutan struct yapısı
    {
        int ucak; //ucak ile gidiyorsa 1 gitmiyorsa 0
        int otobus; //otosbus ile gidiyorsa 1 gitmiyorsa 0
    }vasita;
};

struct musteri_bilgileri *ptr_musteri_listesi;
int toplam_musteri_sayisi = 0; //Dinamik bellek fonksiyonlarında bellekte ayrılan boyutu kontrol edebilmek için bu değişkeni kullandım

#pragma region "Fonksiyonlar"
void ornek_veriler()
{
    // Dinamik bellek tahsisi(Örnek veriler için)
    ptr_musteri_listesi = (struct musteri_bilgileri *)malloc(4 * sizeof(struct musteri_bilgileri));
    if (ptr_musteri_listesi == NULL)
    {
        printf("Bellek tahsisi basarisiz.\n");
        exit(0);
    }
    else
    {
        // Müşteri bilgilerinin doldurulması (Örnek Veriler)
        (ptr_musteri_listesi)->musteri_numarasi = 1;
        strcpy((ptr_musteri_listesi)->isim, "Metehan");
        strcpy((ptr_musteri_listesi)->soyisim, "Ozdenz");
        (ptr_musteri_listesi)->kahvalti_varmi = 0;
        (ptr_musteri_listesi)->aksamyemegi_varmi = 1;
        (ptr_musteri_listesi)->hotel = 2;
        (ptr_musteri_listesi)->vasita.otobus = 0;
        (ptr_musteri_listesi)->vasita.ucak = 1;

        (ptr_musteri_listesi + 1)->musteri_numarasi = 2;
        strcpy((ptr_musteri_listesi + 1)->isim, "Mucahit");
        strcpy((ptr_musteri_listesi + 1)->soyisim, "Duman");
        (ptr_musteri_listesi + 1)->kahvalti_varmi = 1;
        (ptr_musteri_listesi + 1)->aksamyemegi_varmi = 1;
        (ptr_musteri_listesi + 1)->hotel = 2;
        (ptr_musteri_listesi + 1)->vasita.otobus = 1;
        (ptr_musteri_listesi + 1)->vasita.ucak = 0;

        (ptr_musteri_listesi + 2)->musteri_numarasi = 3;
        strcpy((ptr_musteri_listesi + 2)->isim, "Serkan");
        strcpy((ptr_musteri_listesi + 2)->soyisim, "Altnok");
        (ptr_musteri_listesi + 2)->kahvalti_varmi = 1;
        (ptr_musteri_listesi + 2)->aksamyemegi_varmi = 1;
        (ptr_musteri_listesi + 2)->hotel = 3;
        (ptr_musteri_listesi + 2)->vasita.otobus = 0;
        (ptr_musteri_listesi + 2)->vasita.ucak = 1;

        (ptr_musteri_listesi + 3)->musteri_numarasi = 4;
        strcpy((ptr_musteri_listesi + 3)->isim, "Ridvan");
        strcpy((ptr_musteri_listesi + 3)->soyisim, "Kunt");
        (ptr_musteri_listesi + 3)->kahvalti_varmi = 1;
        (ptr_musteri_listesi + 3)->aksamyemegi_varmi = 1;
        (ptr_musteri_listesi + 3)->hotel = 1;
        (ptr_musteri_listesi + 3)->aksamyemegi_varmi = 1;
        (ptr_musteri_listesi + 3)->hotel = 2;
        (ptr_musteri_listesi + 3)->vasita.otobus = 1;
        (ptr_musteri_listesi + 3)->vasita.ucak = 0;

        toplam_musteri_sayisi = 4;
    }
}

void musteriler(struct musteri_bilgileri *bilgiler, int musteri_sayisi) //Müşteri listeleme fonksiyonu
{
    printf("Musteri Listesi :\n");
    for (int i = 0; i < musteri_sayisi; i++)
    {
        printf("Musteri No : %d\t Ad : %s\t Soyad : %s\t Otel: %d\t Kahvalti: %d\t Aksam Yemegi: %d\t Otobus: %d\t Ucak : %d\n", (bilgiler + i)->musteri_numarasi, (bilgiler + i)->isim, (bilgiler + i)->soyisim, (bilgiler + i)->hotel, (bilgiler + i)->kahvalti_varmi, (bilgiler+i)->aksamyemegi_varmi, (bilgiler + i)->vasita.otobus, (bilgiler + i)->vasita.ucak);
    }
}

void musteri_ekle(struct musteri_bilgileri *mus, int musteri_sayisi)
{
    // Yeniden bellek tahsisi
    struct musteri_bilgileri *ptr;
    ptr = mus;
    ptr = (struct musteri_bilgileri *)realloc(mus, (musteri_sayisi + 1) * sizeof(struct musteri_bilgileri));
    if (ptr == NULL)
    {
        printf("Bellek tahsisi basarisiz.\n");
        exit(0);
    }

    printf("Musteri numarasini girin :");
    scanf("%d", &(ptr + musteri_sayisi)->musteri_numarasi);
    printf("Musterinin ismini girin :");
    scanf("%s", &(ptr + musteri_sayisi)->isim);
    printf("Musterinin soyismini girin : ");
    scanf("%s", &(ptr + musteri_sayisi)->soyisim);
    printf("Musteri otelde kaliyor mu?( kalmiyorsa 1, 2 yildizli ise 2, 3 yildizli ise 3 girin) : ");
    scanf("%d", &(ptr + musteri_sayisi)->hotel);
    printf("Musteri kahvalti aldi mi? (evet ->1, hayir ->0) : ");
    scanf("%d", &(ptr + musteri_sayisi)->kahvalti_varmi);
    printf("Musteri aksam yemegi aldi mi? (evet ->1, hayir ->0) : ");
    scanf("%d", &(ptr + musteri_sayisi)->aksamyemegi_varmi);
    printf("Musteri ucakla mi gidecek? (Ucakla gidcekse 1 gitemeyecekse 0) :");
    scanf("%d", &(ptr + musteri_sayisi)->vasita.ucak);
    printf("Musteri otobusle mi gidecek? (Otobusle gidcekse 1 gitemeyecekse 0) :");
    scanf("%d", &(ptr + musteri_sayisi)->vasita.otobus);

    toplam_musteri_sayisi++;
    musteriler(ptr, toplam_musteri_sayisi);
    printf("Musteri eklendi....\n");
    ptr_musteri_listesi = ptr;
}

void musteri_sil(struct musteri_bilgileri *musteri, int musteri_sayisi)
{
    int musteri_numarasi, konum = -1;

    printf("Silinecek musterinin numarasini girin: ");
    scanf("%d", &musteri_numarasi);

    // Müşterinin konumunun bulunması
    for (int i = 0; i < musteri_sayisi; i++)
    {
        if ((musteri + i)->musteri_numarasi == musteri_numarasi)
        {
            konum = i;
            break;
        }
    }

    if (konum == -1)
    {
        printf("Musteri bulunamadi.\n");
    }
    else
    {
        printf(KIRMIZI"Silinecek musteri: %d, %s, %s\n"RENK_SIFIRLA, (musteri + konum)->musteri_numarasi, (musteri + konum)->isim, (musteri + konum)->soyisim);
        // Müşteri silinir. Yani silinecek kişiden sonraki müşteriler, bellekte bir öncekinin yerine yazılır.
        for (int i = konum; i < musteri_sayisi; i++)
        {
            *(musteri + i) = *(musteri + i + 1);
        }
        // Dinamik bellek boyutunun azaltılması
        struct musteri_bilgileri *ptrmusteri;
        ptrmusteri = musteri;
        ptrmusteri = (struct musteri_bilgileri *)realloc(musteri, (musteri_sayisi - 1) * sizeof(struct musteri_bilgileri));
        toplam_musteri_sayisi--;
        ptr_musteri_listesi = ptrmusteri;
        musteriler(ptr_musteri_listesi,toplam_musteri_sayisi);
        printf(KIRMIZI"Musteri silindi....\n"RENK_SIFIRLA);
    }
}

void musteri_guncelle(struct musteri_bilgileri *musteri, int musteri_sayisi)
{
    int musteri_numarasi, konum = -1;
    printf("Guncellemek istediginiz musterinin numarasini girin: ");
    scanf("%d", &musteri_numarasi);

    // Müşterinin konumunun bulunması
    for (int i = 0; i < musteri_sayisi; i++)
    {
        if ((musteri + i)->musteri_numarasi == musteri_numarasi)
        {
            konum = i;
            break;
        }
    }

    if (konum == -1)
    {
        printf(KIRMIZI"Musteri bulunamadi.\n"RENK_SIFIRLA);
    }
    else
    {
        printf("Guncellenecek musteri: %d, %s, %s, Otel : %d, Kahvalti : %d, Aksam Yemegi : %d\n", (musteri + konum)->musteri_numarasi, (musteri + konum)->isim, (musteri + konum)->soyisim, (musteri+konum)->hotel,(musteri+konum)->kahvalti_varmi, (musteri+konum)->aksamyemegi_varmi);
        printf("Yeni isim : ");
        scanf("%s", &(musteri + konum)->isim);
        printf("Yeni soyisim : ");
        scanf("%s", &(musteri + konum)->soyisim);
        printf("Musteri otelde kalacak mi? (kalmiyorsa 1, 2 yildizli ise 2, 3 yildizli ise 3 girin) :");
        scanf("%d", &(musteri + konum)->hotel);
        printf("Musteri kahvalti aldi mi? (evet ->1, hayir ->0) : ");
        scanf("%d", &(musteri + konum)->kahvalti_varmi);
        printf("Musteri aksam yemegi aldi mi? (evet ->1, hayir ->0) : ");
        scanf("%d", &(musteri + konum)->aksamyemegi_varmi);
        printf("Musteri otobusle mi gidecek? (evet ->1, hayir ->0) : ");
        scanf("%d", &(musteri + konum)->vasita.otobus);
        printf("Musteri ucakla mi gidecek? (evet ->1, hayir ->0) : ");
        scanf("%d", &(musteri + konum)->vasita.ucak);

        musteriler(ptr_musteri_listesi, toplam_musteri_sayisi);
        printf(YESIL"Musteri basariyla gunclendi...\n"RENK_SIFIRLA);
    }
}

void fatura_hesapla(struct musteri_bilgileri *musteri, int musteri_sayisi)
{
    int musteri_numarasi,toplam_fatura = 0, konum = -1;
    printf("Faturasini hesaplamak istediginiz musterinin numarasini girin: ");
    scanf("%d", &musteri_numarasi);

    // Müşterinin konumunun bulunması
    for (int i = 0; i < musteri_sayisi; i++)
    {
        if ((musteri + i)->musteri_numarasi == musteri_numarasi)
        {
            konum = i;
            break;
        }
    }

    if (konum == -1)
    {
        printf(KIRMIZI"Musteri bulunamadi.\n"RENK_SIFIRLA);
    }
    else
    {
        if ((musteri + konum)->hotel == 2)
        {
            toplam_fatura+=200;
        }
        if ((musteri + konum)->hotel == 3)
        {
            toplam_fatura+=300;
        }
        if ((musteri + konum)->kahvalti_varmi == 1)
        {
            toplam_fatura+=75;
        }
        if ((musteri + konum)->aksamyemegi_varmi == 1)
        {
            toplam_fatura+=100;
        }
        if ((musteri + konum)->vasita.otobus == 1)
        {
            toplam_fatura+=150;
        }
        if ((musteri + konum)->vasita.ucak == 1)
        {
            toplam_fatura +=250;
        }
        
        printf(YESIL"%s %s"RENK_SIFIRLA KIRMIZI" isimli musterinin toplam fatura tutari : %d TL\n"RENK_SIFIRLA,(musteri+konum)->isim, (musteri+konum)->soyisim ,toplam_fatura);
    }
}

void vasita_listele(struct musteri_bilgileri *musteri, int musteri_sayisi)
{
    printf(YESIL"Otobus ile gidenler Listesi :\n"RENK_SIFIRLA);
    for (int i = 0; i < musteri_sayisi; i++)
    {
        if ((musteri+i)->vasita.otobus == 1)
        {
            printf("%s %s\n", (musteri + i)->isim, (musteri + i)->soyisim);
        }
    }
    printf(YESIL"Ucak ile gidenler listesi : \n"RENK_SIFIRLA);
    for (int i = 0; i < musteri_sayisi; i++)
    {
        if ((musteri+i)->vasita.ucak == 1)
        {
            printf("%s %s\n", (musteri + i)->isim, (musteri + i)->soyisim);
        }
    }
}

#pragma endregion

int main(int argc, char const *argv[])
{
    system("cls");
    printf(MAGENTA"             _    _                      _     _ _       _     \n            | |  | |                    | |   | (_)     (_)    \n            | |__| | ___  ___  __ _  ___| | __| |_ _ __  _ ____\n            |  __  |/ _ \\/ __|/ _` |/ _ \\ |/ _` | | '_ \\| |_  /\n            | |  | | (_) \\__ \\ (_| |  __/ | (_| | | | | | |/ / \n            |_|  |_|\\___/|___/\\__, |\\___|_|\\__,_|_|_| |_|_/___|\n                              __ / |                           \n                              |___/   "RENK_SIFIRLA KALIN_MAVI"https://m3t3h4n.github.io\n"RENK_SIFIRLA);
    ornek_veriler(); //Struct yapısının içi boş görünmesin diye örnek veriler ekledim.
    
    int secim;
    while (1)
    {
        printf(CYN"[1] Musteri Listele\n");
        printf("[2] Musteri Ekle\n");
        printf("[3] Musteri Sil\n");
        printf("[4] Musteri Guncelle\n");
        printf("[5] Fatura Hesapla\n");
        printf("[6] Vasita Listele\n"RENK_SIFIRLA);
        printf(KIRMIZI"[7] Cikis Yap\n"RENK_SIFIRLA);
        printf("Lutfen bir secenek girin: ");
        scanf("%d", &secim);

        switch (secim)
        {
            case 1:
                musteriler(ptr_musteri_listesi,toplam_musteri_sayisi);
                break;
            case 2:
                musteri_ekle(ptr_musteri_listesi, toplam_musteri_sayisi);
                break;
            case 3:
                musteri_sil(ptr_musteri_listesi, toplam_musteri_sayisi);
                break;
            case 4:
                musteri_guncelle(ptr_musteri_listesi, toplam_musteri_sayisi);
                break;
            case 5:
                fatura_hesapla(ptr_musteri_listesi, toplam_musteri_sayisi);
                break;
            case 6:
                vasita_listele(ptr_musteri_listesi, toplam_musteri_sayisi);
                break;
            case 7:
                printf(KIRMIZI"Cikis Yapildi...\n"RENK_SIFIRLA);
                exit(0);
            default:
                printf(KIRMIZI"Gecersiz secenek!\n"RENK_SIFIRLA);
                break;
        }
    }
    return 0;
}
~~~