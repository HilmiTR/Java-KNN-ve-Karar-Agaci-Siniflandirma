# ProLab-2: KNN ve Karar Ağacı Sınıflandırma 

Selamlar Bu repo, Kocaeli Üniversitesi Bilgisayar Mühendisliği "Programlama Laboratuvarı-II" dersi için Asya ile birlikte geliştirdiğimiz makine öğrenmesi projesini içeriyor.

Projeyi kısaca özetlemek gerekirse: Kocaeli'deki bir marketin müşteri verilerini (`MarketSalesKocaeli.xlsx`) kullanarak, tüketicilerin yaş, cinsiyet, harcama potansiyeli gibi profillerine göre hangi ürün kategorisine yöneleceğini tahmin etmeye çalıştık. 

Buradaki en büyük olayımız (ve hocanın en çok önemsediği şey): **Hiçbir hazır makine öğrenmesi kütüphanesi (Scikit-learn, Weka vb.) kullanmadık.** Tüm algoritmaları Java'da sıfırdan, Nesneye Yönelik Programlama (OOP) prensiplerine uygun şekilde kendimiz inşa ettik.

## Neler Yaptık?
* **Veri Ön İşleme (Data Preprocessing):** Excel'den gelen ham verideki boş (null) satırları uçurduk, metin tabanlı verileri sayılara çevirdik (Label Encoding) ve sayısal verilerin hepsini 0-1 arasına ölçekledik (Min-Max Normalization).
* **KNN Algoritması:** k=5 için Öklid mesafesi hesaplayarak en yakın komşuları bulan yapıyı kurduk. Tembel öğrenme (lazy learning) olduğu için test aşamasında biraz yavaş kalıyor tabii :)
* **Karar Ağacı (Decision Tree):** Entropi ve Information Gain (Bilgi Kazancı) hesaplayarak kural tabanlı, özyinelemeli (recursive) bir ağaç yapısı oluşturduk. Ağaç sonsuza gitmesin diye maxDepth=10 olarak kısıtladık.
* **Koyu OOP Mimarisi:** Abstract class'lar, Interface'ler (`IClassifier`), Encapsulation falan havada uçuşuyor. Sistem tamamen genişletilebilir yapıda.
* **Görsel Arayüz:** Sonuçları siyah ekranda (terminalde) kuru kuru bırakmak istemedik. HTML, CSS ve Chart.js kullanarak algoritmaların performansını ve hata matrislerini karşılaştıran şık bir web arayüzü ekledik.

## Nasıl Çalıştırılır?
İşi hiç karmaşıklaştırmadık, derleyip çalıştırmanız yeterli:
1. Projeyi favori Java IDE'nizde (VS Code, IntelliJ vb.) açın.
2. `src/main/java` yolundaki `Main.java` dosyasını çalıştırın.
3. Terminalde doğruluk oranlarını ve milisaniye cinsinden çalışma sürelerini göreceksiniz.
4. Çıkan bu sonuçların grafiksel ve radar tablo hallerini görmek için projenin ana dizinindeki `index.html` dosyasını tarayıcınızda açmanız yeterli.

## Geliştirici Ekip
* **Hilmi Aziz Öztürk** (Veri ön işleme, Karar Ağacı implementasyonu, Arayüz entegrasyonu)
* **Asya Berin Uzuner** (OOP mimarisi, KNN implementasyonu, Evaluator ve Hata Ayıklama)

