# Laporan Proyek Machine Learning - Muhammad Zaki Alfadilah

## Domain Proyek

Di era digital saat ini, jumlah konten hiburan yang tersedia secara online meningkat secara eksponensial, termasuk film, serial, dan dokumenter. Platform seperti Netflix, Disney+, Amazon Prime, dan lainnya menghadirkan ribuan pilihan bagi pengguna. Namun, kelimpahan pilihan ini justru dapat menimbulkan permasalahan yang dikenal dengan istilah information overload, di mana pengguna kesulitan untuk menemukan konten yang sesuai dengan preferensi mereka secara efisien.

Untuk mengatasi masalah tersebut, sistem rekomendasi (recommender systems) telah menjadi solusi utama. Sistem ini membantu mempersonalisasi pengalaman pengguna dengan menyarankan film berdasarkan perilaku pengguna lain (collaborative filtering) atau berdasarkan karakteristik konten itu sendiri (content-based filtering). Dalam dunia nyata, implementasi sistem rekomendasi telah meningkatkan engagement pengguna dan durasi penggunaan platform secara signifikan [Gómez-Uribe & Hunt, 2016](https://doi.org/10.1145/2843948).

Penerapan machine learning dalam sistem rekomendasi memungkinkan pengambilan keputusan berbasis data yang lebih akurat dan adaptif. Algoritma seperti neural collaborative filtering dan deep content-based models telah terbukti meningkatkan performa rekomendasi dibandingkan metode konvensional.

**Mengapa Masalah Ini Penting untuk Diselesaikan**
- Pengalaman Pengguna yang Lebih Baik
Sistem rekomendasi dapat mengurangi waktu pencarian pengguna dan meningkatkan kepuasan terhadap layanan streaming.

- Meningkatkan Retensi dan Engagement
Rekomendasi yang relevan meningkatkan kemungkinan pengguna tetap menggunakan platform, berdampak pada keberlangsungan bisnis.

- Adaptif terhadap Preferensi Dinamis
Model machine learning dapat belajar dan menyesuaikan terhadap perubahan preferensi pengguna seiring waktu.

**Riset Terkait dan Referensi**

Gómez-Uribe, C. A., & Hunt, N. (2016). The Netflix Recommender System: Algorithms, Business Value, and Innovation. ACM Transactions on Management Information Systems, 6(4), 13. https://doi.org/10.1145/2843948

He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural Collaborative Filtering. Proceedings of the 26th International Conference on World Wide Web (WWW '17), 173–182. https://doi.org/10.1145/3038912.3052569

## Business Understanding

Platform penyedia layanan streaming, misalnya, memiliki kepentingan untuk mempertahankan pengguna, meningkatkan engagement, dan menyediakan pengalaman menonton yang personal dan memuaskan. Oleh karena itu, sistem rekomendasi yang akurat dan efisien menjadi salah satu aspek paling vital dalam mendukung tujuan bisnis tersebut.

**Problem Statements**
- Pernyataan Masalah 1:
Pengguna sering mengalami kesulitan dalam menemukan film yang sesuai dengan preferensi mereka akibat banyaknya pilihan yang tersedia.

- Pernyataan Masalah 2:
Sistem rekomendasi konvensional yang hanya berbasis popularitas atau rating tidak mampu mempersonalisasi rekomendasi untuk setiap individu secara akurat.

- Pernyataan Masalah 3:
Kurangnya adaptasi sistem terhadap perubahan preferensi pengguna seiring waktu menyebabkan rekomendasi menjadi tidak relevan.

### Goals

- Jawaban pernyataan masalah 1:
Mengembangkan sistem rekomendasi film berbasis machine learning yang dapat secara otomatis menyarankan film sesuai minat dan preferensi pengguna dengan akurasi tinggi.

- Jawaban pernyataan masalah 2:
Menerapkan metode content-based filtering dan collaborative filtering agar sistem mampu mengenali pola preferensi unik dari setiap pengguna.

- Jawaban pernyataan masalah 3:
Mengintegrasikan pendekatan deep learning atau model berbasis embedding (seperti Neural Collaborative Filtering) agar sistem dapat beradaptasi terhadap preferensi dinamis pengguna.

### Solution Statements
Menggunakan dua pendekatan rekomendasi:

- Content-based filtering: 
-Membuat profil pengguna berdasarkan film yang disukai dan mencocokkannya dengan metadata film (genre).

- Collaborative filtering (RecommenderNet & NCF):  
-Menggunakan interaksi antar pengguna dan item untuk menemukan kesamaan preferensi antar pengguna.

Evaluasi model dengan metrik yang terukur:

- Gunakan metrik seperti Root Mean Square Error (RMSE), Mean Absolute Error (MAE) untuk memprediksi rating secara kuantitatif.

- Membandingkan performa model sederhana (baseline) dengan model yang lebih kompleks seperti RecommenderNet atau NCF (Neural Collaborative Filtering).

## Data Understanding & EDA
Proyek sistem rekomendasi film ini menggunakan **dataset publik dari Kaggle** berjudul [Movie Recommendation System](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system). Dataset ini terdiri dari dua file utama:

- **`movies.csv`**  
  Berisi informasi mengenai film, termasuk ID, judul, dan genre.

- **`ratings.csv`**  
  Berisi penilaian (rating) yang diberikan pengguna terhadap film, termasuk ID pengguna, ID film, nilai rating, dan timestamp.

---

### Deskripsi Dataset

#### `movies.csv`

| Nama Kolom | Tipe Data | Deskripsi |
|------------|-----------|-----------|
| `movieId`  | Integer   | ID unik untuk setiap film |
| `title`    | String    | Judul lengkap film beserta tahun rilis |
| `genres`   | String    | Genre film 

#### `ratings.csv`

| Nama Kolom | Tipe Data | Deskripsi |
|------------|-----------|-----------|
| `userId`   | Integer   | ID unik untuk setiap pengguna |
| `movieId`  | Integer   | ID film yang dinilai |
| `rating`   | Float     | Nilai rating yang diberikan pengguna |
| `timestamp`| Integer   | Waktu saat rating diberikan|

---

### Cek data sample movie & rating
```
df_movie
df_rating
```
![datamovie](https://github.com/zakialfadilah/Movie-Recommendations/blob/main/assets/head_datamovies.png?raw=true)

Setelah menampilkan data sample dari df_movie, didapatkan bahwa df_movie memiliki fitur seperti movieId, title dan genres dan dengan total data sebanyak 62.423 data.

![datarating](https://github.com/zakialfadilah/Movie-Recommendations/blob/main/assets/head_dataratings.png?raw=true)

Setelah menampilkan data sample dari df_movie, didapatkan bahwa df_movie memiliki fitur seperti userId, movieId, rating, timestamp dan dengan total data sebanyak 25.000.095 data.

### Cek data null

`print(df_movie.isnull().sum())`
`print(df_rating.isnull().sum())`
```
movieId    0
title      0
genres     0
dtype: int64
userId       0
movieId      0
rating       0
timestamp    0
```
Berdasarkan pengecekan data null, didapatkan bahwa dataset tidak memiliki data null.

### Cek data duplikat

`print(df_movie.duplicated().sum())`
`print(df_rating.duplicated().sum())`
```
0
0
```
Berdasarkan pengecekan data duplikat, didapatkan bahwa dataset tidak memiliki data duplikat.

### Distribusi data rating

berikut merupakan visualisasi distribusi data rating, didapatkan penyebaran rating dari 0,5 hingga 5.

![distribusirating](https://github.com/zakialfadilah/Movie-Recommendations/blob/main/assets/distribusi_datarating.png?raw=true)

### Distribusi film berdasarkan paling banyak di-rating
![movierate](https://github.com/zakialfadilah/Movie-Recommendations/blob/main/assets/most_rated.png?raw=true)

Berdasarkan visualisasi ini, didapatkan top 10 film, berdasarkan total banyaknya film di-rating pengguna.

### Distribusi genre film terpopuler
![moviegenre](https://github.com/zakialfadilah/Movie-Recommendations/blob/main/assets/most%20genres.png?raw=true)

Berdasarkan visualisasi ini, didapatkan urutangenre film dari genre terpopuler hingga genre yang kurang populer. Drama menduduki posisi satu sebagai genre film terpopuler dan adventure menjadi genre film paling tidak populer.

## Data Preparation

### Data preparation untuk content based filtering
- ### Subset dataset
    `df_movie_sample = df_movie.sample(n=20000, random_state=42).reset_index(drop=True)`

    Tahap pertama dilakukan pembuatan data frame baru dengan mengambil 20.000 data dari dataset utama, hal ini dilakukan dikarenakan terbatasnya RAM & kemampuan mesin untuk mengelola data dikarenakan dataset utama yang sangat banyak.

- ### Preprocessing genre
    ```
    df_movie_sample['genres'] = df_movie_sample['genres'].str.replace('|', ' ')
    ```
    Tahap kedua, dilakukan preprocessing genre untuk memisahkan genre menjadi terpisah. dikarenakan pada dataset awal, genre sebuah film bisa lebih dari satu, dengan kode ini genre bisa dipisah yang nantinya bisa diproses TF-ID.

    sebelum= 
    `Adventure Animation Children Comedy`
    sesudah=
    `[Adventure, Animation, Children, Comedy]`

- ### TF-IDF
    ```
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_movie_sample['genres'])
    ```
    Tahap selanjutnya menggunakan TF-IDF (Term Frequency-Inverse Document Frequency) untuk mengubah data genre yang sudah dipisah menjadi representasi numerik. TF-IDF memberi bobot pada setiap genre berdasarkan seberapa sering genre tersebut muncul dalam satu film (TF) dan seberapa jarang genre tersebut muncul di seluruh film (IDF). Hasilnya berupa matriks yang merepresentasikan setiap film dalam bentuk vektor, yang dapat digunakan untuk mengukur kemiripan antar film dalam sistem rekomendasi.

### Data preparation untuk Collaborative Filtering
- ### Subset dataset
    `df_rating_sample = df_rating.sample(n=20000, random_state=42).copy()`

     Tahap pertama dilakukan pembuatan data frame baru dengan mengambil 20.000 data dari dataset rating utama, hal ini dilakukan dikarenakan terbatasnya RAM & kemampuan mesin untuk mengelola data dikarenakan dataset utama yang sangat banyak.
 
 - ### Encoding & Statistik Dataset Interaksi

1. **Extract unique IDs**  
    ```
    user_ids = df_rating_sample['userId'].unique().tolist()
    movie_ids = df_rating_sample['movieId'].unique().tolist()
    ```
   Menghasilkan daftar `user_ids` dan `movie_ids` unik agar bisa dibuat mapping ke index numerik.

2. **Create bijective mapping**  
    ```
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    userencoded2user = {i: x for x, i in user2user_encoded.items()}
    movieencoded2movie = {i: x for x, i in movie2movie_encoded.items()}
    ```
   `user2user_encoded`, `movie2movie_encoded`: mapping dari ID ke numeric index.  
   
   `userencoded2user`, `movieencoded2movie`: mapping balik untuk interpretasi hasil.

3. **Map dan simpan di dataframe**  
    ```
    df_rating_sample['user'] = df_rating_sample['userId'].map(user2user_encoded)
    df_rating_sample['movie'] = df_rating_sample['movieId'].map(movie2movie_encoded)
    ```
   Menambahkan kolom `user` dan `movie` yang berisi index numerik sebagai input ke model.

4. **Compute dataset stats**  
    ```
    num_users = len(user2user_encoded)
    num_movies = len(movie2movie_encoded)
    min_rating = df_rating_sample['rating'].min()
    max_rating = df_rating_sample['rating'].max()
    ```
   `num_users`, `num_movies`: jumlah total unique user dan film untuk define embedding sizes.  
   `min_rating`, `max_rating`: rentang nilai rating untuk keperluan normalisasi dan output scaling.
- ### Extraksi fitur, target & split data
    ```
    x = df_rating_sample[['user', 'movie']].values
    y = df_rating_sample['rating'].values.astype(np.float32)
    
    # Train/Test split
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    ```
    Pada tahapan ini data dibagi menjadi 80% untuk data pelatihan (training set)
    20% untuk data validasi (validation set)


## Modeling

### Content Based Filtering
- ### Cosine Similarity
    `cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)`
Tahapan ini digunakan untuk mengukur tingkat kemiripan antara dua film berdasarkan representasi vektor mereka dari TF-IDF. Nilai similarity berkisar dari 0 (tidak mirip) hingga 1 (sangat mirip).
- ### Test fungsi rekomendasi
    ```
    movie_title = 'The Silent Revolution (2018)'
    ```
    ```
    recommendations_content = get_recommendations_content(movie_title)
    print(f"Rekomendasi untuk '{movie_title}':")
    for i, rec in enumerate(recommendations_content):
        print(f"{i+1}. {rec}")
    ```
    Berdasarkan tes fungsi rekomendasi diatas, didapatkan 10 rekomendasi film sebagai berikut:
    ```
        Rekomendasi untuk 'The Silent Revolution (2018)':
    1. Tony Takitani (2004)
    2. Hunger (2001)
    3. Yugotrip (2004)
    4. Afterimage (2017)
    5. Mourning for Anna (2010)
    6. The Tobacconist (2018)
    7. Snowland (2005)
    8. The Song of Sway Lake (2017)
    9. Sexual Dependency (Dependencia sexual) (2003)
    10. Arrhythmia (2017)
    ```
### Collaborative Filtering

Untuk pendekatan Collaborative Filtering, digunakan dua model neural network, yaitu RecommenderNet dan Neural Collaborative Filtering (NCF). Kedua model ini memanfaatkan interaksi historis antara pengguna dan item (dalam hal ini, film) untuk mempelajari pola preferensi pengguna.

RecommenderNet merupakan model deep learning berbasis embedding yang memetakan pengguna dan item ke dalam ruang vektor berdimensi rendah, lalu menghitung kesamaan antara embedding pengguna dan item melalui operasi dot product. Model ini sederhana namun cukup efektif dalam menangkap hubungan laten antara pengguna dan item.

Neural Collaborative Filtering (NCF) adalah pengembangan lebih lanjut dari RecommenderNet yang menggantikan operasi dot product dengan multilayer perceptron (MLP), sehingga mampu mempelajari hubungan non-linear antara pengguna dan item. NCF cenderung lebih fleksibel dan powerful karena dapat menyesuaikan arsitektur dan parameter jaringan untuk meningkatkan akurasi prediksi.

| Aspek          | RecommenderNet                        | Neural Collaborative Filtering (NCF)     |
| -------------- | ---------------------------------------- | -------------------------------------------- |
| **Kelebihan**  | - Arsitektur sederhana dan cepat dilatih | - Menangkap hubungan non-linear kompleks     |
|                | - Cocok untuk dataset skala sedang       | - Lebih akurat dalam data besar dan kompleks |
|                | - Menghasilkan user/item embedding       | - Fleksibel: bisa ditambah fitur lain        |
| **Kekurangan** | - Kurang efektif untuk hubungan kompleks | - Pelatihan lebih lama dan kompleks          |
|                | - Rentan terhadap cold start             | - Overfitting jika tidak dikontrol           |
|                |                                          | - Butuh banyak data interaksi                |

### Recommendernet
Arsitektur model RecommenderNet dirancang sebagai sistem rekomendasi berbasis pembelajaran mendalam yang menggunakan pendekatan embedding untuk merepresentasikan relasi antara pengguna dan film. Model ini dikembangkan dengan mewarisi kelas Model dari Keras, memungkinkan fleksibilitas dalam mendefinisikan struktur dan perilaku model secara kustom.

Pada tahap inisialisasi, model menerima parameter jumlah total pengguna dan film yang telah dipetakan sebelumnya, serta dimensi embedding sebesar 50 sebagai representasi fitur laten dari masing-masing entitas. Dua buah layer embedding didefinisikan—masing-masing untuk pengguna dan film—yang akan mengubah ID diskrit menjadi vektor berdimensi tetap. Vektor-vektor ini selanjutnya digabungkan menggunakan operasi dot product untuk menghasilkan nilai kecocokan antara pengguna dan film.

Nilai dot product yang diperoleh kemudian diproses melalui layer dense dengan aktivasi sigmoid, yang berfungsi membatasi output dalam rentang 0 hingga 1. Untuk mengembalikan hasil prediksi ke skala rating sebenarnya, model juga menyimpan parameter min_rating dan max_rating yang nantinya digunakan dalam proses rescaling output.
```
class RecommenderNet(Model):
    def __init__(self, num_users, num_movies, embedding_size=50, min_rating=0.5, max_rating=5.0):
        super().__init__()
        self.user_embedding = layers.Embedding(num_users, embedding_size, embeddings_initializer='he_normal')
        self.movie_embedding = layers.Embedding(num_movies, embedding_size, embeddings_initializer='he_normal')
        self.dot = layers.Dot(axes=1)
        self.output_dense = layers.Dense(1, activation='sigmoid')
        self.min_rating = min_rating
        self.max_rating = max_rating

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        dot_product = self.dot([user_vector, movie_vector])
        scaled_output = self.output_dense(dot_product) * (self.max_rating - self.min_rating) + self.min_rating
        return scaled_output

```

Setelah arsitektur RecommenderNet berhasil didefinisikan, langkah berikutnya adalah menginisialisasi dan menyusun model tersebut untuk proses pelatihan. Model diinisialisasi dengan parameter jumlah total pengguna (num_users) dan film (num_movies) yang telah dipetakan dari dataset, serta batas nilai rating minimum dan maksimum sesuai dengan skala yang digunakan.

Model kemudian dikompilasi menggunakan fungsi loss Mean Squared Error (MSE), yang umum digunakan dalam tugas regresi untuk mengukur rata-rata kuadrat selisih antara nilai prediksi dan nilai sebenarnya. Optimizer yang digunakan adalah Adam dengan learning rate sebesar 0.001, yang dikenal efektif dalam mempercepat proses konvergensi pada pelatihan model neural network.

Proses pelatihan model dilakukan dengan memanggil fungsi fit(), di mana data pelatihan (x_train, y_train) digunakan untuk mengajarkan model mempelajari pola interaksi antara pengguna dan film. Selain itu, data validasi (x_val, y_val) digunakan untuk memantau performa model selama pelatihan, sehingga dapat diketahui apakah model mengalami overfitting. Model dilatih selama 10 epoch dengan ukuran batch sebesar 64, dan parameter verbose=1 digunakan untuk menampilkan progres pelatihan secara detail di konsol.

```
model_recommendernet = RecommenderNet(num_users, num_movies, min_rating=min_rating, max_rating=max_rating)

# Kompilasi model
model_recommendernet.compile(
    loss='mse',
    optimizer=Adam(learning_rate=0.001)
)

# Melatih model
history_rnet = model_recommendernet.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=64,
    epochs=10,
    verbose=1
)

```



### NCF
Arsitektur NCF (Neural Collaborative Filtering) dirancang sebagai sistem rekomendasi berbasis deep learning yang lebih kompleks dan fleksibel dibandingkan model klasik seperti matrix factorization. Model ini menggabungkan embedding pengguna dan item (film) dan kemudian memproses hasil gabungan tersebut melalui beberapa lapisan fully connected, sehingga mampu menangkap pola interaksi non-linear yang lebih kaya antara pengguna dan film.

Pada tahap inisialisasi, model menerima parameter jumlah pengguna (num_users) dan film (num_movies), serta dimensi embedding (embedding_size) yang digunakan untuk merepresentasikan fitur laten masing-masing entitas. Dua buah layer embedding didefinisikan—masing-masing untuk pengguna dan film—yang akan mengubah ID menjadi vektor berdimensi tetap.

Lapisan pertama memiliki 128 neuron dengan aktivasi ReLU, dan diikuti oleh dropout sebesar 0.3 sebagai mekanisme regularisasi untuk mencegah overfitting. Lapisan kedua memiliki 64 neuron dengan aktivasi ReLU, juga diikuti oleh dropout dengan rasio yang sama.

Output dari jaringan neural ini kemudian diproses melalui layer dense terakhir dengan satu neuron dan aktivasi sigmoid. Output sigmoid yang berada di antara 0 dan 1 dikalibrasi ulang ke skala rating yang sebenarnya dengan memanfaatkan nilai min_rating dan max_rating yang disimpan sebelumnya.
```
class NCF(Model):
    def __init__(self, num_users, num_movies, min_rating=0.5, max_rating=5.0, embedding_size=50):
        super().__init__()

        # Membuat layer embedding untuk user
        self.user_embedding = layers.Embedding(num_users, embedding_size)

        # Membuat layer embedding untuk movie
        self.movie_embedding = layers.Embedding(num_movies, embedding_size)

        # Menggabungkan vektor user dan movie
        self.concat = layers.Concatenate()

        # Fully connected layer pertama
        self.dense1 = layers.Dense(128, activation='relu')

        # Dropout untuk regularisasi
        self.dropout1 = layers.Dropout(0.3)

        # Fully connected layer kedua
        self.dense2 = layers.Dense(64, activation='relu')

        # Dropout kedua untuk mengurangi overfitting
        self.dropout2 = layers.Dropout(0.3)

        # Output layer dengan aktivasi sigmoid
        self.output_dense = layers.Dense(1, activation='sigmoid')

        # Nilai minimum dan maksimum rating
        self.min_rating = min_rating
        self.max_rating = max_rating

    # Fungsi pemanggilan model
    # Menerima input pasangan user dan movie (dalam bentuk ID), lalu menghasilkan prediksi rating

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        x = self.concat([user_vector, movie_vector])
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        output = self.output_dense(x) * (self.max_rating - self.min_rating) + self.min_rating
        return output
```

Langkah selanjutnya adalah menginisialisasi dan menyusun model tersebut agar siap untuk proses pelatihan. Model diinisialisasi dengan parameter jumlah total pengguna dan film yang telah dipetakan dari dataset, serta rentang nilai rating yang digunakan dalam sistem rekomendasi.

Model kemudian dikompilasi dengan fungsi loss Mean Squared Error (MSE), yang umum digunakan untuk masalah regresi, karena mengukur rata-rata kuadrat perbedaan antara nilai prediksi dan nilai aktual. Sebagai optimizer, digunakan Adam dengan learning rate sebesar 0.001. Adam merupakan pilihan populer karena mampu menggabungkan keunggulan metode momentum dan adaptive learning rate, sehingga mempercepat proses pelatihan dan meningkatkan stabilitas.

Proses pelatihan dilakukan dengan memanggil fungsi fit(), menggunakan data pelatihan (x_train, y_train) untuk melatih model mempelajari pola interaksi antara pengguna dan film. Selain itu, data validasi (x_val, y_val) disediakan agar model dapat dievaluasi secara berkala setiap epoch, sehingga kinerja pada data yang belum pernah dilihat dapat dipantau dan risiko overfitting dapat dikendalikan.

Model dilatih selama 10 epoch dengan ukuran batch sebesar 64. Pengaturan verbose=1 memastikan bahwa proses pelatihan ditampilkan secara terperinci pada konsol, termasuk perkembangan nilai loss dan val_loss di setiap epoch.



```
model_ncf = NCF(num_users, num_movies, min_rating=min_rating, max_rating=max_rating)
model_ncf.compile(
    loss='mse',
    optimizer=Adam(learning_rate=0.001)
)

# Melatih model dengan data pelatihan
history_ncf = model_ncf.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=64,
    epochs=10,
    verbose=1
)
```
### Top N Recommendation RecommenderNet & NFC
Fungsi get_top_n_recommendations dirancang untuk menghasilkan daftar rekomendasi film terbaik bagi seorang pengguna berdasarkan model yang telah dilatih. Fungsi ini bersifat fleksibel dan dapat digunakan baik untuk model RecommenderNet maupun NCF, dengan menyesuaikan format input sesuai arsitektur masing-masing.


```
def get_top_n_recommendations(model, user_id, n=10, model_type='recommendernet'):
    # Ambil movie yang belum pernah diberi rating oleh user
    user_encoded = user2user_encoded[user_id]
    watched_movie_ids = df_rating_sample[df_rating_sample['userId'] == user_id]['movieId'].tolist()
    watched_movie_encoded = [movie2movie_encoded[m] for m in watched_movie_ids if m in movie2movie_encoded]
    all_movies = set(range(num_movies))
    movies_to_predict = list(all_movies - set(watched_movie_encoded))

    user_array = np.full(len(movies_to_predict), user_encoded)
    movie_array = np.array(movies_to_predict)

    if model_type == 'recommendernet':
        preds = model.predict(np.stack([user_array, movie_array], axis=1)).flatten()
    elif model_type == 'ncf':
        preds = model.predict([user_array, movie_array]).flatten()

    top_indices = preds.argsort()[-n:][::-1]
    top_movie_encoded = movie_array[top_indices]
    top_movie_ids = [movieencoded2movie[i] for i in top_movie_encoded]
    top_titles = df_movie[df_movie['movieId'].isin(top_movie_ids)]['title'].values
    return top_titles
```

Proses diawali dengan mengonversi user_id asli ke dalam format encoding numerik (user_encoded) menggunakan pemetaan yang telah dibangun sebelumnya. Kemudian, sistem mengidentifikasi seluruh film yang belum pernah diberi rating oleh pengguna tersebut. Ini dilakukan dengan mengambil semua ID film (movieId) yang pernah dinilai oleh pengguna dari dataset rating (df_rating_sample), lalu mengubahnya ke dalam bentuk encoded melalui dictionary movie2movie_encoded.

Setelah diketahui film apa saja yang belum ditonton, dibuatlah dua array: satu berisi ID pengguna yang di-replicate sebanyak jumlah film yang akan diprediksi (user_array), dan satu lagi berisi encoded ID dari film-film yang belum ditonton (movie_array).

Selanjutnya, prediksi rating dilakukan dengan memanfaatkan model yang telah dilatih. Jika model yang digunakan adalah RecommenderNet, input diberikan dalam bentuk array dua dimensi hasil stacking dari user_array dan movie_array. Sementara untuk NCF, model menerima dua array terpisah sesuai definisi fungsi call pada arsitektur tersebut. Hasil prediksi diratakan menjadi satu dimensi agar memudahkan proses pemeringkatan.

Model akan mengembalikan prediksi rating untuk semua film yang belum ditonton. Dengan menggunakan fungsi argsort, sistem mengambil indeks dari n film dengan prediksi tertinggi. Kemudian, indeks ini diubah kembali ke ID film asli menggunakan movieencoded2movie, dan akhirnya dikonversi ke dalam bentuk judul film dengan mencocokkannya pada dataset film (df_movie).

Setelah dilakukan percobaan untuk salah satu user terhadap Top N recommendation menggunakan RecommenderNet, didapatkan hasil seperti berikut:
```Top 10 rekomendasi film untuk user: ['Sense and Sensibility (1995)'
 'Star Wars: Episode IV - A New Hope (1977)' "Nobody's Fool (1994)"
 'Gone with the Wind (1939)' 'Rain Man (1988)' "Look Who's Talking (1989)"
 'Spirited Away (Sen to Chihiro no kamikakushi) (2001)' 'Serenity (2005)'
 'Chronicles of Narnia: The Lion, the Witch and the Wardrobe, The (2005)'
 'V for Vendetta (2006)']
 ```
 
 dan hasil Top N recommendation menggunakan model NCF, mendapatkan hasil seperti berikut=
 ```
 ['Apartment, The (1960)' 'Third Man, The (1949)' 'Atlantic City (1980)'
 'Gods and Monsters (1998)' 'Run Lola Run (Lola rennt) (1998)'
 "Guess Who's Coming to Dinner (1967)" 'Brokeback Mountain (2005)'
 'Death Race (2008)' 'Keeper of Lost Causes, The (Kvinden i buret) (2013)'
 'Leviathan (2014)']
 ```
 


## Evaluation

Pada tahap evaluasi, digunakan tiga metrik utama yang umum dipakai dalam tugas regresi, khususnya dalam sistem rekomendasi: **MSE (Mean Squared Error)**, **RMSE (Root Mean Squared Error)**, dan **MAE (Mean Absolute Error)**. Masing-masing metrik memberikan perspektif yang berbeda terhadap seberapa akurat model dalam memprediksi rating pengguna terhadap film.

`Mean Squared Error (MSE)`

MSE mengukur rata-rata dari kuadrat selisih antara nilai aktual dan nilai prediksi. Semakin kecil nilai MSE, semakin baik performa model.

**Rumus:**

`MSE = (1/n) × Σ (yᵢ - ŷᵢ)²`

- yᵢ = nilai aktual  
- ŷᵢ = nilai prediksi  
- n  = jumlah sampel  

MSE memberikan penalti yang lebih besar terhadap kesalahan besar karena menggunakan kuadrat dari selisih.
`Root Mean Squared Error (RMSE)`

RMSE merupakan akar kuadrat dari MSE. Metode ini memiliki satuan yang sama dengan skala rating, sehingga hasilnya lebih mudah diinterpretasikan dalam konteks evaluasi sistem rekomendasi.

**Rumus:**

`RMSE = √[ (1/n) × Σ (yᵢ - ŷᵢ)² ]`

RMSE sering digunakan sebagai metrik utama karena menggabungkan penalti MSE terhadap outlier, namun tetap memberikan hasil dalam satuan yang intuitif.

`Mean Absolute Error (MAE)`

MAE mengukur rata-rata dari nilai absolut selisih antara prediksi dan nilai aktual. Berbeda dengan MSE dan RMSE, MAE tidak memberikan penalti tambahan terhadap kesalahan besar.

**Rumus:**

`MAE = (1/n) × Σ |yᵢ - ŷᵢ|`

MAE sering digunakan sebagai pelengkap evaluasi karena lebih robust terhadap outlier, namun memberikan estimasi kesalahan rata-rata secara langsung.

### Evaluasi - Content based filtering

Setelah pembuatan rekomendasi film menggunakan Content based filtering, dilakukan evaluasi precision@k untuk mengetahui apakah model bekerja cukup baik, berikut merupakan kode yang saya gunakan untuk evaluasi model content based filtering
```
def evaluate_similarity(df, title, top_n=10):
    recommended = get_recommendations_content(title, top_n)
    target_genres = df[df['title'] == title]['genres'].values[0]

    match_count = 0
    for rec_title in recommended:
        rec_genres = df[df['title'] == rec_title]['genres'].values[0]
        if any(genre in target_genres for genre in rec_genres.split('|')):
            match_count += 1

    precision = match_count / top_n
    print(f"Precision@{top_n} untuk '{title}': {precision:.2f}")
    return precision
```
Berdasarkan evaluasi terhadap film The Silent Revolution (2018), diperoleh hasil Precision@10 sebesar 1.00. Hal ini menunjukkan bahwa seluruh film yang direkomendasikan memiliki setidaknya satu genre yang sama dengan film acuan. Artinya, sistem berhasil memberikan rekomendasi yang secara konten sangat relevan dan sesuai dengan karakteristik genre dari film input tersebut.
    

---
### Visualiasasi perbandingan evaluasi RMSE Model RecommenderNet & NCF

![rmse](https://github.com/zakialfadilah/Movie-Recommendations/blob/main/assets/rmse.png?raw=true)

`RMSE RecommenderNet: 1.0964`
`RMSE NCF: 1.1055`
Berdasarkan hasil evaluasi menggunakan metrik RMSE (Root Mean Squared Error), model RecommenderNet menunjukkan performa yang sedikit lebih baik dibandingkan dengan model NCF . Selisih ini mengindikasikan bahwa prediksi rating dari RecommenderNet lebih mendekati nilai aktual secara rata-rata.


### Visualiasasi perbandingan evaluasi MSE Model RecommenderNet & NCF

![mse](https://github.com/zakialfadilah/Movie-Recommendations/blob/main/assets/mse.png?raw=true)

Berdasarkan visualisasi, RecommenderNet memiliki nilai RMSE lebih rendah dibandingkan NCF , grafik validasi menunjukkan bahwa model ini mulai overfit terhadap data pelatihan setelah beberapa epoch. Sebaliknya, NCF menunjukkan performa yang lebih stabil dan konsisten di data validasi.

### Evaluasi MAE Model RecommenderNet & NCF

```
mae_rec = mean_absolute_error(y_val, y_pred_rec)
mae_ncf = mean_absolute_error(y_val, y_pred_ncf)
```


`MAE RecommenderNet: 0.8942`
`MAE NCF: 0.8639`


Meskipun RecommenderNet unggul pada metrik RMSE, model NCF menunjukkan keunggulan pada metrik MAE, yang mengindikasikan bahwa prediksi NCF secara umum lebih stabil dan lebih konsisten terhadap nilai rating aktual, meskipun mungkin kurang presisi terhadap nilai-nilai ekstrem.


### Evaluasi top n RecommenderNet
```
example_user_encoded = 2

top_10_recommendations = get_top_n_recommendations(model_recommendernet, example_user_encoded, model_type='recommendernet', top_n=10)

print("Top 10 rekomendasi film untuk user:", top_10_recommendations)
```
berdasarkan top n dari RecommenderNet dengan mengambil index user "2" akan menghasilkan
`['Piano, The (1993)' 'Son in Law (1993)'
 'Monty Python and the Holy Grail (1975)'
 'Seventh Seal, The (Sjunde inseglet, Det) (1957)'
 'Elephant Man, The (1980)' 'Unbreakable (2000)' 'Notebook, The (2004)'
 'Wedding Crashers (2005)' 'Guardians of the Galaxy (2014)'
 'John Wick: Chapter Two (2017)']`

### Evaluasi top n NCF
```
example_user_encoded = 2

top_10_recommendations = get_top_n_recommendations(model_ncf, example_user_encoded, model_type='ncf', top_n=10)

print("Top 10 rekomendasi film untuk user:", top_10_recommendations)
```
berdasarkan top n dari RecommenderNet dengan mengambil index user "2" akan menghasilkan
`['Surviving the Game (1994)' 'Third Man, The (1949)' 'Shine (1996)'
 'Gods and Monsters (1998)' 'American Flyers (1985)'
 'Of Mice and Men (1992)' 'Hellraiser (1987)' 'Death Race (2008)'
 'Keeper of Lost Causes, The (Kvinden i buret) (2013)' 'Leviathan (2014)']`
 
 ## Conclusion
 
 Dalam proyek pengembangan sistem rekomendasi film ini, saya mengimplementasikan dua pendekatan utama, yaitu Content-Based Filtering dan Collaborative Filtering. Pada pendekatan Collaborative Filtering, saya menguji dua arsitektur model deep learning, yakni RecommenderNet dan Neural Collaborative Filtering (NCF).

Hasil evaluasi menunjukkan bahwa kedua model memiliki performa yang cukup kompetitif, dengan nilai RMSE sebesar 1.0964 untuk RecommenderNet dan 1.1055 untuk NCF. Meskipun RecommenderNet sedikit lebih unggul dalam hal RMSE dan MSE, model NCF menunjukkan performa yang lebih stabil dan konsisten, terutama dalam proses pelatihan dan validasi.

Detail metrik evaluasi dari masing-masing model adalah sebagai berikut:

RecommenderNet:

`RMSE: 1.0964`

`MAE: 0.8942`

Final Training MSE: 0.0261

Final Validation MSE: 1.2021

NCF:

`RMSE: 1.1055`

`MAE: 0.8639`

`Final Training MSE: 0.0290`

`Final Validation MSE: 1.2222`

Berdasarkan evaluasi tersebut, saya memilih NCF sebagai model akhir yang digunakan dalam sistem rekomendasi ini karena kestabilannya dalam proses pelatihan dan kemampuannya dalam menghasilkan prediksi yang lebih merata, meskipun memiliki RMSE sedikit lebih tinggi dibandingkan RecommenderNet. Selain itu, nilai MAE yang lebih rendah pada NCF menunjukkan bahwa prediksi model ini secara umum lebih mendekati nilai rating sesungguhnya.

Pendekatan content-based digunakan sebagai pelengkap untuk memberikan rekomendasi yang lebih personal terutama ketika data interaksi pengguna masih terbatas (cold start problem).
 
 




