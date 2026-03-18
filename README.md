# TubesML

Implementasi Feedforward Neural Network (FFNN) from scratch untuk Tugas Besar 1 IF3270 Pembelajaran Mesin 2025/2026.

## Identitas

Nama tim: AlphaAziz

Anggota:
- Leonard Arif Sutiono / 1823120
- Harfhan Ikhtiar Ahmad Ridzky / 18223123
- Izhar Alif Akbar / 18223129


## Gambaran Proyek

Repository ini berisi:
- Modul FFNN from scratch di folder src/model.
- Notebook pengujian dan analisis eksperimen hyperparameter di src/main_notebook2.ipynb.
- Dataset pengujian global_student_placement_and_salary di src/data/datasetml_2026.csv.
- Dokumen spesifikasi tugas pada folder doc dan file TASK.md.

Target klasifikasi pada dataset:
- placement_status
- Label yang dipakai di notebook: Placed = 1, Not Placed = 0.

## Struktur Repository

Struktur utama:

	TubesML/
	|- doc/
    |  |- AlphaAziz_Tugas Besar Machine Learning 1
	|- src/
	|  |- data/
	|  |  |- datasetml_2026.csv
	|  |- model/
	|  |  |- activation.py
	|  |  |- initializer.py
	|  |  |- loss.py
	|  |  |- model.py
	|  |  |- regularizer.py
	|  |- main.ipynb
	|- requirements.txt
    |- README.md

## Fitur Implementasi FFNN

Implementasi ada pada src/model/model.py dengan komponen berikut.

1. Arsitektur fleksibel
- Jumlah layer bisa ditentukan sendiri melalui add_layer.
- Jumlah neuron tiap layer ditentukan saat membuat Layer.

2. Fungsi aktivasi
- linear
- relu
- sigmoid
- tanh
- softmax
- leaky_relu
- swish

3. Fungsi loss
- mse
- bce
- cce

4. Inisialisasi bobot
- zero
- uniform
- normal
- xavier
- he

5. Regularisasi
- tanpa regularisasi
- l1
- l2

6. Optimizer
- sgd
- adam

7. RMSNorm opsional
- Dapat diaktifkan per hidden layer lewat parameter use_rmsnorm pada Layer.

8. Training dan backpropagation
- Mendukung mini-batch.
- Forward dan backward berjalan untuk input batch.
- Verbose mode mendukung progress bar dengan tqdm.

9. Visualisasi distribusi parameter
- Distribusi bobot + bias (gabungan dan per-layer).
- Distribusi gradien bobot + bias (gabungan dan per-layer).

10. Persistensi model
- save(filename)
- load(filename)

## Setup Environment

Prasyarat:
- Python 3.10 atau lebih baru.

Contoh setup di Windows PowerShell:

	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
	python -m pip install --upgrade pip
	pip install -r requirements.txt

Catatan:
- Jika eksekusi notebook memakai kernel lain, pastikan dependensi terpasang di kernel tersebut.

## Cara Menjalankan Notebook Analisis

Notebook utama pengujian ada di:
- src/main.ipynb

Langkah umum:
1. Buka folder repository di VS Code.
2. Pastikan kernel notebook aktif pada environment yang sudah berisi dependensi.
3. Jalankan seluruh cell dari atas ke bawah.

Isi eksperimen di notebook:
1. Analisis width dan depth.
2. Analisis fungsi aktivasi hidden layer.
3. Analisis learning rate.
4. Analisis regularisasi.
5. Analisis optimizer (Adam vs SGD).
6. Analisis normalisasi RMSNorm.
7. Perbandingan dengan sklearn MLPClassifier.
8. Ringkasan akhir hasil eksperimen.

## Contoh Penggunaan Modul FFNN

Contoh ringkas pemakaian model dari kode Python:

	import numpy as np
	from model.model import FFNN, Layer

	# Misal jumlah fitur input setelah preprocessing = 28
	input_dim = 28

	model = FFNN(loss_name='bce', regularization_type='l2', lam=0.001, optimizer='adam')
	model.add_layer(Layer(input_dim, 32, 'relu', use_rmsnorm=True))
	model.add_layer(Layer(32, 16, 'relu', use_rmsnorm=True))
	model.add_layer(Layer(16, 1, 'sigmoid', use_rmsnorm=False))

	model.initialize_weights(method='xavier', input_dim=input_dim)

	# X_train, y_train, X_val, y_val harus sudah dipreprocess ke numpy array
	model.fit(
		X_train=X_train,
		y_train=y_train,
		X_val=X_val,
		y_val=y_val,
		epochs=30,
		batch_size=32,
		learning_rate=0.01,
		verbose=1
	)

	y_pred = model.forward(X_val)

	# Plot distribusi untuk layer hidden dan output
	# Index layer: 0..n-1, layer terakhir adalah output
	model.plot_weights_distribution([0, 1, 2])
	model.plot_gradients_distribution([0, 1, 2])

	model.save('ffnn_model.pkl')
	loaded_model = FFNN.load('ffnn_model.pkl')

## Catatan Implementasi

- Layer input bersifat implisit (diwakili X), sehingga layer yang disimpan pada model adalah hidden layer dan output layer.
- Untuk plotting gradien, jalankan training/backward terlebih dahulu agar dW dan db tersedia.

## Referensi File Penting

- Implementasi model: src/model/model.py
- Aktivasi: src/model/activation.py
- Loss: src/model/loss.py
- Initializer: src/model/initializer.py
- Regularizer: src/model/regularizer.py
- Notebook analisis utama: src/main.ipynb

## Pembagian Tugas

| Anggota | Detail Kontribusi |
| --- | --- |
| Izhar Alif Akbar (18223129) | Mengimplementasikan struktur utama model FFNN (arsitektur layer, forward propagation, backward propagation, dan alur training mini-batch). Mengimplementasikan komponen inti sesuai spesifikasi (fungsi aktivasi dan loss, inisialisasi bobot, regularisasi, visualisasi distribusi bobot/gradien, serta save/load model). Mengimplementasikan struktur utama notebook pengujian dan pipeline eksperimen. Menulis laporan. Melakukan analisis pada notebook|
| Harfhan Ikhtiar Ahmad Ridzky (18223123) | Mengimplementasikan struktur utama model FFNN (forward propagation, backward propagation, dan alur training mini-batch). Melakukan Analisis pada Notebook. Menulis Laporan |
| Leonard Arif Sutiono (1823120) | Mengimplementasikan Leaky ReLU, Swish. Mengimplementasikan Xavier initialization dan He initialization. Mengimplementasikan metode normalisasi RMSNorm. Mengimplementasikan optimizer Adam. Melakukan Analisis pada Notebook. Menulis Laporan |