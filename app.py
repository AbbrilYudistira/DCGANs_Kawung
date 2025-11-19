import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Atur tata letak halaman
st.set_page_config(layout="wide")

# Mengatur kebijakan presisi campuran
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Fungsi untuk memuat model dengan penanganan kesalahan
@st.cache_resource
def load_gan_models():
    try:
        # Tambahkan compile=False untuk menghindari peringatan saat memuat
        generator = load_model('generator_final201.h5', compile=False)
        discriminator = load_model('discriminator_final201.h5', compile=False)
        return generator, discriminator
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None, None

# Fungsi untuk memproses gambar yang diunggah
def preprocess_image(image):
    # Pastikan gambar dalam mode RGB
    img = image.convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = (img_array.astype(np.float32) / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Memuat model
generator, discriminator = load_gan_models()

# Judul dan deskripsi aplikasi
st.title("Generator Batik Kawung dengan DCGAN")
st.write(
    "Aplikasi ini menggunakan model DCGAN yang dilatih pada dataset Batik Kawung. "
    "Anda dapat menghasilkan gambar batik baru secara acak atau mengunggah gambar Anda sendiri "
    "untuk diperiksa oleh diskriminator."
)

# Tampilkan UI jika model berhasil dimuat
if generator and discriminator:
    col1, col2 = st.columns(2)

    with col1:
        st.header("Generator Batik")
        
        # 1. Slider untuk memilih jumlah gambar
        jumlah_gambar = st.slider("Jumlah gambar:", min_value=1, max_value=10, value=1)

        # 2. Siapkan "Ingatan" (Session State) agar gambar tidak hilang
        if 'batik_generated' not in st.session_state:
            st.session_state['batik_generated'] = []

        # 3. Tombol untuk membuat gambar baru
        if st.button("Buat Gambar Batik Baru"):
            st.write(f"Sedang membuat {jumlah_gambar} variasi batik...")
            temp_images = []
            
            for i in range(jumlah_gambar):
                # Membuat random noise
                latent_dim = 128 
                random_latent_vector = tf.random.normal(shape=(1, latent_dim))
                
                # Generate gambar mentah dari model
                generated_image = generator(random_latent_vector, training=False)
                
                # --- PROSES MEMBUAT GAMBAR LEBIH HALUS (FAKE HD) ---
                # Ambil data gambar
                img_array = generated_image[0].numpy()
                
                # Ubah rentang warna ke [0, 255]
                img_array = ((img_array + 1) * 127.5).astype(np.uint8)
                
                # Ubah menjadi objek Gambar PIL
                img_pil = Image.fromarray(img_array)
                
                # Perbesar gambar agar tidak pecah (Upscaling halus)
                img_hd = img_pil.resize((400, 400), Image.Resampling.LANCZOS)
                
                # Simpan ke list sementara
                temp_images.append(img_hd)
                # ---------------------------------------------------
            
            # Simpan hasil ke ingatan aplikasi (menimpa yang lama)
            st.session_state['batik_generated'] = temp_images

        # 4. Tampilkan gambar dari ingatan
        if st.session_state['batik_generated']:
            gambar_cols = st.columns(2)
            for i, img_hd in enumerate(st.session_state['batik_generated']):
                with gambar_cols[i % 2]:
                    st.image(
                        img_hd, 
                        caption=f"Batik #{i+1}", 
                        width=200 
                    )

    with col2:
        st.header("Diskriminator Batik")
        uploaded_file = st.file_uploader("Unggah gambar batik...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Gambar yang Diunggah', use_container_width=True)

            # Pra-pemrosesan gambar dan buat prediksi
            processed_image = preprocess_image(image)
            prediction = discriminator(processed_image, training=False)

            # Tampilkan hasil prediksi
            st.subheader("Hasil Prediksi Diskriminator:")
            if prediction[0][0] > 0.5:
                st.success(f"Gambar ini kemungkinan **Asli** (skor: {prediction[0][0]:.2f})")
            else:
                st.error(f"Gambar ini kemungkinan **Palsu** (skor: {prediction[0][0]:.2f})")