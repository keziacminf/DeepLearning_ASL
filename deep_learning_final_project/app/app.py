import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
import os
import av
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase 

# ==================== CONFIG ====================

st.set_page_config(
    page_title="ASL Hand Sign Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ukuran input yang digunakan saat training MobileNetV2
MODEL_INPUT_SIZE = (224, 224) 

# ==================== PATHS & CONSTANTS ====================
DATASET_BASE_PATH = 'data/train/' 
CM_PATH = 'evaluation/confusion_matrix_test.png'
METRICS_PATH = 'models/evaluation_metrics.json'
TRAINING_HISTORY_PATH = 'models/training_history.png'

# Daftar semua kelas/label yang mungkin
ALL_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    'SPACE', 'DELETE'
]

# ==================== LOAD UTILITIES ====================

@st.cache_resource
def load_model():
    """Memuat model dan label mapping ke memori"""
    model_path = 'models/asl_model.keras'
    label_path = 'models/label_map.json'
    
    if not os.path.exists(model_path):
        st.error(f"Model tidak ditemukan di: {model_path}")
        st.info("Jalankan `python train_model.py` terlebih dahulu!")
        return None, None
    
    try:
        model = tf.keras.models.load_model(model_path)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label_map = json.load(f)
            labels = {int(v): k for k, v in label_map.items()} 
        else:
            labels = None
        
        return model, labels
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def load_evaluation_metrics():
    """Memuat metrik evaluasi model (Loss dan Accuracy) dari file JSON."""
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        st.error(f"Error reading metrics file: {e}")
        return None

def find_sample_image(label):
    """Mencari path gambar sampel pertama untuk label yang diberikan."""
    class_dir = os.path.join(DATASET_BASE_PATH, label)
    
    if os.path.isdir(class_dir):
        # Ambil semua file di folder, lalu filter hanya yang bertipe gambar
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if files:
            # Mengambil file pertama (misal: '0.jpg' atau '1.png')
            return os.path.join(class_dir, sorted(files)[0]) 
            
    return None

def preprocess_image(image, target_size=MODEL_INPUT_SIZE):
    """Preprocess gambar (numpy array RGB) untuk prediksi MobileNetV2"""
    image = cv2.resize(image, target_size)
    image = image.astype('float32')
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    
    return image

# ==================== REAL-TIME PROCESSOR (INTI DETEKSI REAL-TIME) ====================

class RealTimeProcessor(VideoProcessorBase): 
    # (Kode RealTimeProcessor tidak berubah)
    def __init__(self, model, labels, confidence_threshold):
        self.model = model
        self.labels = labels
        self.confidence_threshold = confidence_threshold
        self.prediction_history = deque(maxlen=10) 

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame: 
        img = frame.to_ndarray(format="rgb24")
        height, width, _ = img.shape
        
        # --- ROI (Region of Interest) ---
        ROI_PERCENTAGE = 0.5
        roi_w = int(width * ROI_PERCENTAGE)
        roi_h = int(height * ROI_PERCENTAGE)
        roi_size = min(roi_w, roi_h)
        roi_w = roi_size
        roi_h = roi_size
        roi_x = int((width - roi_w) / 2)
        roi_y = int((height - roi_h) / 2)
        
        roi = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
        
        # --- PREDICT & SMOOTHING ---
        if roi.size == 0:
            label = "Waiting for Hand..."
            confidence = 0.0
        else:
            processed_roi = preprocess_image(roi, target_size=MODEL_INPUT_SIZE)
            predictions = self.model.predict(processed_roi, verbose=0)[0]
            
            confidence = np.max(predictions)
            predicted_class = np.argmax(predictions)
            
            self.prediction_history.append(predicted_class)
            smoothed_prediction = max(set(self.prediction_history), key=self.prediction_history.count)
            label = self.labels.get(smoothed_prediction, "Unknown")
        
        # --- DRAW RESULT ---
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        text = f"Sign: {label} ({confidence:.2f})"
        color = (0, 255, 0) if confidence >= self.confidence_threshold else (0, 165, 255) # Green or Orange
        
        cv2.rectangle(img_bgr, (5, 5), (400, 45), (0, 0, 0), -1)
        cv2.putText(img_bgr, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        img_rgb_output = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        return av.VideoFrame.from_ndarray(img_rgb_output, format="rgb24")


# ==================== MAIN APP ====================

def main():
    # header
    st.title("🤟 ASL Hand Sign Recognition")
    st.markdown("### Deteksi American Sign Language menggunakan Deep Learning")
    st.markdown("---")
    
    # load model
    with st.spinner("Loading model..."):
        model, labels = load_model()
        eval_metrics = load_evaluation_metrics() 
    
    if model is None:
        st.stop()
    
    st.success("✅ Model berhasil dimuat!")
    
    # sidebar
    with st.sidebar:
        st.header("📋 Menu")
        app_mode = st.radio(
            "Pilih Mode:",
            ["🏠 Home", "📸 Real-time Webcam", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.header("🔧 Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.6, 0.05
        )
        
        # Selectbox untuk memilih Alfabet di Sidebar
        st.markdown("---")
        selected_label = "A" # Default value
        if app_mode == "📸 Real-time Webcam":
             st.subheader("Pilih Gestur Referensi")
             selected_label = st.selectbox(
                 "Pilih Alfabet / Gestur yang ingin dicoba:",
                 options=ALL_LABELS,
                 index=0 # Default ke 'A'
             )


        st.markdown("---")
        if labels:
            st.header("📊 Model Info")
            st.metric("Jumlah Kelas", len(labels))
            with st.expander("Lihat Semua Kelas"):
                class_list = sorted(labels.values())
                cols = st.columns(3)
                for i, cls in enumerate(class_list):
                    cols[i % 3].write(f"• {cls}")
    
    # ==================== MODE: HOME ====================
    
    if app_mode == "🏠 Home":
        st.header("Selamat Datang! 👋")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Tentang Aplikasi
            
            Aplikasi ini menggunakan **Deep Learning** dengan **Transfer Learning (MobileNetV2)** untuk mengenali bahasa isyarat ASL (American Sign Language) dari video *real-time*.
            
            #### 🎯 Fitur Utama:
            - Pengenalan isyarat tangan ASL
            - Deteksi *Real-time* melalui webcam menggunakan WebRTC
            - Visualisasi Confidence Score dan hasil prediksi langsung pada *overlay* video.
            
            #### 🚀 Cara Menggunakan:
            1. Pilih mode **📸 Real-time Webcam** di sidebar.
            2. Klik **START** dan posisikan tangan di kotak hijau (ROI) di video.
            3. Lihat hasil prediksi yang muncul langsung di video.
            """)
        
        with col2:
            st.info("""
            **💡 Tips Performa Terbaik:**
            
            - Gunakan kotak hijau (ROI) sebagai fokus.
            - Pastikan *background* polos.
            - Cahaya (lighting) yang baik.
            - Tangan terlihat jelas dan tidak terpotong.
            """)
            
            st.warning("""
            **⚠️ Catatan:**
            
            Model dilatih menggunakan dataset (Raw) Mendeley ASL Alphabet (28 kelas).
            """)
    
    # ==================== MODE: REAL-TIME WEBCAM (DENGAN GAMBAR REFERENSI DINAMIS) ====================
    
    elif app_mode == "📸 Real-time Webcam":
        st.header("Deteksi Real-time dari Webcam")
        
        # Menggunakan dua kolom, Video di kiri (lebih lebar) dan Gambar Referensi di kanan
        col_video, col_guide = st.columns([2, 1])

        with col_video:
            st.markdown("""
            **Instruksi:**
            1. Klik **START** di bawah.
            2. Posisikan tangan di dalam **kotak hijau** (ROI).
            3. Hasil prediksi ditampilkan langsung pada *overlay* video.
            """)
            
            processor = RealTimeProcessor(model, labels, confidence_threshold)
            
            webrtc_streamer(
                key="realtime_detection_key",
                video_processor_factory=lambda: processor,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
            )
        
        with col_guide:
            st.subheader(f"Contoh Gestur '{selected_label}'")
            
            sample_path = find_sample_image(selected_label)
            
            if sample_path:
                st.image(
                    sample_path, 
                    use_column_width=True
                )
            else:
                st.warning("Gambar referensi dataset tidak ditemukan.")
            
            st.markdown("---")
            st.info("""
            **Tips Deteksi Terbaik:**
            - Gunakan gestur yang jelas dan tegas.
            - Pastikan pencahayaan (lighting) baik.
            - Usahakan menggunakan background polos di kotak hijau.
            """)


    # ==================== MODE: ABOUT (DENGAN METRIK EVALUASI DAN CM) ====================
    
    elif app_mode == "ℹ️ About":
        st.header("Tentang Project")
        
        st.markdown("""
        ### 🎓 ASL Hand Sign Recognition
        
        Project ini menggunakan **Deep Learning** untuk mengenali bahasa isyarat 
        ASL (American Sign Language) dari video.
        
        #### 🧠 Teknologi & Arsitektur:
        - **Framework**: TensorFlow/Keras
        - **Model**: MobileNetV2 (Transfer Learning)
        - **Frontend**: Streamlit + streamlit-webrtc (WebRTC API)
        - **Strategi Pelatihan**: Pelatihan dua fase (Fine-tuning)
        """)
        
        st.header("📈 Hasil Evaluasi Model")
        
        if eval_metrics:
            st.success("Metrik Evaluasi Model (Validation & Test Set):")
            
            # --- Metrik Utama (Validation & Test) ---
            col_v_acc, col_v_loss, col_t_acc, col_t_loss = st.columns(4)
            
            if 'Validation_Accuracy' in eval_metrics:
                 col_v_acc.metric("Validation Acc.", f"{eval_metrics['Validation_Accuracy']*100:.2f}%")
            if 'Validation_Loss' in eval_metrics:
                 col_v_loss.metric("Validation Loss", f"{eval_metrics['Validation_Loss']:.4f}")
            if 'Test_Accuracy' in eval_metrics:
                 col_t_acc.metric("Test Accuracy", f"{eval_metrics['Test_Accuracy']*100:.2f}%")
            if 'Test_Loss' in eval_metrics:
                 col_t_loss.metric("Test Loss", f"{eval_metrics['Test_Loss']:.4f}")

            # --- Classification Report Detail (Test Set) ---
            if 'Classification_Report_Test' in eval_metrics:
                st.markdown("---")
                st.subheader("Classification Report (Test Set)")
                st.markdown("Berikut adalah performa metrik per kelas (Precision, Recall, F1-Score):")

                report_data = eval_metrics['Classification_Report_Test']
                
                table_data = {
                    "Class": [],
                    "Precision": [],
                    "Recall": [],
                    "F1-Score": [],
                    "Support": []
                }
                
                for cls, metrics in report_data.items():
                    table_data["Class"].append(cls)
                    table_data["Precision"].append(f"{metrics['precision']:.4f}")
                    table_data["Recall"].append(f"{metrics['recall']:.4f}")
                    table_data["F1-Score"].append(f"{metrics['f1_score']:.4f}")
                    table_data["Support"].append(metrics['support'])

                st.dataframe(table_data, use_container_width=True, height=500)

                if 'macro avg' in report_data:
                    macro = report_data['macro avg']
                    weighted = report_data['weighted avg']
                    
                    st.markdown("---")
                    col_macro, col_weighted = st.columns(2)
                    
                    with col_macro:
                        st.caption("Macro Average (rata-rata unweighted per kelas)")
                        st.metric("Macro Avg F1-Score", f"{macro['f1_score']:.4f}")
                        st.write(f"Precision: {macro['precision']:.4f}, Recall: {macro['recall']:.4f}")
                    
                    with col_weighted:
                        st.caption("Weighted Average (rata-rata berdasarkan Support)")
                        st.metric("Weighted Avg F1-Score", f"{weighted['f1_score']:.4f}")
                        st.write(f"Precision: {weighted['precision']:.4f}, Recall: {weighted['recall']:.4f}")
                
        else:
            st.info("""
            Metrik evaluasi model detail belum tersedia.
            """)
            st.markdown("---")

        # === TRAINING HISTORY PLOT ===
        st.header("📊 History Pelatihan (Loss & Accuracy)")
        
        if os.path.exists(TRAINING_HISTORY_PATH):
            st.image(TRAINING_HISTORY_PATH, caption='Training & Validation Loss/Accuracy over Epochs', use_column_width=True)
            st.info("""
            Plot ini menunjukkan bagaimana performa model selama pelatihan. Idealnya, kurva Training dan Validation harus
            menurun/meningkat secara bersamaan dan stabil. Jika Validation Loss/Accuracy berpisah jauh
            dengan Training Loss/Accuracy, ini bisa berarti *overfitting*. 
            """)
        else:
            st.warning(f"File Training History (Loss dan Accuracy) tidak ditemukan di: {TRAINING_HISTORY_PATH}")
        
        st.markdown("---")

        # === CONFUSION MATRIX ===
        st.header("🖼️ Visualisasi Confusion Matrix")
        
        if os.path.exists(CM_PATH):
            st.image(CM_PATH, caption='Confusion Matrix - Test Set', use_column_width=True)
            st.info("""
            Confusion Matrix menunjukkan performa model secara detail per kelas. 
            Diagonal utama (dari kiri atas ke kanan bawah) adalah jumlah prediksi yang benar (True Positives).
            """)
            
            cm_val_path = 'evaluation/confusion_matrix_val.png'
            if os.path.exists(cm_val_path):
                st.markdown("---")
                st.image(cm_val_path, caption='Confusion Matrix - Validation Set', use_column_width=True)
                
        else:
            st.warning(f"File Confusion Matrix tidak ditemukan di: {CM_PATH}")
        
        # === END OF CONFUSION MATRIX ===

        st.markdown("---")
        st.subheader("📚 Dataset:")
        st.markdown("""
        - Source: [Mendeley Data](https://data.mendeley.com/datasets/48dg9vhmyk/2)
        - Total Images: ~11,200
        - Jumlah Kelas: 28 (A-Z + DELETE + SPACE)
        """)

# ==================== RUN APP ====================

if __name__ == "__main__":
    main()