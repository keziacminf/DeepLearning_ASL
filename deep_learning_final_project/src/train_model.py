import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ==================== DATA LOADING ====================

def load_data(data_dir, img_size=(224, 224)):
    """Load gambar dari folder"""
    images = []
    labels = []
    label_map = {}
    
    print(f"\nMemuat data dari: {data_dir}")
    
    # mapping untuk handle nama folder yang beda
    folder_name_mapping = {
        'DEL': 'DELETE',
        'SPACE': 'SPACE',
        'NOTHING': 'NOTHING'
    }
    
    # ambil semua folder (setiap folder = 1 kelas)
    classes = []
    for d in sorted(os.listdir(data_dir)):
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.'):
            class_dir = os.path.join(data_dir, d)
            files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(files) > 0:
                class_name = folder_name_mapping.get(d, d)
                classes.append((d, class_name))
    
    if len(classes) == 0:
        raise ValueError("Tidak ada kelas dengan data yang ditemukan!")
    
    label_names = [label_name for _, label_name in classes]
    print(f"Kelas ditemukan ({len(classes)}): {label_names}")
    
    # bikin mapping label
    for idx, (folder_name, label_name) in enumerate(classes):
        label_map[label_name] = idx
    
    # load semua gambar
    print("\nMemuat gambar...")
    for folder_name, label_name in classes:
        class_dir = os.path.join(data_dir, folder_name)
        class_idx = label_map[label_name]
        
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  [{label_name}] {len(image_files)} gambar")
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is not None:
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(class_idx)
    
    print(f"\nTotal gambar: {len(images)}")
    return np.array(images), np.array(labels), label_map

# ==================== MODEL BUILDING ====================

def create_transfer_model(input_shape, num_classes):
    """Model dengan transfer learning MobileNetV2"""
    
    # load MobileNetV2 pre-trained
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # freeze base model
    base_model.trainable = False
    
    # tambah custom layers
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

# ==================== PLOTTING ====================

def plot_training_history(history, save_path='models/training_history.png'):
    """Plot grafik training"""
    plt.figure(figsize=(14, 5))
    
    # accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.title('Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
    plt.title('Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"✓ Training history: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='evaluation/confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    os.makedirs('evaluation', exist_ok=True)
    plt.savefig(save_path, dpi=200)
    print(f"✓ Confusion matrix: {save_path}")
    plt.close()

# ==================== MAIN PROGRAM ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ASL HAND SIGN RECOGNITION - TRAINING & EVALUATION")
    print("="*70)
    
    # ========== SETUP ==========
    
    train_dir = 'data/train'
    test_dir = 'data/test'
    
    if not os.path.exists(train_dir):
        print(f"\nError: Folder '{train_dir}' tidak ditemukan!")
        print("Pastikan struktur:")
        print("  data/train/A/")
        print("  data/train/B/")
        print("  ...")
        exit()
    
    # ========== LOAD DATA ==========
    
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    X_train, y_train, label_map = load_data(train_dir, img_size=(224, 224))
    
    if len(X_train) == 0:
        print("\nError: Tidak ada data training!")
        exit()
    
    print(f"\nInfo Dataset:")
    print(f"Total gambar: {len(X_train)}")
    print(f"Ukuran gambar: {X_train[0].shape}")
    print(f"Jumlah kelas: {len(label_map)}")
    
    # preprocessing MobileNetV2
    print("\nPreprocessing...")
    X_train = tf.keras.applications.mobilenet_v2.preprocess_input(X_train)
    
    # split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=42,
        stratify=y_train
    )
    
    print(f"Data training: {len(X_train)}")
    print(f"Data validation: {len(X_val)}")
    
    # data augmentation
    print("\nData augmentation...")
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2),
    ])
    
    # ========== BUILD MODEL ==========
    
    print("\n" + "="*70)
    print("STEP 2: BUILDING MODEL")
    print("="*70)
    
    input_shape = X_train[0].shape
    num_classes = len(label_map)
    
    model, base_model = create_transfer_model(input_shape, num_classes)
    
    print("\nModel Architecture:")
    model.summary()
    
    # compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # setup dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(1000).batch(32)
    train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    
    # ========== TRAINING PHASE 1 ==========
    
    print("\n" + "="*70)
    print("STEP 3: TRAINING (PHASE 1 - TOP LAYERS)")
    print("="*70)
    print("\nTraining top layers dengan base model frozen...")
    
    history1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )
    
    # ========== FINE-TUNING PHASE 2 ==========
    
    print("\n" + "="*70)
    print("STEP 4: FINE-TUNING (PHASE 2 - UNFREEZE LAYERS)")
    print("="*70)
    
    base_model.trainable = True
    
    # freeze semua kecuali 30 layer terakhir
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    print(f"\nTrainable layers: {sum(1 for l in model.layers if l.trainable)}")
    
    # compile ulang dengan learning rate lebih kecil
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nFine-tuning model...")
    history2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )
    
    # gabungkan history
    history_combined = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss']
    }
    
    class CombinedHistory:
        def __init__(self, history_dict):
            self.history = history_dict
    
    combined_history = CombinedHistory(history_combined)
    
    # ========== SAVE MODEL ==========
    
    print("\n" + "="*70)
    print("STEP 5: SAVING MODEL")
    print("="*70)
    
    os.makedirs('models', exist_ok=True)
    model.save('models/asl_model.keras')
    print("\nModel: models/asl_model.keras")
    
    with open('models/label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    print("Label map: models/label_map.json")
    
    plot_training_history(combined_history)
    
    # ========== EVALUATION ON VALIDATION SET ==========
    
    print("\n" + "="*70)
    print("STEP 6: EVALUATION (VALIDATION SET)")
    print("="*70)
    
    val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)
    print(f"\nValidation Results:")
    print(f"Loss: {val_loss:.4f}")
    print(f"Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    # prediksi untuk confusion matrix
    print("\nGenerating predictions...")
    y_val_pred = model.predict(val_dataset, verbose=0)
    y_val_pred = np.argmax(y_val_pred, axis=1)
    
    # class names
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    
    # classification report
    print("\n" + "-"*70)
    print("Classification Report (Validation):")
    print("-"*70)
    report = classification_report(y_val, y_val_pred, target_names=class_names, digits=4)
    print(report)
    
    # confusion matrix
    plot_confusion_matrix(y_val, y_val_pred, class_names, 'evaluation/confusion_matrix_val.png')
    
    # ========== EVALUATION ON TEST SET ==========
    
    if os.path.exists(test_dir):
        print("\n" + "="*70)
        print("STEP 7: EVALUATION (TEST SET)")
        print("="*70)
        
        X_test, y_test, _ = load_data(test_dir, img_size=(224, 224))
        
        if len(X_test) > 0:
            # preprocessing
            X_test = tf.keras.applications.mobilenet_v2.preprocess_input(X_test)
            
            # dataset
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
            
            # evaluate
            test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
            print(f"\nTest Results:")
            print(f"Loss: {test_loss:.4f}")
            print(f"Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            
            # prediksi
            print("\nGenerating predictions...")
            y_test_pred = model.predict(test_dataset, verbose=0)
            y_test_pred = np.argmax(y_test_pred, axis=1)
            
            # classification report
            print("\n" + "-"*70)
            print("Classification Report (Test):")
            print("-"*70)
            report = classification_report(y_test, y_test_pred, target_names=class_names, digits=4)
            print(report)
            
            # confusion matrix
            plot_confusion_matrix(y_test, y_test_pred, class_names, 'evaluation/confusion_matrix_test.png')
    
    # ========== SUMMARY ==========
    
    print("\n" + "="*70)
    print("TRAINING & EVALUATION COMPLETE!")
    print("="*70)
    
    print("\nFinal Accuracy:")
    print(f"Validation: {val_accuracy*100:.2f}%")
    if os.path.exists(test_dir) and len(X_test) > 0:
        print(f"Test: {test_accuracy*100:.2f}%")