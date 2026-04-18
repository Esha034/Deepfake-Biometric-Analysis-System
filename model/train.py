import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import config

def get_data_generators():
    # Production-grade data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading training data from processed split...")
    train_data = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        classes=['real', 'fake']
    )

    print("Loading validation data...")
    val_data = val_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        classes=['real', 'fake']
    )
    
    return train_data, val_data

def build_efficientnet():
    # EfficientNetB0 is more robust for smaller datasets like CIPLAB
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*config.IMG_SIZE, 3)
    )

    # Initial Freeze
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(config.DROPOUT_RATE)(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(config.L2_REG))(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def train():
    if not os.path.exists(config.TRAIN_DIR) or len(os.listdir(config.TRAIN_DIR)) == 0:
        print("Error: No processed data found. Run data_loader.py first.")
        return

    train_data, val_data = get_data_generators()
    model, base_model = build_efficientnet()
    
    callbacks = [
        ModelCheckpoint(config.BEST_MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
    ]

    print("=== Phase 1: Warming up Classifier Head (AdamW) ===")
    # Using AdamW for better weight decay handling
    model.compile(
        optimizer=AdamW(learning_rate=config.LEARNING_RATE_1, weight_decay=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    history1 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=config.EPOCHS_PHASE_1,
        callbacks=callbacks
    )

    print("=== Phase 2: Fine-Tuning EfficientNetB0 Blocks ===")
    # Unfreeze the whole model but use a very low learning rate
    base_model.trainable = True
    
    model.compile(
        optimizer=AdamW(learning_rate=config.LEARNING_RATE_2, weight_decay=1e-5),
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
                  
    history2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=config.EPOCHS_PHASE_2,
        callbacks=callbacks
    )
    
    print(f"Strategy II Training Complete. Optimized model saved to {config.BEST_MODEL_PATH}")

if __name__ == '__main__':
    train()
