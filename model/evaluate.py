import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import config

def evaluate_v2():
    print("Loading test data...")
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_data = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        classes=['real', 'fake'],
        shuffle=False
    )
    
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"Error: Model not found at {config.BEST_MODEL_PATH}. Train the model first.")
        return
        
    print(f"Loading Strategy II model from {config.BEST_MODEL_PATH}...")
    model = tf.keras.models.load_model(config.BEST_MODEL_PATH)
    
    print("\nRunning full evaluation...")
    predictions = model.predict(test_data)
    y_pred = (predictions > 0.5).astype(int).reshape(-1)
    y_true = test_data.classes
    
    # Global metrics
    report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'], output_dict=True)
    print("\nGlobal Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
    
    # --- Difficulty Breakdown (For Resume STAR Points) ---
    print("\n--- Performance Breakdown by Difficulty ---")
    filenames = test_data.filenames
    difficulties = ['easy', 'mid', 'hard']
    
    for diff in difficulties:
        diff_indices = [i for i, f in enumerate(filenames) if diff in f.lower() and y_true[i] == 1]
        if not diff_indices:
            continue
            
        diff_true = [y_true[i] for i in diff_indices]
        diff_pred = [y_pred[i] for i in diff_indices]
        
        # Accuracy on this difficulty (all are fake, so how many did we catch?)
        acc = np.mean(np.array(diff_true) == np.array(diff_pred)) * 100
        print(f"Detection Accuracy on [{diff.upper()}] fakes: {acc:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Strategy II: Deepfake Detection Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    cm_path = os.path.join(config.MODEL_SAVE_DIR, 'confusion_matrix_v2.png')
    plt.savefig(cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")
    
    # Save a summary of metrics for resume reference
    summary_path = os.path.join(config.MODEL_SAVE_DIR, 'resume_metrics.txt')
    with open(summary_path, 'w') as f:
        f.write("RESUME STAR METRICS\n")
        f.write("===================\n")
        f.write(f"Overall Accuracy: {report['accuracy']*100:.2f}%\n")
        f.write(f"Fake Detection Recall: {report['Fake']['recall']*100:.2f}%\n")
        f.write(f"F1-Score (Fake): {report['Fake']['f1-score']*100:.2f}%\n")
        # Log difficulty stats... (simplified for script)
    
    print(f"Resume metrics summary saved to {summary_path}")

if __name__ == '__main__':
    evaluate_v2()
