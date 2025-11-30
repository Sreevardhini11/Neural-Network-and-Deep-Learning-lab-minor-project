# mnist_cnn.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import itertools

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load MNIST dataset
def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # normalize and expand dims
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)  # shape: (N,28,28,1)
    x_test = np.expand_dims(x_test, -1)
    return x_train, y_train, x_test, y_test

# 2. Build CNN model
def build_model(input_shape=(28,28,1), num_classes=10):
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 3. Plot helper
def plot_history(history, save_path=None):
    plt.figure(figsize=(12,4))
    # accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.legend()

    # loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# 4. Confusion matrix plotting
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save_path=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# 5. Train and evaluate
def train_and_evaluate(epochs=12, batch_size=128, model_dir='model_out'):
    x_train, y_train, x_test, y_test = load_data()

    # split validation from train
    val_split = 0.1
    val_count = int(len(x_train) * val_split)
    x_val = x_train[:val_count]
    y_val = y_train[:val_count]
    x_train2 = x_train[val_count:]
    y_train2 = y_train[val_count:]

    model = build_model(input_shape=(28,28,1), num_classes=10)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'best_model.h5'),
                                        save_best_only=True, monitor='val_accuracy'),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    ]
    os.makedirs(model_dir, exist_ok=True)

    history = model.fit(x_train2, y_train2,
                        validation_data=(x_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks)

    # save final model
    model.save(os.path.join(model_dir, 'final_model.h5'))

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

    # predictions & metrics
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    plot_history(history, save_path=os.path.join(model_dir, 'training_plot.png'))
    plot_confusion_matrix(cm, classes=list(range(10)), normalize=False,
                          title='Confusion matrix', save_path=os.path.join(model_dir, 'confusion_matrix.png'))

    # Save predictions csv (optional)
    df = pd.DataFrame({'true': y_test, 'pred': y_pred})
    df.to_csv(os.path.join(model_dir, 'predictions.csv'), index=False)

    return model, history, (x_test, y_test, y_pred)

# 6. Quick predict function (use saved model)
def load_model_and_predict(model_path, images):
    # images -> numpy array shape (N,28,28) or (N,28,28,1), values 0-255 or 0-1
    model = keras.models.load_model(model_path)
    imgs = np.array(images).astype('float32')
    if imgs.ndim == 3:
        imgs = np.expand_dims(imgs, -1)
    imgs /= 255.0
    preds = model.predict(imgs)
    return np.argmax(preds, axis=1), preds

# 7. Run training if script called directly
if __name__ == '__main__':
    model, history, test_info = train_and_evaluate(epochs=20, batch_size=128, model_dir='model_out')
    # show a few test images and predictions
    x_test, y_test, y_pred = test_info
    import random
    indices = random.sample(range(len(x_test)), 6)
    plt.figure(figsize=(12,4))
    for i, idx in enumerate(indices):
        plt.subplot(1,6,i+1)
        plt.imshow(x_test[idx].squeeze(), cmap='gray')
        plt.title(f"T:{y_test[idx]} P:{y_pred[idx]}")
        plt.axis('off')
    plt.show()
