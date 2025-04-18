
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (SeparableConv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, 
                                     GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom, RandomContrast)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# Enable mixed precision
#mixed_precision.set_global_policy('mixed_float16')

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Visualize sample images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([]), plt.yticks([]), plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i][0]])
plt.tight_layout()
plt.show()

# Create tf.data datasets
def preprocess_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.2)
    return image, label

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
    .shuffle(50000).map(preprocess_image).map(augment_image).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))\
    .map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)

# Build model
model = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomContrast(0.1),

    SeparableConv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax', dtype='float32')
])

# Compile model
initial_learning_rate = 0.001
lr_schedule = CosineDecay(initial_learning_rate, decay_steps=len(x_train) // 32 * 50)
optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint("best_model.keras", save_best_only=True, save_format='keras'),
    TensorBoard(log_dir="logs")
]

# Train model
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=test_dataset,
    callbacks=callbacks
)

# Evaluate model
best_model = tf.keras.models.load_model("best_model.keras")
loss, accuracy = best_model.evaluate(test_dataset, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()
