import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Resize images to 96x96
train_images_resized = tf.image.resize(train_images, [96, 96])
test_images_resized = tf.image.resize(test_images, [96, 96])

# Normalize pixel values to be between 0 and 1
train_images_resized, test_images_resized = train_images_resized / 255.0, test_images_resized / 255.0

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2
)
datagen.fit(train_images_resized)

# Load the MobileNetV2 model, excluding the top layers (the fully connected layers)
base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')

# Freeze the base model
base_model.trainable = True

for layer in base_model.layers[:50]:  # Example: freeze the first 50 layers
    layer.trainable = False

# Add new layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x) # Global average pooling reduces each feature map to a single value
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)  # Fully connected layer with 128 units
x = Dropout(0.5)(x)  # Added dropout for regularization
predictions = Dense(10, activation='softmax')(x)  # Output layer with 10 classes

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return float(lr * tf.math.exp(-0.1))  # Ensure the return value is a Python float


# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('cifar10_mobilenetv2_evenbetter.keras', save_best_only=True, monitor='val_accuracy')

# Train the model
model.fit(datagen.flow(train_images_resized, train_labels, batch_size=32),
          validation_data=(test_images_resized, test_labels),
          epochs=5,
          callbacks=[LearningRateScheduler(lr_scheduler), early_stopping, model_checkpoint])

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images_resized, test_labels, verbose=2)


# Print the test accuracy
print(f'Test accuracy: {test_acc * 100:.2f}%')

model.save('cifar10_mobilenetv2_best.keras')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

predictions = model.predict(test_images_resized)

# Plot the first 5 test images, their predicted labels, and the true labels
plt.figure(figsize=(10,10))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images_resized[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i][0]
    color = 'green' if predicted_label == true_label else 'red'
    plt.xlabel(f"{class_names[predicted_label]} ({class_names[true_label]})", color=color)
plt.show()


