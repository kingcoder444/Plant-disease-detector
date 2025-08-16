import tensorflow as tf
from tensorflow.keras import layers, models

DATASET_DIR = r"D:\plants\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"


train_ds = tf.keras.utils.image_dataset_from_directory(
DATASET_DIR,
 batch_size=32,
   image_size=(224, 224),
 validation_split=0.2,
 subset="training",
 seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
DATASET_DIR,
batch_size=32,
image_size=(224, 224),
validation_split=0.2,
subset="validation",
seed=123
)


normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

model = models.Sequential([
layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
layers.MaxPooling2D(),
layers.Conv2D(64, 3, activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(128, 3, activation='relu'),
layers.MaxPooling2D(), layers.Flatten(),
layers.Dense(128, activation='relu'),
layers.Dense(38, activation='softmax') 
])

model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])


model.fit(train_ds, validation_data=val_ds, epochs=2)

model.save("plant_disease_model.h5")
