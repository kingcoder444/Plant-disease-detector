import kagglehub
import os
import tensorflow as tf


path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print("Path to dataset files:", path)

train_dir = os.path.join(path, "New Plant Diseases Dataset", "train")
val_dir = os.path.join(path, "New Plant Diseases Dataset", "valid")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.keras.utils.image_dataset_from_directory(
train_dir,
image_size=IMG_SIZE,
batch_size=BATCH_SIZE,
shuffle=True
)
val_ds = tf.keras.utils.image_dataset_from_directory(
val_dir,
image_size=IMG_SIZE,
batch_size=BATCH_SIZE,
shuffle=False
)


train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


data_augmentation = tf.keras.Sequential([
tf.keras.layers.RandomFlip("horizontal"),
tf.keras.layers.RandomRotation(0.1),
tf.keras.layers.RandomZoom(0.1),
])


normalization_layer = tf.keras.layers.Rescaling(1./255)


base_model = tf.keras.applications.EfficientNetB0(
input_shape=IMG_SIZE + (3,),
include_top=False,
weights="imagenet"
)
base_model.trainable = False  

model = tf.keras.Sequential[
data_augmentation,
normalization_layer,
base_model,
tf.keras.layers.GlobalAveragePooling2D(),
tf.keras.layers.Dropout(0.3),
tf.keras.layers.Dense(len(train_ds.class_names), activation="softmax")
]
model.compile(
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
loss=tf.keras.losses.SparseCategoricalCrossentropy(),
metrics=["accuracy"]
)


history = model.fit(
train_ds,
validation_data=val_ds,
epochs=10
)

base_model.trainable = True
model.compile(
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
loss=tf.keras.losses.SparseCategoricalCrossentropy(),
metrics=["accuracy"]
)
history_fine = model.fit(
train_ds,
validation_data=val_ds,
epochs=5
)

model.save("plant_disease_model.h5")

print(" Model training complete and saved as plant_disease_model.h5")
