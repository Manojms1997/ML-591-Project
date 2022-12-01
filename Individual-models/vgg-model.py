import tensorflow as tf 
import numpy as np

input_shape = (224,224,3)
classes = 2
classifier_activation='softmax'
learning_rate = 0.0001
batch_size = 16
epochs = 80
DESTINATION = "D:\ASU-Notes\Fall-2022\ML\Project\DATA\OUTPUT_DIR"

train_image_path = DESTINATION + "/New_train_images.npy"   
train_labels_path = DESTINATION + "/New_train_labels.npy"   
test_image_path = DESTINATION + "/New_test_images.npy"
test_labels_path = DESTINATION + "/New_test_labels.npy"
x_train = np.load(train_image_path)
y_train = np.load(train_labels_path)
x_test = np.load(test_image_path)
y_test = np.load(test_labels_path)

base_model = tf.keras.applications.VGG19(weights = None, include_top = False, input_shape = input_shape)
x = base_model.output         
x = tf.keras.layers.GlobalAveragePooling2D()(x)
output = tf.keras.layers.Dense(classes, activation=classifier_activation)(x)

model = tf.keras.Model(inputs = base_model.input, outputs = output)

optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

model.compile(optimizer = optimizer,
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics = ['accuracy'])
    
results = model.fit(x_train, y_train, epochs = epochs,
                    validation_data = (x_test, y_test), 
                    batch_size=batch_size, 
                    callbacks = None
                    )

#losses = pd.DataFrame(model.history.history)
#losses[['loss','val_loss']].plot()

save_model = "D:\ASU-Notes\Fall-2022\ML\Project\Vgg_model.h5"
model.save(save_model)
