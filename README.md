# Image-Encoder-Challenge
SPRING CAMP RECRUITMENT TASK
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
# Load the dataset
# Assume images are stored in a folder named 'images'
# You may need to adjust this based on the actual structure of the dataset
image_paths = [...]  # List of file paths to images
images = [plt.imread(path) for path in image_paths]
images = np.array(images) / 255.0  # Normalize pixel values

# Flatten the images
input_shape = images[0].shape
images_flat = images.reshape(images.shape[0], -1)
# Define Autoencoder architecture
input_layer = Input(shape=(np.prod(input_shape),))
encoded = Dense(128, activation='relu')(input_layer)
decoded = Dense(np.prod(input_shape), activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder.fit(images_flat, images_flat, epochs=50, batch_size=32, shuffle=True)
# Encode and decode the images
encoded_images = autoencoder.predict(images_flat)

# Visualize original and decoded images
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))

for i in range(n):
    # Display original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(images[i].reshape(*input_shape))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(encoded_images[i].reshape(*input_shape))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
# Get the encoded representations of the images
encoder = Model(input_layer, encoded)
latent_space = encoder.predict(images_flat)

# Visualize latent space using scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(latent_space[:, 0], latent_space[:, 1], c=LabelEncoder().fit_transform(categories), cmap='viridis')
plt.title('Latent Space Visualization')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.colorbar()
plt.show()
