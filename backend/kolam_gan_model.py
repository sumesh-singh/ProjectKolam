"""
Kolam Pattern Generation GAN Model

This module implements a Generative Adversarial Network for creating new kolam patterns
that capture the intricate details of traditional kolam designs.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KolamGAN:
    """
    GAN model for kolam pattern generation.
    """

    def __init__(self, img_height: int = 128, img_width: int = 128, channels: int = 3):
        """
        Initialize the GAN model.

        Args:
            img_height: Height of input images
            img_width: Width of input images
            channels: Number of image channels
        """
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.latent_dim = 100

        # Model components
        self.generator = None
        self.discriminator = None
        self.gan = None

        # Training configuration
        self.config = {
            'batch_size': 32,
            'epochs': 1000,
            'd_learning_rate': 0.0002,
            'g_learning_rate': 0.0002,
            'beta_1': 0.5,
            'save_interval': 100
        }

        # Training history
        self.d_losses = []
        self.g_losses = []

    def build_generator(self) -> keras.Model:
        """
        Build the generator model.

        Returns:
            Generator model
        """
        model = keras.Sequential([
            # Input layer
            layers.Dense(8 * 8 * 256, input_dim=self.latent_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((8, 8, 256)),

            # Upsampling block 1
            layers.Conv2DTranspose(
                128, (4, 4), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),

            # Upsampling block 2
            layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),

            # Upsampling block 3
            layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),

            # Output layer
            layers.Conv2DTranspose(self.channels, (4, 4), strides=(
                2, 2), padding='same', activation='tanh')
        ], name='generator')

        model.summary()
        self.generator = model
        return model

    def build_discriminator(self) -> keras.Model:
        """
        Build the discriminator model.

        Returns:
            Discriminator model
        """
        model = keras.Sequential([
            # Input layer
            layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=(
                self.img_height, self.img_width, self.channels)),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),

            # Convolutional block 1
            layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),

            # Convolutional block 2
            layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),

            # Convolutional block 3
            layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),

            # Global Average Pooling for consistent output size
            layers.GlobalAveragePooling2D(),
            layers.Dense(1, activation='sigmoid')
        ], name='discriminator')

        # Compile the discriminator
        optimizer = tf.keras.optimizers.Adam(
            self.config['d_learning_rate'], self.config['beta_1'])
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy', metrics=['accuracy'])

        model.summary()
        self.discriminator = model
        return model

    def build_gan(self):
        """
        Build the combined GAN model.
        """
        # Freeze discriminator weights when training generator
        self.discriminator.trainable = False

        # Create GAN model
        gan_input = layers.Input(shape=(self.latent_dim,))
        generated_img = self.generator(gan_input)
        gan_output = self.discriminator(generated_img)

        self.gan = keras.Model(gan_input, gan_output, name='gan')
        self.gan.compile(
            optimizer=keras.optimizers.Adam(
                self.config['g_learning_rate'], self.config['beta_1']),
            loss='binary_crossentropy'
        )

        # Unfreeze discriminator for separate training
        self.discriminator.trainable = True

    def load_and_preprocess_data(self, dataset_path: str) -> np.ndarray:
        """
        Load and preprocess the dataset for GAN training.

        Args:
            dataset_path: Path to the dataset directory

        Returns:
            Preprocessed image data
        """
        images = []

        logger.info(f"Loading images from {dataset_path}")

        # Walk through directory and load images
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        img_path = os.path.join(root, file)
                        img = tf.keras.preprocessing.image.load_img(
                            img_path,
                            target_size=(self.img_height, self.img_width)
                        )
                        img_array = tf.keras.preprocessing.image.img_to_array(
                            img)
                        # Normalize to [-1, 1] for tanh activation
                        img_array = (img_array.astype(
                            'float32') - 127.5) / 127.5
                        images.append(img_array)
                    except Exception as e:
                        logger.warning(f"Error loading image {file}: {str(e)}")
                        continue

        images = np.array(images)
        logger.info(f"Loaded {len(images)} images for GAN training")
        return images

    def train_step(self, real_images: np.ndarray) -> Tuple[float, float]:
        """
        Perform one training step.

        Args:
            real_images: Batch of real images

        Returns:
            Tuple of (discriminator_loss, generator_loss)
        """
        batch_size = real_images.shape[0]

        # Train discriminator
        # Generate fake images
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        fake_images = self.generator.predict(noise, verbose=0)

        # Labels for real and fake images
        real_labels = np.ones((batch_size, 1)) * 0.9  # Label smoothing
        fake_labels = np.zeros((batch_size, 1)) + 0.1

        # Train discriminator on real images
        d_loss_real = self.discriminator.train_on_batch(
            real_images, real_labels)

        # Train discriminator on fake images
        d_loss_fake = self.discriminator.train_on_batch(
            fake_images, fake_labels)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        valid_labels = np.ones((batch_size, 1))

        g_loss = self.gan.train_on_batch(noise, valid_labels)

        # Ensure losses are scalars
        d_loss_scalar = float(d_loss[0]) if hasattr(
            d_loss, '__len__') else float(d_loss)
        g_loss_scalar = float(g_loss[0]) if hasattr(
            g_loss, '__len__') else float(g_loss)

        return d_loss_scalar, g_loss_scalar

    def train(self, dataset_path: str, epochs: int = None):
        """
        Train the GAN model.

        Args:
            dataset_path: Path to the dataset
            epochs: Number of epochs to train
        """
        if epochs is not None:
            self.config['epochs'] = epochs

        # Load data
        images = self.load_and_preprocess_data(dataset_path)

        if len(images) == 0:
            logger.error("No images found for training")
            return

        # Build models
        self.build_generator()
        self.build_discriminator()
        self.build_gan()

        # Training loop
        for epoch in range(self.config['epochs']):
            # Select random batch
            idx = np.random.randint(
                0, images.shape[0], self.config['batch_size'])
            real_images = images[idx]

            # Train step
            d_loss, g_loss = self.train_step(real_images)

            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)

            # Log progress
            if epoch % 100 == 0:
                logger.info(
                    f"Epoch {epoch}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")

            # Save generated images
            if epoch % self.config['save_interval'] == 0:
                self.save_generated_images(epoch)

        # Save final models
        self.save_models()

    def save_generated_images(self, epoch: int, num_images: int = 16):
        """
        Save generated images at specified epoch.

        Args:
            epoch: Current epoch number
            num_images: Number of images to generate
        """
        os.makedirs('backend/generated_images', exist_ok=True)

        noise = np.random.normal(0, 1, (num_images, self.latent_dim))
        generated_images = self.generator.predict(noise, verbose=0)

        # Convert from [-1, 1] to [0, 1]
        generated_images = 0.5 * generated_images + 0.5

        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(num_images):
            row = i // 4
            col = i % 4
            axes[row, col].imshow(generated_images[i])
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(
            f'backend/generated_images/epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def save_models(self):
        """
        Save the trained models.
        """
        os.makedirs('backend/models', exist_ok=True)

        self.generator.save('backend/models/kolam_generator.h5')
        self.discriminator.save('backend/models/kolam_discriminator.h5')
        self.gan.save('backend/models/kolam_gan.h5')

        logger.info("Models saved successfully")

    def load_models(self):
        """
        Load pre-trained models.
        """
        try:
            self.generator = keras.models.load_model(
                'backend/models/kolam_generator.h5')
            self.discriminator = keras.models.load_model(
                'backend/models/kolam_discriminator.h5')
            self.build_gan()
            self.gan.load_weights('backend/models/kolam_gan.h5')
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")

    def generate_kolam(self, num_images: int = 1) -> np.ndarray:
        """
        Generate new kolam patterns.

        Args:
            num_images: Number of images to generate

        Returns:
            Generated images array
        """
        if self.generator is None:
            logger.error("Generator not initialized. Load models first.")
            return None

        noise = np.random.normal(0, 1, (num_images, self.latent_dim))
        generated_images = self.generator.predict(noise, verbose=0)

        # Convert from [-1, 1] to [0, 1]
        generated_images = 0.5 * generated_images + 0.5

        return generated_images

    def plot_training_history(self):
        """
        Plot the training losses.
        """
        plt.figure(figsize=(10, 5))

        plt.plot(self.d_losses, label='Discriminator Loss', alpha=0.7)
        plt.plot(self.g_losses, label='Generator Loss', alpha=0.7)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('backend/models/gan_training_history.png',
                    dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Main function to train the GAN model.
    """
    # Initialize GAN
    gan = KolamGAN(img_height=128, img_width=128)

    # Train the model
    dataset_path = 'backend/rangoli_dataset_complete/images'
    gan.train(dataset_path, epochs=500)  # Reduced epochs for demo

    # Plot training history
    gan.plot_training_history()

    # Generate sample images
    logger.info("Generating sample kolam patterns...")
    sample_images = gan.generate_kolam(16)

    if sample_images is not None:
        # Save sample images
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i in range(16):
            row = i // 4
            col = i % 4
            axes[row, col].imshow(sample_images[i])
            axes[row, col].axis('off')
            axes[row, col].set_title(f'Generated Kolam {i+1}')

        plt.tight_layout()
        plt.savefig('backend/generated_images/final_samples.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(
            "Sample images saved to backend/generated_images/final_samples.png")


if __name__ == "__main__":
    main()
