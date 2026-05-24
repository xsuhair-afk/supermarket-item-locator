import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class VGG16Extractor:
    def __init__(self):
        # Load VGG16 pre-trained on ImageNet
        base_model = VGG16(weights='imagenet', include_top=True)
        # Extract features from the fc2 layer (second fully connected layer)
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    def extract(self, img_bytes: bytes) -> np.ndarray:
        # Load the image from bytes
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        # Resize to 224x224 as required by VGG16
        img = img.resize((224, 224))
        # Convert to numpy array
        x = image.img_to_array(img)
        # Expand dimensions to match batch size
        x = np.expand_dims(x, axis=0)
        # Preprocess input
        x = preprocess_input(x)
        # Extract features
        features = self.model.predict(x, verbose=0)
        # Flatten and normalize
        features = features.flatten()
        features = features / np.linalg.norm(features) # L2 normalization
        return features
