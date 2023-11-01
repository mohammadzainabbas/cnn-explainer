import tensorflow as tf
from pathlib import Path
from os.path import join

# Load the model
model = tf.keras.models.load_model(join(Path(__file__).parent, 'cat_dog_detector_v1.h5'))

# Get the weights
weights = model.get_weights()

# Convert to a serializable format
import numpy as np
serializable_weights = [w.tolist() for w in weights]

# Save to a JSON file
import json
with open(join(Path(__file__).parent, 'weights.json'), 'w') as f:
    json.dump(serializable_weights, f)
