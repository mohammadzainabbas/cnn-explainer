import tensorflow as tf
from pathlib import Path
from os.path import join

def get_model_weights(model):


if __name__ == '__main__':
    # Load the model
    model = tf.keras.models.load_model(join(Path(__file__).parent, 'cat_dog_detector_v1.h5'))

    # Get the weights
    weights = model.get_weights()

