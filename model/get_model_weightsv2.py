import tensorflow as tf
from pathlib import Path
from os.path import join

def get_model_weights(model) -> list:
    # Create an empty list to hold the layers' data
    layers_data = []

    # Iterate through the layers of the model
    for layer in model.layers:
        # Only process layers with weights
        if len(layer.get_weights()) > 0:
            # Get layer details
            layer_name = layer.name
            input_shape = list(layer.input_shape)
            output_shape = list(layer.output_shape)
            num_neurons = layer.output_shape[-1]

            # Get weights and biases
            weights, biases = layer.get_weights()

            # Debugging information
            print(f'Layer: {layer_name}, Weights Shape: {weights.shape}')

            # Create a list to hold the weights for each neuron
            neurons_data = []
            for i in range(num_neurons):
                # Adjust the indexing logic based on the dimensions of the weights tensor
                if len(weights.shape) == 3:
                    neuron_weights = weights[:,:,i]
                elif len(weights.shape) == 2:
                    neuron_weights = weights[:,i]
                else:
                    print(f'Unhandled weights shape: {weights.shape} in layer {layer_name}')
                    continue  # Skip this neuron

                # Create a dictionary for the current neuron
                neuron_data = {
                    "bias": float(biases[i]),
                    "weights": neuron_weights.tolist()
                }

                # Append the neuron data to the list
                neurons_data.append(neuron_data)

            # Create a dictionary for the current layer
            layer_data = {
                "name": layer_name,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "num_neurons": num_neurons,
                "weights": neurons_data
            }

            # Append the layer data to the list
            layers_data.append(layer_data)

    return layers_data

if __name__ == '__main__':
    # Load the model
    model = tf.keras.models.load_model(join(Path(__file__).parent, 'cat_dog_detector_v1.h5'))

    # Get the weights
    weights = get_model_weights(model)

    # Save to a JSON file
    import json
    with open(join(Path(__file__).parent, 'weights.json'), 'w') as f:
        json.dump(weights, f)
