"""
Author: Alok Pandey
Git username: wildoctopus
License: MIT License
"""

import tensorflow as tf
import numpy as np

def apply_fine_tuning(model, fine_tune_at):
    """
    Applies fine-tuning to a TensorFlow model.

    Args:
        model (tf.keras.Model): The pre-trained model to apply fine-tuning to.
        fine_tune_at (int): The index of the layer from which fine-tuning should start.
                            Layers before this index will be frozen.

    Returns:
        tf.keras.Model: The modified model with fine-tuning applied.
    """
    # Check the number of trainable parameters before fine-tuning
    num_trainable_params_before = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print("Number of trainable parameters before fine-tuning:", num_trainable_params_before)

    # Apply fine-tuning
    for i, layer in enumerate(model.layers):
        if i < fine_tune_at:
            layer.trainable = False

        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Check the number of trainable parameters after fine-tuning
    num_trainable_params_after = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print("Number of trainable parameters after fine-tuning:", num_trainable_params_after)

    return model


# Load the pre-trained ResNet model
resnet = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)

# Check the number of trainable parameters in TensorFlow
num_trainable_params_tensorflow = np.sum([tf.keras.backend.count_params(w) for w in resnet.trainable_weights])
print("Number of trainable parameters in TensorFlow model:", num_trainable_params_tensorflow)


# Apply fine-tuning in TensorFlow
resnet = apply_fine_tuning(resnet, fine_tune_at=125)

# Check the number of trainable parameters in TensorFlow after fine-tuning
num_trainable_params_tensorflow = np.sum([tf.keras.backend.count_params(w) for w in resnet.trainable_weights])
print("Number of trainable parameters in TensorFlow model after fine-tuning:", num_trainable_params_tensorflow)
