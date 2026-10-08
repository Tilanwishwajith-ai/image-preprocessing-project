import numpy as np

FLOWER_CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def get_predicted_class(probabilities):
    return FLOWER_CLASSES[np.argmax(probabilities)]
