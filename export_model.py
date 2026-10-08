import tensorflow as tf

def save_model(model, filepath='flower_classifier.keras'):
    model.save(filepath)
    print(f'Saved trained model weights to: {filepath}')
