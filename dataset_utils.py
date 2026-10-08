import os

def get_flower_dataset_dir():
    base_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets', 'flower_photos')
    nested = os.path.join(base_dir, 'flower_photos')
    return nested if os.path.exists(nested) else base_dir
