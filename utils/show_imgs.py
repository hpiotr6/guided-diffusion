import argparse
import datetime
import numpy as np
from matplotlib import pyplot as plt
import os

parser = argparse.ArgumentParser(description='Show first image')
parser.add_argument('path', metavar='N', type=str)

args = parser.parse_args()

current_datetime = datetime.datetime.now()
folder_name = current_datetime.strftime(r"%m-%d_%H-%M-%S")

# Ścieżka do folderu
folder_path = os.path.join('results', folder_name)

# Tworzenie folderu
os.makedirs(folder_path)


with np.load(args.path) as data:
    img_vec = data['arr_0']

# img_arr = np.swapaxes()
img_arr = img_vec.squeeze()
# img_arr = np.swapaxes(img_vec.squeeze(),0,2)

for key in range(img_arr.shape[0]):
    # Odczytaj zdjęcie
    image = img_arr[key]

    # Wygeneruj ścieżkę do zapisu zdjęcia
    output_path = os.path.join(folder_path, f'{key}.png')

    # Zapisz zdjęcie
    plt.imsave(output_path, image)

    print(f'Zapisano zdjęcie {key} jako {output_path}')
