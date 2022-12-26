import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup


def download_photo(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the URL of the photo
    photo_element = soup.find('meta', property='og:image')
    if photo_element:
        photo_url = photo_element['content']

        # Download the photo and save it to a file
        response = requests.get(photo_url)
        open('downloadedPhotos/photo.jpg', 'wb').write(response.content)
    else:
        print('Photo URL not found')


def prediciton():
    loaded_model = keras.models.load_model('downloadedPhotos/model.h5')
    image_path = 'downloadedPhotos/photo.jpg'

    image = keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    image = keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = loaded_model.predict(image)

    return prediction[0][0]
