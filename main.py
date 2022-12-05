import numpy
from bs4 import *
import requests
import os
from newspaper import Article
from htmldate import find_date
import cv2
import numpy as np


# import cv2.cv2 as cv2

def folder_create(images):
    try:
        folder_name = input("Enter Folder Name:- ")
        # folder creation
        os.mkdir(folder_name)

    # if folder exists with that name, ask another name
    except:
        print("Folder Exist with that name!")
        folder_create()

    # image downloading start
    download_images(images, folder_name)


# DOWNLOAD ALL IMAGES FROM THAT URL
def download_images(images, folder_name):
    # initial count is zero
    count = 0

    # print total images found in URL
    print(f"Total {len(images)} Image Found!")

    # checking if images is not zero
    if len(images) != 0:
        for i, image in enumerate(images):
            # From image tag ,Fetch image Source URL

            # 1.data-srcset
            # 2.data-src
            # 3.data-fallback-src
            # 4.src

            # Here we will use exception handling

            # first we will search for "data-srcset" in img tag
            try:
                # In image tag ,searching for "data-srcset"
                image_link = image["data-srcset"]

            # then we will search for "data-src" in img
            # tag and so on..
            except:
                try:
                    # In image tag ,searching for "data-src"
                    image_link = image["data-src"]
                except:
                    try:
                        # In image tag ,searching for "data-fallback-src"
                        image_link = image["data-fallback-src"]
                    except:
                        try:
                            # In image tag ,searching for "src"
                            image_link = image["src"]

                        # if no Source URL found
                        except:
                            pass

            # After getting Image Source URL
            # We will try to get the content of image
            try:
                r = requests.get(image_link).content
                try:

                    # possibility of decode
                    r = str(r, 'utf-8')

                except UnicodeDecodeError:

                    # After checking above condition, Image Download start
                    with open(f"{folder_name}/images{i + 1}.jpg", "wb+") as f:
                        f.write(r)

                    # counting number of image downloaded
                    count += 1
            except:
                pass

        # There might be possible, that all
        # images not download
        # if all images download
        if count == len(images):
            print("All Images Downloaded!")

        # if all images not download
        else:
            print(f"Total {count} Images Downloaded Out of {len(images)}")


# MAIN FUNCTION START
def download_article(url):
    # content of URL
    r = requests.get(url)

    # Parse HTML Code
    soup = BeautifulSoup(r.text, 'html.parser')

    # find all images in URL
    images = soup.findAll('img')

    # Call folder create function
    folder_create(images)

    article = Article(url)
    article.download()
    article.parse()


# noinspection PyUnresolvedReferences
def compare_similarity(image1, image2):
    img1 = cv2.imread(image1, 0)
    img2 = cv2.imread(image2, 0)

    if img1.shape[0] * img1.shape[1] < img2.shape[0] * img2.shape[1]:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    else:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # --- take the absolute difference of the images ---
    res = cv2.absdiff(img1, img2)

    # --- convert the result to integer type ---
    res = res.astype(np.uint8)

    # --- find percentage difference based on number of pixels that are not zero ---
    percentage = (numpy.count_nonzero(res) * 100) / res.size
    print(percentage)


# url = input("Enter URL:- ")
#
# download_article(url)

# check date of the article
# print(find_date(url))

# Checking similarity of images
compare_similarity("tank.png", "dice.png")
