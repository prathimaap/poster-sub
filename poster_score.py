import cv2
import numpy as np
import os
import csv

def compute_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def compute_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 1])

def compute_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    return contrast

def compute_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist /= hist.sum()
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
    return entropy

def score_poster(image_path):
    image = cv2.imread(image_path)

    sharpness = compute_sharpness(image)
    saturation = compute_saturation(image)
    contrast = compute_contrast(image)
    entropy = compute_entropy(image)

    sharpness_score = sharpness / 1000.0
    saturation_score = saturation / 256.0
    contrast_score = contrast / 128.0
    entropy_score = entropy / 8.0

    final_score = sharpness_score + saturation_score + contrast_score + entropy_score
    return final_score

if __name__ == "__main__":
    directory = '/Users/geetika/Desktop/Poster Submission/poster_downloads'
    results = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            file_path = os.path.join(directory, filename)
            imdb_score, imdb_id = filename[:-4].split('_')  # Strip out the ".jpg" and split by "_"
            poster_score = score_poster(file_path)
            results.append([imdb_id, imdb_score, poster_score])

    # Save results to CSV
    with open('poster_scores.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['IMDB_ID', 'IMDB_Score', 'Poster_Score'])  # header
        writer.writerows(results)
