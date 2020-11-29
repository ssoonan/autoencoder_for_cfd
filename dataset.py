from sklearn.model_selection import train_test_split
import glob
import os
import cv2
import numpy as np
from preprocess import csv_to_ylabel


def get_label_base_names(label_dir="data"):
    labels = glob.glob("./{}/**/**.csv".format(label_dir), recursive=True)
    labels = [label for label in labels if not label.split(".csv")[-2].endswith("spf")]
    base_names = []
    for label in labels:
        splited = label.split("_")
        base_name = '_'.join(splited[:-1])
        base_names.append(base_name)
    base_names = set(base_names)
    base_names = list(base_names)
    base_names.sort(key=lambda x: int(x.split('/')[-2]))
    return base_names


def data_to_labels(label_dir="data", img_dir="image"):
    x_origins = []
    x_data = []
    y_data = []
    img_dir = os.path.join(label_dir, img_dir)
    base_names = get_label_base_names(label_dir)
    for base_name in base_names:
        only_basename = base_name.split('/')[-1]
        img_name = only_basename + "_image.jpg"
        try:
            x_origin = cv2.imread(os.path.join(img_dir, img_name), 0)
            _, x_label = cv2.threshold(x_origin, 200, 255, cv2.THRESH_BINARY)
            p_label = csv_to_ylabel(base_name + "_P.csv", 280)
            u_label = csv_to_ylabel(base_name + "_u.csv", 280)
            v_label = csv_to_ylabel(base_name + "_v.csv", 280)
            y_label = np.stack([p_label, u_label, v_label], axis=-1)
            x_origins.append(x_origin)
            x_data.append(x_label)
            y_data.append(y_label)
        except Exception as e:
            print(e)
            print(base_name)
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_train_origins, x_test_origins, x_train, x_test, y_train, y_test = train_test_split(x_origins, x_data, y_data, test_size=0.1)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return x_train_origins, x_test_origins, x_train, x_test, y_train, y_test