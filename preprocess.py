from scipy import interpolate
import cv2
import os
import glob
import numpy as np
import pandas as pd


# 정사각형 추출 -> 300, 300 resizing 하는 함수
def preprocess_img(img_path):
    img = cv2.imread(img_path, 0)
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    ys = np.where(thresh == 0)[0]
    xs = np.where(thresh == 0)[1]
    xmin, xmax = np.min(xs)+5, np.max(xs)-5
    ymin, ymax = np.min(ys)+5, np.max(ys)-5
    square_img = img[ymin:ymax, xmin:xmax]
    square_img = cv2.resize(square_img, (300, 300))
    return square_img


# 개별 csv -> shape 채워서 하나의 np array로 변환
def csv_to_np(csv_filename, arange=300):
    label_category = csv_filename.split('_')[-1]
    try:
        label = pd.read_csv(csv_filename, skiprows=8, names=["x", "y", label_category])
    except:
        csv_filename = csv_filename.lower()
        label = pd.read_csv(csv_filename, skiprows=8, names=["x", "y", label_category])
    label["x"] = (label["x"] + 12) * arange / 24
    label["x"] = np.int32(label["x"].round())
    label["y"] = (label["y"] + 12) * arange / 24
    label["y"] = np.int32(label["y"].round())
    # 중복 행 제거 및 좌표화
    label = label.drop_duplicates(ignore_index=True, subset=['x', 'y'])
    xy_label = label.pivot("y", "x", label_category)
    coord = np.arange(arange)
    xy_label = xy_label.reindex(index=coord)
    xy_label = xy_label.T.reindex(index=coord).T
    return xy_label.values


def interpolate2d(np_array, arange=300):
    coord = np.arange(arange)
    xx, yy = np.meshgrid(coord, coord)
    masked_array = np.ma.masked_invalid(np_array)
    x1 = xx[~masked_array.mask]
    y1 = yy[~masked_array.mask]
    newarr = np_array[~masked_array.mask]
    interpolated_arr = interpolate.griddata((x1, y1), newarr, (xx, yy), method='nearest')
    return interpolated_arr


# 각 데이터들을 다 합친 이후에 적용하기
def standardize_data(np_array):
    standardized_data = (np_array - np_array.mean()) / np_array.std()
    return standardized_data


def csv_to_ylabel(csv_filename, arange=300):
    arr = csv_to_np(csv_filename, arange)
    arr = np.flip(arr, 0)  # 바뀐 y축 다시 전환
    arr = interpolate2d(arr, arange)
    arr = standardize_data(arr)
    return arr