import numpy as np
import cv2

toy = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0]
])
toyHorizontalLabel = np.array([
    0,
    1,
    2,
    2,
    3,
    5
])
toyVerticalLabel = np.array([0, 1, 0, 1, 2, 3])



def FindHorizontalCrossings(toy):
    label = np.zeros((len(toy)))
    for i in range(len(toy)-1):
        current_col = toy[:,i]
        next_col = toy[:,i+1]
        truth = current_col!=next_col
        label+=truth
    return label

def FindVerticalCrossings(toy):
    label = np.zeros((len(toy)))
    for i in range(len(toy)-1):
        current_col = toy[i,:]
        next_col = toy[i+1,:]
        truth = current_col!=next_col
        label+=truth
    return label


def Density(toy):
    total = np.sum(np.sum(toy))
    avgVal = total/(len(toy)**2)
    return avgVal


def Symmetry(toy):
    verticalFlip = np.flip(toy, axis=0)
    symmetryTable = toy^verticalFlip
    return Density(symmetryTable)


def GenerateLabel(toy):
    # Its a grayscale image so lets get rid of those pesky color channels
    density = Density(toy)
    normal = toy > 0
    symmetry = Symmetry(normal)
    verticalCrossings = FindVerticalCrossings(normal)
    horizontalCrossings = FindHorizontalCrossings(normal)
    avgVertical = np.average(verticalCrossings)
    avgHorizontal = np.average(horizontalCrossings)
    sumVertical = np.sum(verticalCrossings)
    sumHorizontal = np.sum(horizontalCrossings)
    return [density,symmetry,avgHorizontal,sumHorizontal,avgVertical,sumVertical]


def load_image():
    img = cv2.imread('mnist_data/0/1.jpg')
    if len(np.shape(img)) > 2:
        img = np.average(img, axis=2)
    label = GenerateLabel(img)
    return img, label


load_image()