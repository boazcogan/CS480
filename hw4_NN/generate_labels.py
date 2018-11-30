import numpy as np
import cv2
import glob

toy = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0]
])


toy2 = [
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
    [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7]

]


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
    symmetryTable = np.bitwise_xor(toy,verticalFlip)#toy^verticalFlip
    return Density(symmetryTable)


def GenerateLabel(toy):
    # Its a grayscale image so lets get rid of those pesky color channels
    density = Density(toy)
    normal = toy <= 128
    symmetry = Symmetry(normal)
    verticalCrossings = FindVerticalCrossings(normal)
    horizontalCrossings = FindHorizontalCrossings(normal)
    avgVertical = np.average(verticalCrossings)
    avgHorizontal = np.average(horizontalCrossings)
    maxVertical = np.max(verticalCrossings)
    maxHorizontal = np.max(horizontalCrossings)
    return [density,symmetry,avgHorizontal,maxHorizontal,avgVertical,maxVertical]


def load_example_image():
    img = cv2.imread('mnist_data/0/1.jpg')
    if len(np.shape(img)) > 2:
        img = np.average(img, axis=2)
    label = GenerateLabel(img)
    return img, label


def LoadAllImagesOfNumber(num):
    files = sorted(glob.glob('mnist_data/'+str(num)+'/*'))
    all_labels = []
    for elem in range(len(files)):
        inner_list = [str(files[elem])]
        img = cv2.imread(str(files[elem]))
        if len(np.shape(img)) > 2:
            img = np.average(img, axis=2)
        label = GenerateLabel(img)
        inner_list.append(label)
        all_labels.append(inner_list)
    return [img, all_labels]

def LoadAllLabelsOfNumber(num):
    files = sorted(glob.glob('mnist_data/'+str(num)+'/*'))
    all_labels = []
    for elem in range(len(files)):
        img = cv2.imread(str(files[elem]))
        if len(np.shape(img)) > 2:
            img = np.average(img, axis=2)
        label = GenerateLabel(img)
        all_labels.append(label)
    return all_labels

def Part3LoadAllFeaturesOfNumber(num):
    files = sorted(glob.glob('mnist_data/'+str(num)+'/*'))
    all_labels = []
    for elem in range(len(files)):
        img = cv2.imread(str(files[elem]))
        if len(np.shape(img)) > 2:
            img = np.average(img, axis=2)
        label = GeneratePart3Features(img)
        all_labels.append(label)
    return all_labels


def GeneratePart3Features(img):
    features = []
    for i in range(len(img)//4):
        for j in range(len(img)//4):
            avg = 0
            for k in range(4):
                for l in range(4):
                    avg+=img[i*4+k][j*4+l]
            avg/=16
            features.append(avg)
    return features

def GenJSON():
    all_data = []
    for i in range(10):
        all_data +=LoadAllImagesOfNumber(i)[1]
    outFile = open("AllLabels.json", "w+")
    outFile.write('{\n')
    for elem in all_data:
        outFile.write('\t"filename": "'+str(elem[0]) + '",\n')
        outFile.write('\t"features": [\n')
        outFile.write('\t\t"density": "' + str(elem[1][0]) + '",\n')
        outFile.write('\t\t"symmetry": "' + str(elem[1][1]) + '",\n')
        outFile.write('\t\t"avgHorizontal": "' + str(elem[1][2]) + '",\n')
        outFile.write('\t\t"maxHorizontal": "' + str(elem[1][3]) + '",\n')
        outFile.write('\t\t"avgVertical": "' + str(elem[1][4]) + '",\n')
        outFile.write('\t\t"maxVertical": "' + str(elem[1][5]) + '"\n')
        outFile.write('\t],\n')
    outFile.write('}')
    outFile.close()


def main():
    GenJSON()

if __name__ == "__main__":
    GeneratePart3Features(toy2)
    choice = input("Would you really like to re-generate the entire JSON object? Y/N")
    if choice == 'y':
        choice = input("Are you really sure? Y/N")
        if choice == 'y':
            GenJSON()
