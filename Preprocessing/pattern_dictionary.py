from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import cv2
import os 

def copy_table_file():
    srcs = [ i for i in glob.iglob("Data/*/*.png") ]
    srcs += [ i for i in glob.iglob("Data/*/*.jpg") ]
    dsts = [ i.replace(os.path.dirname(i), "Table") for i in glob.iglob("Data/*/*.png") ]
    dsts += [ i.replace(os.path.dirname(i), "Table") for i in glob.iglob("Data/*/*.jpg") ]
    for src, dst in zip(srcs, dsts):
        copyfile(src, dst)

def crop_num():
    image = cv2.imread("Table/0001742 083012 p.png", 0)
    w, h = 10, 12
    location = [
        (606, 651), (798, 635), (284, 666), (714, 634), (778, 666), 
        (818, 534), (882, 534), (862, 487), (882, 487), (842, 487)
    ]

    for idx, loc in enumerate(location):
        x, y = loc
        crop_img = image[y : y+h, x : x+w]
        cv2.imwrite("Numbers/%d.png" % idx, crop_img)

def crop_num_PD():
    image = cv2.imread("Table/0001742 083012 p.png", 0)
    w, h = 15, 12

    location = [
        (651, 606), (634, 798), (634, 734), (634, 754), (634, 778), 
        (634, 200), (634, 176), (535, 842), (182, 92), (83, 175)
    ]

    for idx, loc in enumerate(location):
        x, y = loc
        crop_image = image[x : x+h, y : y+w]
        cv2.imwrite("Numbers/PD/%d.png" % idx, crop_image)

def crop_num_cal():
    image = cv2.imread("Table/02650 033117 p.png", 0)
    w, h = 15, 12

    location = [
        (286, 90), (754, 798), (754, 880), (754, 842), (754, 476), 
        (754, 606), (754, 412), (754, 946), (302, 732), (590, 925)
    ]

    # plt.imshow(image, cmap='gray')
    # plt.show()
    for idx, loc in enumerate(location):
        x, y = loc
        crop_image = image[x : x+h, y : y+w]
        cv2.imwrite("Numbers/CAL/%d.png" % idx, crop_image)



def num2dict():
    files = [i for i in glob.iglob("Numbers/CAL/*.png")]
    pattern_dict = {}
    for idx, file in enumerate(files):
        base = os.path.basename(file)
        name = os.path.splitext(base)[0]


        image = cv2.imread(file, 0)
        # pattern_dict[int(np.sum(image))] = int(name)
        image = np.where(image == 240 , 255, 0)
        unique, counts = np.unique(image, return_counts=True)
        freq = np.asarray((unique, counts)).T
        
        pattern_dict[int(freq[0][1])] = int(name)
    
    with open("Pattern_CAL.json", 'w') as f:
        json.dump(pattern_dict, f, indent=4)
    

def crop_tooth_num():
    image = cv2.imread("Table/0001742 083012 p.png", 0)
    # w, h = 10, 12
    # w, h = 20, 12

    
    tooth_location_upper = [ 
        (26, 133), (90, 133), (154, 133), (218, 133), (282, 133), (346, 133), (410, 133), (475, 133),  (540, 133),
        (604, 133), (666, 133), (730, 133), (794, 133), (858, 133), (922, 133), (986, 133)
    ] # 26 + 64 * n

    tooth_location_lower = [
        (985, 585), (921, 585), (857, 585), (793, 585), (729, 585), (665, 585), (601, 585), (537, 585),
        (471, 585), (407, 585), (343, 585), (279, 585), (215, 585), (151, 585), (87,  585), (23, 585)
    ] # 985 - 64 * n

    all_location = tooth_location_upper + tooth_location_lower
    
    for idx, loc in enumerate(all_location):
        w1, w2, h = 10, 20, 12
        x, y = loc
        crop_img = image[y : y+h, x : x+w1] if idx + 1 < 10 else image[y : y+h, x : x+w2]
        cv2.imwrite("Tooth_Numbers/%d.png" % ( idx + 1 ) , crop_img)

def crop_num_more_ten():
    image = cv2.imread("Table/02650 033117 p.png", 0)
    w, h = 15, 12
    location = [(752 ,302), (712, 138), (752, 138)]

    for idx, loc in enumerate(location):
        x, y = loc
        crop_img = image[y : y+h, x : x+w]
        cv2.imwrite("Numbers/%d.png" % (idx + 10), crop_img)


def tooth_num2dict():
    files = [i for i in glob.iglob("Tooth_Numbers/*.png")]
    pattern_dict = {}
    for file in files:
        image = cv2.imread(file, 0)
        base = os.path.basename(file)
        number = int(os.path.splitext(base)[0])
        total = np.sum(image)
        if total in pattern_dict.keys():
            print("Replace %d with %d" % (pattern_dict[total], number) )
        pattern_dict[int(total)] = number
    
    with open("Tooth_Num_Pattern.json", 'w') as f:
        json.dump(pattern_dict, f, indent=4)


def detect_crop(table_path):
    has_1, has_32 = False, False

    image = cv2.imread(table_path, 0)
    height, width = image.shape

    for row in range(height):
        for col in range(80):
            crop_img_1 = image[row : row + 12, col:col + 10]
            total_1 = int(np.sum(crop_img_1))
            if total_1 == 20754:
                has_1 = True

            crop_img_32 = image[row : row + 12, col:col + 20]
            total_32 = int(np.sum(crop_img_32))
            
            if total_32 == 38340:
                has_32 = True

    return has_1 and has_32

def mapping_dir2table():
    images = [ i for i in glob.iglob("Data/*/*.png") ]
    images += [ i for i in glob.iglob("Data/*/*.jpg") ]
    
    import pandas as pd
    mapping = pd.DataFrame(columns=["Directory", "id"])
    for img in images:
        dir_names = img[ img.find("\\")+1 : img.rfind("\\") ]
        base=os.path.basename(img)
        filename = os.path.splitext(base)[0]
        item_info = {
            "Directory"    : dir_names, # eliminate extension
            "id"           : filename
        }

        mapping = mapping.append(item_info, ignore_index=True)
    
    mapping.to_csv("Mapping.csv", sep=",", index= False)
    


if __name__ == "__main__":
    # num2dict()
    mapping_dir2table()