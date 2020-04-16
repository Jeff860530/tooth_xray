from time import time
from tqdm import tqdm
from collections import Counter

import numpy as np
import json
import glob
import cv2
import os

 # [dark, light]
pattern1 = ([0, 1, 2], [6, 7, 8])
pattern2 = ([1, 2, 5], [3, 6, 7])
pattern3 = ([2 ,5, 8], [0, 3 ,6])
pattern4 = ([5, 8, 7], [0, 1, 3])

def match_patterns(matrix, c_range, delta):
    min_gray, max_gray = c_range
    center = matrix[1, 1]
    # print(type(min_gray), type(max_gray), type(center))
    # if min_gray <= center <= max_gray is False: 
    #     return False
    
    if min_gray > center or center > max_gray: 
        return False

    # print("illigal central!: {}".format(center))
    elements = matrix.flatten()


    # 0 1 2
    # 3 4 5   ==> 0....8
    # 6 7 8
    
    # To Do:
    # [0, 1, 2], [6, 7, 8] 
    # [1, 2, 5], [3, 6, 7]
    # [2 ,5, 8], [0, 3 ,6]
    # [5, 8, 7], [0, 1, 3]
     
    # [dark, light]
    # pattern1 = ([1, 2, 5, 8], [0, 3, 6, 7])
    # pattern2 = ([2, 5, 8, 7], [0, 1, 3, 6])
    # pattern3 = ([0, 1, 2, 5], [3, 6, 7, 8])
    # pattern4 = ([0, 1, 2, 3], [5, 6, 7, 8])


    threshold = [center - delta, center + delta]
    cond1 = all( item < threshold[0] for item in elements[pattern1[0]] ) and all( item > threshold[1] for item in elements[pattern1[1]] )
    cond2 = all( item < threshold[0] for item in elements[pattern2[0]] ) and all( item > threshold[1] for item in elements[pattern2[1]] )

    cond3 = all( item < threshold[0] for item in elements[pattern1[1]] ) and all( item > threshold[1] for item in elements[pattern1[0]] )
    cond4 = all( item < threshold[0] for item in elements[pattern2[1]] ) and all( item > threshold[1] for item in elements[pattern2[0]] )

    cond5 = all( item < threshold[0] for item in elements[pattern3[0]] ) and all( item > threshold[1] for item in elements[pattern3[1]] )
    cond6 = all( item < threshold[0] for item in elements[pattern4[0]] ) and all( item > threshold[1] for item in elements[pattern4[1]] )

    cond7 = all( item < threshold[0] for item in elements[pattern3[1]] ) and all( item > threshold[1] for item in elements[pattern3[0]] )
    cond8 = all( item < threshold[0] for item in elements[pattern4[1]] ) and all( item > threshold[1] for item in elements[pattern4[0]] )

    return cond1 or cond2 or cond3 or cond4 or cond5 or cond6 or cond7 or cond8

def detection(image_gray):
    mark_points = []
    row, colume = image_gray.shape
    for i in tqdm(range(1, row-1)):
        for j in range(1, colume-1):
            box = image_gray[ i-1: i+2 , j-1: j+2]
            param = {
                'matrix'  : box,
                'c_range' : [10, 200],# [65 , 95],
                'delta'   : 6
            }
            if match_patterns(**param) is True:
                box = box.flatten()
                box_pixel = [ 
                    sum(box[pattern1[0]]), sum(box[pattern1[1]]), 
                    sum(box[pattern2[0]]), sum(box[pattern2[1]]),
                    sum(box[pattern3[0]]), sum(box[pattern3[1]]),
                    sum(box[pattern4[0]]), sum(box[pattern4[1]])
                ]
                most_dark = min (box_pixel)

                mark_points.append(
                    {
                        'value': most_dark,
                        'point': (j, i)
                    }
                )
    return mark_points
          

def init_directory(directory): 
    if not os.path.exists(directory):
        os.makedirs(directory)


def dataArrayGenerator(rootdir, replace_src, replace_dst, enhanced=False):
    pink, yellow = "Pink", "Yellow"
    target_dirs = [ i for i in glob.iglob(rootdir) if "." not in i ]
    result_dirs = []

    if enhanced == False:
        # result_dirs += [ i.replace(replace_src, "{}/Result_{}".format(replace_dst, pink))   for i in target_dirs ]
        result_dirs += [ i.replace(replace_src, "{}/Result_{}".format(replace_dst, yellow)) for i in target_dirs ]

    else:
        # result_dirs += [ i.replace(replace_src, "{}/Enhance_Result_{}".format(replace_dst, pink))   for i in target_dirs ]
        result_dirs += [ i.replace(replace_src, "{}/Enhance_Result_{}".format(replace_dst, yellow)) for i in target_dirs ]

    target_dirs += target_dirs # for zip cannot be half  
    return target_dirs, result_dirs

if __name__ == "__main__":
    result_root_dir = "Pattern_Detection_Yellow"

    params = [
        {
            "enhanced": False,   
            "rootdir" : "Data/*/*",
            "replace_src": "Data",
            "replace_dst": result_root_dir
        },
    
        # {
        #     "enhanced": True,   
        #     "rootdir" : "Enhance_Contrast/he/*/*",
        #     "replace_src": "Enhance_Contrast/he",
        #     "replace_dst": result_root_dir
        # },

        # {
        #     "enhanced": True,   
        #     "rootdir" : "Enhance_Contrast/dhe/*/*",
        #     "replace_src": "Enhance_Contrast",
        #     "replace_dst": result_root_dir
        # }
    ]

    target_dirs, result_dirs = [], []
    for param in params:
        targets, results = dataArrayGenerator(**param)
        target_dirs += targets
        result_dirs += results

    gray_scale = 255
    pink = [gray_scale, 0,  gray_scale]
    yellow = [0, gray_scale, gray_scale]

    for target_dir, result_dir in zip(target_dirs, result_dirs):
        if os.path.exists(result_dir) is False:
            init_directory(result_dir)

            all_image = [i for i in os.listdir(target_dir)]
            
            for image_file in all_image:
                filename = "{}/{}".format(target_dir, image_file)
                image = cv2.imread(filename)
                image_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) 
                points = detection(image_gray)

                for idx, point in enumerate(points):
                    if idx-1 >= 0 and idx + 1 < len(points):
                        draw = False
                        
                        # both left and right
                        if points[idx]['point'][0]-1 == points[idx-1]['point'][0] and points[idx]['point'][0]+1 == points[idx+1]['point'][0]:
                            continue

                        # either left or right
                        elif points[idx]['point'][0]-1 == points[idx-1]['point'][0]:
                            if points[idx]['value'] < points[idx-1]['value']:
                                draw = True 
                        
                        elif points[idx]['point'][0]+1 == points[idx+1]['point'][0]:
                            if points[idx]['value'] < points[idx+1]['value']:
                                draw = True

                        # neither left nor right
                        else:
                            draw = True

                        if draw == True:    
                            image[(point['point'][1], point['point'][0])] = pink if "Pink" in result_dir else yellow
                
                name = os.path.splitext(image_file)[0]
                cv2.imwrite('{}/{}.png'.format(result_dir, name), image)

