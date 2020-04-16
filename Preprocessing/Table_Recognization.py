import matplotlib.pyplot as plt
from itertools import chain
from PIL import Image
from tqdm import tqdm
import pytesseract
import pandas as pd
import numpy as np
import itertools
import glob
import json
import cv2
import os

def find_table_start_point(index_of_240):
    start_x, start_y = 0, 0

    i = 0
    max_threshold, min_threshold = 80, 40
    total_len = len(index_of_240[0])
    while i < total_len:
        sequence_len = 1
        while i + sequence_len < total_len:
            if index_of_240[1][i] + sequence_len != index_of_240[1][i+sequence_len]:
                break
            sequence_len += 1

        if  max_threshold >= sequence_len >= min_threshold:
            start_x = index_of_240[0][i]
            # start_y = index_of_240[1][i]
            break

        i += sequence_len

    index = index_of_240[0].tolist().index(start_x)
    while True:
        # if index_of_240[1][index] != 0: # 不是最上方的白線
        if  index_of_240[1][index] + 1 == index_of_240[1][ index + 1 ] and index_of_240[1][index] + 2 == index_of_240[1][ index + 2 ] and \
            index_of_240[1][index] + 3 == index_of_240[1][ index + 3 ] and index_of_240[1][index] + 4 == index_of_240[1][ index + 4 ] and \
            index_of_240[1][index] + 5 == index_of_240[1][ index + 5 ] and index_of_240[1][index] + 6 == index_of_240[1][ index + 6 ]:
            
            start_y = index_of_240[1][index]
            break
        index += 1

    return (start_x, start_y)



def get_subseq_pair(index_of_240, start_value, direction):
    start_end_pair = []
    all_index = None
    if direction == 'x':
        all_index = [ j for i, j in zip(index_of_240[0], index_of_240[1]) if i == start_value ]
    if direction == 'y':
        all_index = [ i for i, j in zip(index_of_240[0], index_of_240[1]) if j == start_value ]
    
    start = end = 0 # start index and end index
    total_len = len(all_index)
    i = 0
    while i < total_len:
        if ( (i+1) < total_len and (all_index[i] + 1) != all_index[i+1]) or (i+1 == total_len):
                end = i
                distance = all_index[end] - all_index[start]
                if distance > 7:
                    start_end_pair.append((all_index[start], all_index[end]))
                start = end = i+1
        i+=1

    return start_end_pair



def method_1_cross_product(point_row_list, point_col_list):
    return list(itertools.product(point_row_list, point_col_list))


def points_to_cell(point_row_list, point_col_list):
    row = 0
    x = [ int(i) for i in point_row_list ]
    y = [ int(i) for i in point_col_list ]
    
    total_row = len(point_row_list)
    table_cells = []
    while row < total_row:
        col = 0
        while col + 1 < 32:
            position = {
                "location": (col//2, row//2),
                "corner": [
                    (x[row]  , y[col]  ), # upper-left  
                    (x[row]  , y[col+1]), # upper-right
                    (x[row+1], y[col]), # lower-left
                    (x[row+1], y[col+1])  # lower-right
                ]
            }
            table_cells.append(position)
            col += 2
        row += 2
    return table_cells



def table_cell_pipeline(image_path):

    image = cv2.imread(file, 0)
    index_of_240 = np.where(image==240)
    start_point = find_table_start_point(index_of_240)
    

    point_column     = get_subseq_pair(index_of_240, start_point[0], 'x') # left -> right
    point_row     = get_subseq_pair(index_of_240, start_point[1], 'y') # top -> down    
    point_column_second = get_subseq_pair(index_of_240, point_row[1][0], 'x')

    point_column_list = [ i for i in list(chain(*point_column)) if i >= start_point[1] ]
    point_row_list = [ i for i in list(chain(*point_row)) if i >= start_point[0] ]
    point_column_second_list = [ i for i in list(chain(*point_column_second)) if i >= start_point[1] ]
    table_cells = points_to_cell(point_row_list, point_column_second_list)

    image = np.where(image==240, 255, 0)
    image_color = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    for cell in table_cells:
        for point in cell["corner"]:
             image_color[point] = [0, 0, 255]
    
    basename = os.path.basename(image_path)
    filename = os.path.splitext(basename)[0]
    with open("Information/table_info/%s.json" % filename, 'w', encoding='utf-8') as f:
        json.dump(table_cells, f, indent=4)

    cv2.imwrite("Information/Mark_Cell/%s.png" % filename, image_color)

def check_loss_cell():
    files = [i for i in glob.iglob("Information/table_info/*.json")]
    for file in files:
        info = json.load(open(file, 'r'))
        if len(info) != 512:
            print(file)


def statistic_cell_size():
    # file = "Information/table_info/0004359 033016 p.json"
    files = [i for i in glob.iglob("Information/table_info/*.json")]
    data = pd.DataFrame(columns=["id", "width", "height", "complete"])
    for file in files:
        infos = json.load(open(file, 'r'))
        widths = []
        heights = []
        for item in infos:
            width = item["corner"][1][1] - item["corner"][0][1]
            widths.append(width)

            height = item["corner"][2][0] - item["corner"][0][0]
            heights.append(height)

        width_freq = np.array(np.unique(widths, return_counts=True)).T
        height_freq = np.array(np.unique(heights, return_counts=True)).T
        complete = True if len(height_freq) == 1 else False

        item_info = {
            "id"    : os.path.basename(file)[:-5], # eliminate extension
            "width" : width_freq[0, 0],
            "height": height_freq[0, 0],
            "complete": complete
        }

        data = data.append(item_info, ignore_index=True)
    
    data.to_csv("cell_size.csv", sep=",", index= False)
    
def image_to_data(image_id):
    file = "Table/%s.png" % image_id
    pattern_dict_cal = json.load(open("Pattern_CAL.json", 'r'))
    pattern_dict_pd = json.load(open("Pattern_PD.json", 'r'))

    table_info   = json.load(open("Information/table_info/%s.json" % image_id, 'r'))
    
    # upper_back_CAL, upper_back_PD     = [[i, 3] for i in range(16)], [[i, 5] for i in range(16)]
    # upper_behind_PD, upper_behind_CAL = [[i, 10] for i in range(16)], [[i, 13] for i in range(16)]
    
    # lower_behind_CAL, lower_behind_PD = [[i, 18] for i in range(15, -1, -1)], [[i, 21] for i in range(15, -1, -1)]
    # lower_back_PD, lower_back_CAL     = [[i, 26] for i in range(15, -1, -1)], [[i, 28] for i in range(15, -1, -1)]
    
    image = cv2.imread(file, 0)
    image = np.where(image == 240, 255, 0)
    # image_color = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    tooth_info = { i:{
                        "front":{
                                    "CAL": 0,
                                    "PD" : 0
                                 },
                        "back"  :{
                                    "CAL": 0,
                                    "PD" : 0
                                 }
                    } 
                    for i in range(1, 33, 1)
                }

    
    lower_count = 31
    cal_upper_loc = check_swap_cal(image, table_info, pattern_dict_cal, "upper") # to do
    cal_lower_loc = check_swap_cal(image, table_info, pattern_dict_cal, "lower")
    

    for cell in table_info:
        if cell["location"][1] not in [3, 5, 10, cal_upper_loc, cal_lower_loc, 21, 26, 28]:
            continue

        origin = cell["corner"][0]
        width = cell["corner"][1][1] - cell["corner"][0][1]

        block_1 = image[origin[0]: origin[0]+14, origin[1]          : origin[1]+ width//3    ]
        block_2 = image[origin[0]: origin[0]+14, origin[1]+ width//3: origin[1]+ width//3 * 2]
        block_3 = image[origin[0]: origin[0]+14, origin[1]+ width//3 * 2: origin[1]+ width   ]

        first, second, third = 0, 0, 0

        if cell["location"][1] in [3, cal_upper_loc, cal_lower_loc, 28]:
            first  = search_pattern_dic(block_1, pattern_dict_cal, "CAL")
            second = search_pattern_dic(block_2, pattern_dict_cal, "CAL")
            third  = search_pattern_dic(block_3, pattern_dict_cal, "CAL")
        
        else:
            first  = search_pattern_dic(block_1, pattern_dict_pd,  "PD")
            second = search_pattern_dic(block_2, pattern_dict_pd,  "PD")
            third  = search_pattern_dic(block_3, pattern_dict_pd,  "PD")
        

        if cell["location"][1] == 3: # upper, back, cal
            tooth_num = cell["location"][0] + 1
            tooth_info[tooth_num]["back"]["CAL"] = [first, second, third]

        if cell["location"][1] == 5: # upper, back, pd
            tooth_num = cell["location"][0] + 1
            tooth_info[tooth_num]["back"]["PD"] = [first, second, third]

        if cell["location"][1] == 10: # upper, front, pd
            tooth_num = cell["location"][0] + 1
            tooth_info[tooth_num]["front"]["PD"] = [first, second, third]

        if cell["location"][1] == cal_upper_loc: # upper, front, cal
            tooth_num = cell["location"][0] + 1
            tooth_info[tooth_num]["front"]["CAL"] = [first, second, third]

        if cell["location"][1] == cal_lower_loc: # lower, front, cal
            tooth_num = cell["location"][0] + 1 + lower_count
            tooth_info[tooth_num]["front"]["CAL"] = [first, second, third]
            lower_count = (lower_count - 2) % 32

        if cell["location"][1] == 21: # lower, front, pd
            tooth_num = cell["location"][0] + 1 + lower_count
            tooth_info[tooth_num]["front"]["PD"] = [first, second, third]
            lower_count = (lower_count - 2) % 32
            
        if cell["location"][1] == 26: # lower, back, pd
            tooth_num = cell["location"][0] + 1 + lower_count
            tooth_info[tooth_num]["back"]["PD"] = [first, second, third]
            lower_count = (lower_count - 2) % 32

        if cell["location"][1] == 28: # lower, back, cal
            tooth_num = cell["location"][0] + 1 + lower_count
            tooth_info[tooth_num]["back"]["CAL"] = [first, second, third]
            lower_count = (lower_count - 2) % 32


    with open("Information/Tooth_info/%s.json" % image_id, 'w', encoding='utf-8') as f:
        json.dump(tooth_info, f, indent=4)


def search_pattern_dic(array, pattern_dict, mode):
    white_matrix = np.full(array.shape, 255)
    if np.array_equal(array, white_matrix):
        return -99

    else:
        unique, counts = np.unique(array, return_counts=True)
        freq = np.asarray((unique, counts)).T
        black_count = str(int(freq[0][1]))
        if int(freq[0][1]) > 50:
            loc = np.where(array == 0)
            y = np.count_nonzero(loc[0] == loc[0][-1])
            if y == 2:
                return 10
            if y == 6:
                return 11
            if y == 9:
                return 12

        elif black_count in pattern_dict.keys():
            if pattern_dict[black_count] not in [2, 3, 4]:
                return pattern_dict[black_count]

            else: 
                loc = np.where(array == 0)
                y = np.count_nonzero(loc[0] == loc[0][-1])
                if y == 2:
                    return 4
                if y == 6:
                    return 2
                if y == 5:
                    return 3

                  
def check_swap_cal(image, table_info, pattern_dict, direction):
    
    loc = 13 if direction == "upper" else 18
    
    count = 0
    for cell in table_info:
        if cell["location"][1] == loc:
            origin = cell["corner"][0]
            width = cell["corner"][1][1] - cell["corner"][0][1]

            block_1 = image[origin[0]: origin[0]+14, origin[1]          : origin[1]+ width//3    ]
            block_2 = image[origin[0]: origin[0]+14, origin[1]+ width//3: origin[1]+ width//3 * 2]
            block_3 = image[origin[0]: origin[0]+14, origin[1]+ width//3 * 2: origin[1]+ width   ]

            first  = search_pattern_dic(block_1, pattern_dict, "CAL")
            second = search_pattern_dic(block_2, pattern_dict, "CAL")
            third  = search_pattern_dic(block_3, pattern_dict, "CAL")

            if first == -99: 
                count += 1
            if second == -99: 
                count += 1
            if third == -99: 
                count += 1

        if count > 30:
            if direction == "upper":
                loc -= 1
            else: 
                loc += 1
            break
    
    return loc

    


if __name__ == "__main__":

    # file = "Table/004151 101909 p.png"
    # file = "Table/000408 111819 p.png"
    # file = "Table/0004359 033016 p.png"
    # file = "Table/0004359 033016 p.png"
    
    files = [i for i in glob.iglob("Table/*.png")]
    # files += [i for i in glob.iglob("Table/*.jpg")]

    for file in files:
        print("Process_ID = %s" % file)
        base=os.path.basename(file)
        filename = os.path.splitext(base)[0]
        Target = "Information/Tooth_info/%s.json"
        if os.path.exists(Target) == True:
            continue
        table_cell_pipeline(file)

    check_loss_cell()
    statistic_cell_size()

    mapping = pd.read_csv("Mapping.csv", sep=",")
    data = pd.read_csv("cell_size.csv", sep=",")

    result = pd.merge(mapping, data, how='inner', on=['id'])

    normal_table = result[(result.width == 61) & (result.height == 13)]
    normal_directory = list(normal_table.Directory)
    normal_ids = list(normal_table.id)
    with open("normal_directory", "w") as f:
        f.write("\n".join(normal_directory))

    # for image_id in tqdm(normal_ids):
    #     image_to_data(image_id)
    
    