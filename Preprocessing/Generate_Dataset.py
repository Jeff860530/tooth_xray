
from scipy.spatial import distance
from math import cos, sin
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import glob
import math
import cv2
import os




def pd_dict_2_pd_dataframe(pd_json):
        table = pd.DataFrame(columns=["id", "left", "right"])

        for key in pd_json:
                front = pd_json[key]["front"]["CAL"]
                back  = pd_json[key]["back"]["CAL"]
                left  = min(front[0], back[0])
                right = min(front[2], back[2])
                table = table.append({
                        "id": key,
                        "left": left,
                        "right": right
                }, ignore_index=True)

        table.set_index("id" , inplace=True) 
        return table

def get_PD_pair(pd_table, label_name):
        
        number, side = label_name.split("_") 
        result = pd_table.loc[[number]].values[0]
        
        if side == "LR":
                return (result[0], result[1])
        elif side == "L" or side == "LH" or side == "HL": 
                return (result[0], -99)
        elif side == "R" or side == "RH" or side == "HR":
                return (-99, result[1])
        
def combination(points):
        num = len(points)
        i = 0
        while i < num:
                j = i + 1
                while j < num:
                        yield (i, j)           
                        j = j + 1
                i = i + 1



# def get_rotate_degree(image, points):
#         #print('points',points)
#         all_disance = { key: distance.euclidean( points[key[0]], points[key[1]] ) for key in combination(points) }
#         #print('all_disance',all_disance)
#         all_disance = { k: v for k, v in sorted(all_disance.items(), key=lambda item: item[1]) }
        
#         third_edge, fourth_edge = None, None
#         #print('all_disance',all_disance)
#         #print(type(all_disance))
#         #print('all_disance.items()',all_disance.items())
#         #print(type(all_disance.items()))
#         for idx, item in enumerate(all_disance.items()):  
#             key, value = item 
#             if idx == 2:
#                 third_edge = key
#                 #print(third_edge)
#             if idx == 3:
#                 fourth_edge = key
#                 #print(fourth_edge)
        
#         left_line, right_line = None, None
#         if points[third_edge[0]][0] < points[fourth_edge[0]][0] :
#                 left_line, right_line  = third_edge, fourth_edge
#         else:
#                 right_line , left_line = third_edge, fourth_edge

#         left_line =  [ points[ left_line [0] ], points[ left_line [1] ] ]
#         right_line = [ points[ right_line[0] ], points[ right_line[1] ] ]

#         left_line , left_direction = recognize_line(left_line)
#         #print('left_line , left_direction ',left_line,left_direction)###
#         left_line_angle = get_line_angle(left_line, left_direction)

#         right_line, right_direction = recognize_line(right_line)
#         right_line_angle = get_line_angle(right_line, right_direction)

#         rotate_angle = (abs(left_line_angle) + abs(right_line_angle)) / 2
        
# #         clock = lambda x: "clockwise" if x == True else "counterclockwise"
# #         print("left_angle: %d , %s" % (left_line_angle , clock(left_direction)) )
# #         print("right_angle %d , %s" % (right_line_angle, clock(right_direction)))
        
#         if left_line_angle < 0 and right_line_angle < 0 :
#                 return -rotate_angle
#         elif left_line_angle * right_line_angle < 0:
#                 return 0
#         else:
#                 return rotate_angle


def get_rotate_degree(image, points):
        points = sorted(points ,key=lambda point:point[1])
        up_center_point   = [(points[0][0]+points[1][0])/2,(points[0][1]+points[1][1])/2]
        down_center_point = [(points[2][0]+points[3][0])/2,(points[2][1]+points[3][1])/2]
        midline = [up_center_point,down_center_point]
        midline = np.array(midline)
        midline , midline_direction = recognize_line(midline)
        midline_angle = get_line_angle(midline, midline_direction)
        
        clock = lambda x: "clockwise" if x == True else "counterclockwise"
        #     print("Middle Angle: %d, %s" % (midline_angle, clock(midline_direction)) )
        return midline_angle


def get_line_angle(line, clockwise):
        vector = line[1] - line[0]
        length = np.linalg.norm(vector)
        cos    = np.dot(vector, [0, 1]) / length
        angle = math.acos(cos) *180 / math.pi
        return angle if clockwise == True else -angle

def recognize_line(line):
        sort_up_and_down = lambda line: [line[1], line[0]] if line[1][1] < line[0][1] else line
        check_clockwise  = lambda line: True if line[0][0] > line[1][0] else False    # up >>>> down
        line = sort_up_and_down(line)
        clockwise = check_clockwise(line)
        return line, clockwise

def rotate(image, angle):
        
        height, width = image.shape[:2] # image shape has 3 dimensions
        image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0,0]) 
        abs_sin = abs(rotation_mat[0,1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]
        
        # rotate image with the new bounds and translated rotation matrix
        rotated_image = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
        return rotated_image

def trim_border(image):
        #crop top
        if not np.sum(image[0]):
                return trim_border(image[1:])
        #crop bottom
        elif not np.sum(image[-1]):
                return trim_border(image[:-2])
        #crop left
        elif not np.sum(image[:,0]):
                return trim_border(image[:,1:]) 
        #crop right
        elif not np.sum(image[:,-1]):
                return trim_border(image[:,:-2])    
        return image

def auto_rotate_image(image, points):
        
        rotate_angle = get_rotate_degree(image, points)
        regular_point = points - points.min(axis=0)
        
        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        croped = image[y:y+h, x:x+w].copy() 
        ro_crop = rotate(croped, rotate_angle)
        
        mask = np.zeros(croped.shape, np.uint8)
        cv2.drawContours(mask, [regular_point], -1, (255, 255, 255), -1, cv2.LINE_AA)
        mask = rotate(mask, rotate_angle)
        dst = cv2.bitwise_and(ro_crop, ro_crop, mask=mask)
        dst = trim_border(dst)

        return dst


def append_redunt_name(filename):
        if os.path.exists("%s.png" % filename) == True:
                count = 0
                while True:
                        count += 1
                        filename = filename + "%d" % count
                        if os.path.exists("%s.png" % filename) == False:
                                break
        return filename + ".png" 

def get_table_file(patient_id):
        table_infos = [ i for i in glob.iglob("Information/Tooth_info/*.json")]
        for table_name in table_infos:
                if patient_id in table_name:
                        pd_json   = json.load(open(table_name, "r"))
                        pd_table = pd_dict_2_pd_dataframe(pd_json)
                        return pd_table
        
if __name__ == "__main__":
         
        jsons = [ i for i in glob.iglob("Label/*/*/*.json")]
        images = [ i.replace("json", "png") for i in jsons]
        
        mapping_dict = {}
        no_table = set()
        for data, image in zip(jsons, images):
                tooth_img  = cv2.imread(image, 0)
                tooth_data = json.load(open(data, "r"))["shapes"]
                patient_id = image.split("\\")[1]
                pd_table   = get_table_file(patient_id)
                
                if pd_table is None:
                        no_table.add(patient_id)
                        continue

                for tooth in tooth_data:
                        pd_pair = get_PD_pair(pd_table, tooth["label"])
                        points   = np.array(tooth["points"]).astype(int)
                        ro_tooth = auto_rotate_image(tooth_img, points)
                        filename = 'Dataset/Images/%s_%s' % (image.split("\\")[2], tooth["label"])                       
                        filename = append_redunt_name(filename)
                        cv2.imwrite(filename, ro_tooth)
                        
                        mapping_dict[filename] = pd_pair
                
        json.dump(mapping_dict, open("Dataset/mapping.json", 'w'), indent=4)
        
        with open("Dataset/loss_table", "w") as f:
                no_table = list(no_table)
                no_table = "\n".join(no_table)
                f.write(no_table)