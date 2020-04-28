
from scipy.spatial import distance
from datetime import datetime
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

def get_quarter_points(points, direction=True):
        points = sorted(points ,key=lambda point:point[1])        
        up_lines   = sorted([points[0], points[1]] , key=lambda point:point[0])
        down_lines = sorted([points[2], points[3]] ,key=lambda point:point[0])

        left_coff, right_coff = 0, 0
        # 3/4 right
        if direction == True: 
                left_coff, right_coff = 0.75, 0.25
        # 3/4 left
        else: 
                left_coff, right_coff = 0.25, 0.75

        weighted = lambda x, y : int(x * left_coff + y * right_coff)

        up_quarter_point   = [ weighted(up_lines[0][0]  , up_lines[1][0]  ), weighted(up_lines[0][1]  , up_lines[1][1])   ]
        down_quarter_point = [ weighted(down_lines[0][0], down_lines[1][0]), weighted(down_lines[0][1], down_lines[1][1]) ]
        
        # drawcountor points should be ordered
        if direction == True:
                points = np.array([up_quarter_point, down_quarter_point,  down_lines[1], up_lines[1]])
        else:
                points = np.array([down_lines[0], up_lines[0], up_quarter_point, down_quarter_point])
        return points



def auto_rotate_image(image, points, quarter=False, near_right=True):
        
        rotate_angle = get_rotate_degree(image, points)
        
        if quarter == True:
                points = get_quarter_points(points, near_right)
        
        regular_point = points - points.min(axis=0)
        
        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        croped = image[y:y+h, x:x+w].copy() 
        
        mask = np.zeros(croped.shape, np.uint8)
        cv2.drawContours(mask, [regular_point], -1, (255, 255, 255), -1, cv2.LINE_AA)
        
        dst = cv2.bitwise_and(croped, croped, mask=mask)
        dst = rotate(dst, rotate_angle)
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
        return filename

def get_table_file(patient_id):
        table_infos = [ i for i in glob.iglob("Information/Tooth_info/*.json")]
        for table_name in table_infos:
                if patient_id in table_name:
                        pd_json   = json.load(open(table_name, "r"))
                        pd_table = pd_dict_2_pd_dataframe(pd_json)
                        return pd_table

def init_directory(directory): 
    if not os.path.exists(directory):
        os.makedirs(directory)

def padding(image, padding_height = 700, padding_width = 400):
        mask_size = (padding_height, padding_width)
        tooth_h, tooth_w = image.shape
        mask = np.zeros(mask_size)
        yoff = round((mask_size[0]-tooth_h)/2)
        xoff = round((mask_size[1]-tooth_w)/2)
        result = mask.copy()
        result[yoff:yoff+tooth_h, xoff:xoff+tooth_w] = image
        return result

def generate_data(output_dir, padding_size, quarter_crop=False, near_right=True):
        jsons = [ i for i in glob.iglob("Label/*/*/*.json")]
        images = [ i.replace("json", "png") for i in jsons]
        
        mapping_dict = {}
        no_table, no_side = set(), set()
        init_directory(output_dir)

        for data, image in tqdm(zip(jsons, images)):
                tooth_img  = cv2.imread(image, 0)
                tooth_data = json.load(open(data, "r"))["shapes"]
                patient_id = image.split("\\")[1]
                
                pd_table   = get_table_file(patient_id)
                
                if pd_table is None:
                        no_table.add(patient_id)
                        continue

                for tooth in tooth_data:
                        pd_pair  = get_PD_pair(pd_table, tooth["label"])
                        points   = np.array(tooth["points"]).astype(int)
                        ro_tooth = auto_rotate_image(tooth_img, points, quarter=quarter_crop ,near_right=near_right)
                        
                        tooth_h, tooth_w = ro_tooth.shape
                        if tooth_h > padding_size[0] or tooth_w > padding_size[1]:
                                continue
                        
                        if pd_pair == None:
                                no_side.add(data)
                                continue

                        ro_tooth = padding(ro_tooth, padding_size[0], padding_size[1])
                                            
                        filename = datetime.utcnow().isoformat(sep='-', timespec='milliseconds').replace(".", "-").replace(":", "-")[-12:]
                        number, side = tooth["label"].split("_")
                        
                        # left flip to right
                        if near_right == False:
                                ro_tooth = cv2.flip(ro_tooth, 1)
                        
                        if int(number) <= 16:
                                ro_tooth = cv2.flip(ro_tooth, 0)

                        if quarter_crop == True:
                                value = pd_pair[1] if near_right == True else pd_pair[0]
                                filename = '%s/%s_%s_%s_%d.png' % (output_dir, filename, image.split("\\")[2], number, value)
                                mapping_dict[filename] = value

                        else:
                                filename = '%s/%s_%s_%s.png' % (output_dir, filename, image.split("\\")[2], tooth["label"])   
                                mapping_dict[filename] = pd_pair
                        
                        cv2.imwrite(filename, ro_tooth)
                        
                        
                            
        json.dump(mapping_dict, open("%s/mapping.json" % output_dir, 'w'), indent=4)
        
        with open("%s/loss_table" % output_dir, "w") as f:
                no_table = list(no_table)
                no_table = "\n".join(no_table)
                f.write(no_table)
        
        with open("%s/loss_side" % output_dir, "w") as f:
                no_side = list(no_side)
                no_side = "\n".join(no_side)
                f.write(no_side)


if __name__ == "__main__":

        settings = [
                {
                        "output_dir"  : "Dataset/Tests/Images_quarter_R",
                        "padding_size": (700, 400),
                        "quarter_crop":True,
                        "near_right"  :True
                },

                {
                        "output_dir"  : "Dataset/Tests/Images_quarter_L",
                        "padding_size": (700, 400),
                        "quarter_crop":True,
                        "near_right"  :False
                },

                {
                        "output_dir"  : "Dataset/Tests/Images_normal",
                        "padding_size": (700, 400),
                        "quarter_crop":False,
                        "near_right"  :False
                }
        ]
        
        
        for setting in settings:
                generate_data(**setting)

