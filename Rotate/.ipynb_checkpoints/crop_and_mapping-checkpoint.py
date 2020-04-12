from scipy.spatial import distance
from math import cos, sin
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
        elif side == "L": 
                return (result[0], -99)
        elif side == "R":
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
        all_disance = { key: distance.euclidean( points[key[0]], points[key[1]] ) for key in combination(points) }
        all_disance = { k: v for k, v in sorted(all_disance.items(), key=lambda item: item[1]) }
        
        third_edge, fourth_edge = None, None

        for idx, item in enumerate(all_disance.items()):
                key, value = item
                if idx == 2:
                        third_edge = key
                if idx == 3:
                        fourth_edge = key

        left_line, right_line = None, None
        if points[third_edge[0]][0] < points[fourth_edge[0]][0] :
                left_line, right_line  = third_edge, fourth_edge
        else:
                right_line , left_line = third_edge, fourth_edge

        left_line =  [ points[ left_line [0] ], points[ left_line [1] ] ]
        right_line = [ points[ right_line[0] ], points[ right_line[1] ] ]

        left_line , left_direction = recognize_line(left_line)
        left_line_angle = get_line_angle(left_line, left_direction)

        right_line, right_direction = recognize_line(right_line)
        right_line_angle = get_line_angle(right_line, right_direction)

        rotate_angle = max( abs(left_line_angle), abs(right_line_angle) )
        if left_line_angle < 0 and right_line_angle < 0 :
                return -rotate_angle
        elif left_line_angle * right_line_angle < 0:
                return 0
        else:
                return rotate_angle
                
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


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape
    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h)) 
    return rotated


if __name__ == "__main__":
        
        jsons = [ i for i in glob.iglob("Data/005627/*/*.json")]
        images = [ i.replace("json", "png") for i in jsons]

        pd_json   = json.load(open("Information/Tooth_info/005627 012110 p.json", "r"))
        pd_table = pd_dict_2_pd_dataframe(pd_json)

        jsons = [ i for i in glob.iglob("Data/005627/*/*.json")]
        images = [ i.replace("json", "png") for i in jsons]

        pd_json   = json.load(open("Information/Tooth_info/005627 012110 p.json", "r"))
        pd_table = pd_dict_2_pd_dataframe(pd_json)
        mapping_dict = {}
        
        for data, image in zip(jsons, images):
                
                test_img  = cv2.imread(image, 0)
                test_data = json.load(open(data, "r"))["shapes"]
                
                for tooth in test_data:
                        pd_pair = get_PD_pair(pd_table, tooth["label"])
                        points = np.array(tooth["points"]).astype(int)
                        rotate_angle = get_rotate_degree(test_img, points)
                        
                        # rotate_angle_rad = -rotate_angle * math.pi / 180
                        # rotate_matrix = np.array([
                        #         [cos(rotate_angle_rad), -sin(rotate_angle_rad)],
                        #         [sin(rotate_angle_rad), cos(rotate_angle_rad)]
                        # ])

                        # ro_test_img = rotate(test_img, rotate_angle)
                        # points = np.dot(points, rotate_matrix).astype(int)
                        
                        rect = cv2.boundingRect(points)
                        x, y, w, h = rect
                        croped = test_img[y:y+h, x:x+w].copy() 
                        pts = points - points.min(axis=0)
                        mask = np.zeros(croped.shape, np.uint8)
                        
                        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
                        # mask = rotate(mask, rotate_angle)
                        croped = rotate(croped, rotate_angle)
                        dst = cv2.bitwise_and(croped, croped, mask=mask)
                        filename = '005627/%s' % tooth["label"]
                        
                        if os.path.exists("%s.png" % filename) == True:
                                count = 0
                                while True:
                                        count += 1
                                        filename = filename + "%d" % count
                                        if os.path.exists("%s.png" % filename) == False:
                                                break

                        filename = filename + ".png" 
                        cv2.imwrite(filename, dst)
                        mapping_dict[filename] = pd_pair

        json.dump(mapping_dict, open("005627/mapping.json", 'w'), indent=4)
        