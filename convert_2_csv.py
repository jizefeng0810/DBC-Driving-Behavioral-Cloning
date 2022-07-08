import csv
import math
import os
import numpy as np
import glob

# path = r'C:\carla_data\train\bc_ClearSunset16\\'
# csv_file = path + 'episode.csv'
# with open(csv_file, "w", newline='') as f:
#     writer = csv.writer(f)                  # 创建写的对象
#     # 先写入columns_name
#     writer.writerow(["step","x","y",'z','angle','time'])     # 写入列的名称
#
#     npy_data_list = []
#     path_file_number = len(glob.glob(os.path.join(path,'*'))) // 4
#     print(path_file_number)
#     for i in range(path_file_number):
#         npy_file = (path + 'measurements_%04d.npy'% i)
#         x, y, z, angle, time = np.load(npy_file)
#         angle_degree = math.degrees(angle)
#         angle = angle_degree if angle_degree >= 0 else angle_degree + 360
#         writer.writerow([i,x,y,z,angle,time])

path = r'C:\carla_avoid_data\train\\'
for home, dirs, files in os.walk(path):
    for dir in dirs:
        dir_path = path + '\\' + dir + '\\'
        csv_file = dir_path + 'episode.csv'
        with open(csv_file, "w", newline='') as f:
            writer = csv.writer(f)                  # 创建写的对象
            # 先写入columns_name
            writer.writerow(["step","x","y",'z','angle','time'])     # 写入列的名称

            npy_data_list = []
            path_file_number = len(glob.glob(os.path.join(dir_path, '*'))) // 4
            print(path_file_number)
            for i in range(path_file_number):
                npy_file = (dir_path + 'measurements_%04d.npy'% i)
                x, y, z, angle, time = np.load(npy_file)
                angle_degree = math.degrees(angle)
                angle = angle_degree if angle_degree >= 0 else angle_degree + 360
                writer.writerow([i,x,y,z,angle,time])