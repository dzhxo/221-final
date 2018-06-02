import os
import sys
import collections

DIR_PATH = '../data'

STRESS = set(['str', 'nec', 'sur'])
NOT_STRESS = set(['int', 'sce', 'bre'])

def get_data(path):
    data_map = collections.defaultdict(list)
    for file_name in os.listdir(path):
        info = file_name.split('.')[0].split('_')
        num = int(info[0][5:])
        subject = int(info[1])
        scene = info[2]
        task = info[3]
        data_map[(subject, scene, task)].append((num, os.path.join(path, file_name), task in STRESS))

    for key in data_map:
        data_map[key].sort(key=lambda x:x[0])

    return data_map

