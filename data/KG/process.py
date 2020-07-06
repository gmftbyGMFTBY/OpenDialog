import os
from tqdm import tqdm
import csv
import ipdb
import pickle

'''
Process the 7Lore Knowledge Graph
In SMP-MCC, we only need the five domain: [movie, electric, food, sport, music]
'''

# key map for detecting the Object
object_key_map = {
        'music': ['音乐', '单曲', '流行', '华语', '饶舌', 'POP', '歌手', '琴', '吉他', '架子鼓'],
        'movie': ['电影', '电视剧', '演员', '电视'],
        'sport': ['体育', '足球', '篮球', '乒乓球', '羽毛球', '网球', '田径', '游泳', '武术', '跆拳道', '舞蹈', '健美', '教练', '电子竞技'],
        'electric': ['手机', '华为', '科技产品', '电脑', '笔记本', '电子产品'],
        'food': ['小吃', '美食', '食品', '烹饪', '饮品', '点心', '甜品', '菜品'],}
project_key_map = {
        'music': ['演唱', '作曲', '编曲', '专辑', '谱曲', '音乐风格'],
        'movie': ['演员', '编剧', '导演', '主演', '片长', '制片人'],
        'sport': ['体育', '所属体育队', '运动'],
        'electric': ['电池', 'CPU', '主屏', '屏幕', '操作系统'],
        'food': ['食材', '口味', '辅料'],}

def decide_flag(item):
    _, p, o = item
    # project
    for k, v in project_key_map.items():
        for v_ in v:
            if v_ in p:
                return k, True
    # object:
    for k, v in object_key_map.items():
        for v_ in v:
            if v_ in o:
                return k, True
    return None, False

def read_file():
    with open('/home/lt/data/data/KG/7Lore_triple.csv') as f:
        f = csv.reader(f, delimiter=',')
        cache, cache_flag, key = [], False, None
        filter_ = {'music': [], 'movie': [], 'food': [], 'electric': [], 'sport': []}
        for idx, line in tqdm(enumerate(f)):
            if idx == 0:
                print(line)
            else:
                # assign the flag:
                key_, cache_flag_ = decide_flag(line)
                if cache_flag_:
                    key, cache_flag = key_, cache_flag_
                if not cache:
                    cache.append(line)
                else:
                    if line[0] == cache[-1][0]:
                        cache.append(line)
                    else:
                        # save or not
                        if cache_flag:
                            filter_[key].append(cache)
                        cache = []
                        cache_flag = False
                        key = None
    for k in filter_.keys():
        print(f'[!] {k} collect {len(filter_[k])} SPO')
    return filter_

if __name__ == "__main__":
    data = read_file()
    with open('kg.pkl', 'wb') as f:
        pickle.dump(data, f)
