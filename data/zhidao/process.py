import json
import csv
import ipdb
from tqdm import tqdm

def check(tags):
    key = {'movie': set(['电影', '动漫', '电视', '百度云', '影视', '电影资源', '明星', '动画', '百度云资源', '游戏', '导演', '芭蕾', '华人明星', '表演', '娱乐', '电视剧', '影视评论', '动画片', '经典电影', '鬼吹灯', '钢铁侠', '蜘蛛侠', '韩国电影', '美国电影']),
           'electric': set(['麦克风', '数码', '耳机', '手机', '照相机', '单反', '充电宝', '索尼', 'OPPO', '华为', '荣耀', '小米']),
           'food': set(['食物', '美食', '营养', '寿司', '冰箱', '烹饪', '酒水饮料', '白酒', '蛋糕', '饼干', '烧烤', '火锅', '厨具', '水果', '甜品', '饮食', '烘焙', '料理', '日本料理']),
           'sport': set(['运动', '健身', '体育运动', '运动锻炼', '户外运动', '篮球', '足球', 'CBA', '网球', '减肥', '塑身', '健康', '乒乓球', '球类运动', '跑鞋']),
           'music': set(['电吉他', '吉他', '歌词', '欧美歌曲', '小提琴', '架子鼓', '鼓', '民乐', '明星', '简谱', '乐器', '口琴', '网易云音乐', '乐理', '民谣', 'itunes', '表演', '作曲', '钢琴', '歌词', 'MP3', '音乐', '唱歌', '声乐'])
    }
    for i in tags:
        for k, v in key.items():
            if i in v:
                return True, k
    return False, None

def filter_(msg):
    if len(msg) < 3:
        return False
    if 'http' in msg:
        return False
    return True

def read_file():
    idx, widx = 0, 0
    wf = open('train.txt', 'w')
    wf = csv.writer(wf, delimiter='\t')
    with open('zhidao_qa.json') as f:
        for line in tqdm(f.readlines()):
            try:
                data = json.loads(line)
            except:
                pass
            tags = data['tags']
            if_save, key = check(tags)
            if if_save:
                q, a = data['question'], data['answers']
                a = [i for i in a if len(i) < 100]
                if a:
                    wf.writerow((q, a, key))
                    widx += 1
            idx += 1
            print(f'idx: {idx}; size: {widx}', end='\r')

def process_4_dialog():
    wf = open('train.txt', 'w')
    with open('filter_zhidao.txt') as f:
        f = csv.reader(f, delimiter='\t')
        for item in tqdm(f):
            q, a, key = item
            a = eval(a)    # list
            for i in a:
                if filter_(i):
                    wf.write(f'{q}\n')
                    wf.write(f'{i}\n\n')

if __name__ == "__main__":
    process_4_dialog()
