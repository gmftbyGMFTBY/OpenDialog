import json
import os
import ipdb
import re
from tqdm import tqdm

def filter(x):
    ignore_words = ['傻逼', '上图', '屎', '狗比', '尼玛', '图真好看', '屁事', '妈逼', '脑残', '意淫狗', '谢谢分享']
    for i in ignore_words:
        if i in x:
            return None
    if x.isalnum():
        return None
    x = x.replace('= =', '')
    x = x.replace('- -', '')
    x = x.replace('===', '')
    x = x.replace('QAQ', '')
    x = re.sub('。+', '。', x)
    x = re.sub('，+', '，', x)
    x = re.sub('\~+', '', x)
    x = re.sub('…', '', x)
    x = re.sub('～+', '～', x)
    x = re.sub('！+', '！', x)
    x = re.sub('\.+', '。', x)
    x = re.sub('\[.*\]', '', x)
    x = re.sub('\(.*\)', '', x)
    x = re.sub('，$', '', x)
    x = re.sub('\.$', '', x)
    x = re.sub(',$', '', x)
    x = re.sub('^。', '', x)
    x = re.sub('：？', '？', x)
    x = re.sub('？+', '？', x)
    x = re.sub('<img.*/>', '', x)
    x = re.sub('\n', '。', x)
    x = re.sub('1+', '1', x)
    x = re.sub('【.*】', '', x)
    x = gb2312(x)
    if len(x) < 3:
        return None
    return x

def gb2312(x):
    # x is a string that waited to filter
    cache = []
    for i in list(x):
        try:
            i.encode('gb2312')
            cache.append(i)
        except:
            pass
    x = ''.join(cache)
    return x.strip()

def read_file(topic):
    data = []
    for path in tqdm(os.listdir(topic)):
        path = f'{topic}/{path}'
        with open(path, encoding='utf-8') as f:
            item = json.load(f)
            title = item['title']
            content = item['topic']['content']
            if content:
                if title != content:
                    content = [title, content]
                else:
                    content = [content]
            else:
                content = [title]
            for r in item['replys']:
                r = r['content']
                # filter non GB2312
                c = filter('[SEP]'.join(content))
                r = filter(r)
                if c and r:
                    data.append((c, r))
    print(f'[!] collect {len(data)} dialogs from {topic}')
    return data

def write_file(data):
    with open('train.txt', 'w') as f:
        for item in data:
            c, r = item
            f.write(f'{c}\n')
            f.write(f'{r}\n\n')

if __name__ == "__main__":
    data = []
    for t in ['movie', 'music', 'sport', 'food', 'electric']:
        data.extend(read_file(t))
    print(f'[!] total {len(data)} dialogs')
    write_file(data)
