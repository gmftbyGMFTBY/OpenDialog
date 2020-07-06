import re
from tqdm import tqdm
from collections import Counter

ignore_pattern = ['艹', 'O', '*', '图', '贴', 'via', 'Via', '【', '】', '/', '_', ':', '3', '▽', '﹏']
pattern = ['alink', '——', '-', '…', '·', '\\', '""', '√', "'''", ':)', '="")', '：）', '^_^', '，，', '、', '=']
re_pattern = ['\(.*\)', '（.*\)', '\(.*）', '（.*）', '「.*」', '#.*#']

def filter_(msg):
    # repetitions over 3 words
    counter = Counter(msg)
    for key, value in counter.items():
        if value > 3:
            return None
    for i in ignore_pattern:
        if i in msg:
            return None
    # judge the english character
    for i in list('abcdefghijklmnopqrstuvwxyz'):
        if i in msg:
            return None
    for i in pattern:
        msg = msg.replace(i, '')
    for i in re_pattern:
        msg = re.sub(i, '', msg)
    if len(msg) < 3:
        return None
    if len(msg) > 150:
        return None
    return msg.strip() 
