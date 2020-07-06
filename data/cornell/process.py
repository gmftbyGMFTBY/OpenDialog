'''
Process the file to make the conversation
src-train.txt / tgt-train.txt
src-test.txt / tgt-test.txt
src-dev.txt / tgt-dev.txt
'''

import codecs
import ipdb

lines = {}

with codecs.open('movie_lines.txt', encoding='utf-8') as f:
    for line in f.readlines():
        num, char, _, _, utter = line.split('+++$+++')
        lines[num.strip()] = (char.strip(), utter.strip())

conversations = []
with codecs.open('movie_conversations.txt', encoding='utf-8') as f:
    for line in f.readlines():
        _, _, _, dialog = line.split('+++$+++')
        dialog = eval(dialog.strip())
        # [dialog_size, turn_size]
        conversations.append([lines[i] for i in dialog])

print(len(conversations), conversations[-1])

train_size, test_size, dev_size = 80000, 1500, len(conversations) - 81500

# create the file
def create_file(path, conversation):
    with open(path, 'w') as f:
        for dialog in conversation:
            for turn in dialog:
                user, utter = turn
                f.write(utter + '\n')
            f.write('\n')

# create_file('src-train.txt', 'tgt-train.txt', conversations[:train_size])
# create_file('src-test.txt', 'tgt-test.txt', conversations[train_size:train_size+test_size])
# create_file('src-dev.txt', 'tgt-dev.txt', conversations[train_size+test_size:])
create_file('train.txt', conversations)
