from tqdm import tqdm
import random
import ipdb

'''
From: https://github.com/CSLujunyu/Improving-Contextual-Language-Modelsfor-Response-Retrieval-in-Multi-Turn-Conversation/blob/3ef5f0178495a96cb95a15a83f30c64f724c41aa/Utils/douban_evaluation.py#L40
It can be found that there are some bad cases in the test dataset (donot have the positive label in 10 samples), and they just ignore it. So in this file, we want to process the test dataset, and filte these bad cases.
'''

def transform():
    fw = open('test.txt', 'w')
    with open('test_.txt') as f:
        dataset = [i.strip() for i in f.readlines()]
        responses = [i.split('\t')[-1] for i in dataset]
    s = 0
    for idx in tqdm(range(0, len(dataset), 10)):
        item = dataset[idx:idx+10]
        sample, counter = [], 0
        for i in item:
            i = i.split('\t')
            response = i[-1]
            label, utterances = int(i[0]), '\t'.join(i[1:])
            counter += label
            sample.append((label, f'{label}\t{utterances}\n', response))
        if counter == 0:
            continue
        elif counter == 1:
            # >=1 positive is legal
            # sort the sample; make sure the positive samples is in front of all the negaitve samples
            sample = sorted(sample, key=lambda x:x[0], reverse=True)
            for _, string, _ in sample:
                fw.write(f'{string}')
            s += 1
        elif counter > 1:
            # select one positive and randomly select some as the negative samples to replace
            num = counter - 1
            sample = sorted(sample, key=lambda x:x[0], reverse=True)
            # select one positive
            candidates = list(set(responses) - set([i[-1] for i in sample]))
            sample_positive = sample[0]
            msg = '\t'.join(sample_positive[1].split('\t')[1:-1])
            sample = [sample_positive] + [i for i in sample[1:] if i[0] == 0]
            sample.extend([(0, f'0\t{msg}\t{i}\n', i) for i in random.sample(candidates, num)])
            for _, string, _ in sample:
                fw.write(f'{string}')
            s += 1

    print(f'[!] find {s} legal test session samples from {int(len(dataset)/10)} sessions')
    fw.close()

if __name__ == "__main__":
    transform()
