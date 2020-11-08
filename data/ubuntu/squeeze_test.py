import csv
import ipdb
import random

with open('test.txt') as f:
    csv_f = csv.reader(f, delimiter='\t')
    idx, cache, dataset = 0, [], []
    rawdataset = [i for i in csv_f]
    dataset = []
    for i in range(0, len(rawdataset), 10):
        dataset.append(rawdataset[i:i+10])

dataset = random.sample(dataset, 1000)

with open('test_.txt', 'w') as f:
    csv_f = csv.writer(f, delimiter='\t')
    for session in dataset:
        csv_f.writerows(session)
