from tqdm import tqdm
import ipdb

'''
Filter the training samples that the target length are bigger than 20
'''

def read_file():
    with open('train.txt') as f:
        data = f.read().split('\n\n')
        data = [i.split('\n') for i in data if i.strip()]
        data = [i for i in data if len(i) == 2]
        data = [[c, r] for c, r in data if len(r) <= 20]
    return data

def write_file(data):
    with open('train_.txt', 'w') as f:
        for c, r in tqdm(data):
            f.write(f"{c}\n{r}\n\n")

if __name__ == "__main__":
    write_file(read_file())
