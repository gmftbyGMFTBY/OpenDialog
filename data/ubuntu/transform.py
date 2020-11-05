import csv
from tqdm import tqdm
import ipdb
import random

'''transformer the dataset format into the E-Commerence dataset format'''

def transform(path, writepath):
    with open(path) as f:
        csv_f = csv.reader(f)
        header = next(csv_f)    # ignore the header
        dataset = [line for line in csv_f]
        print(f'[!] collect {len(dataset)} samples for {args["mode"]} dataset')
    with open(writepath, 'w') as f:
        if args['mode'] == 'train':
            for line in tqdm(dataset):
                context, utterance, label = line
                label = int(float(label))
                context = context.replace(' __eou__ __eot__ ', '\t').replace(' __eou__ ', '\t').strip()
                utterance = utterance.replace('__eou__', '\t').strip()
                
                line = f'{label}\t{context}\t{utterance}\n'
                f.write(line)
        else:
            for line in tqdm(dataset):
                context, utterances = line[0], line[1:]
                context = context.replace(' __eou__ __eot__ ', '\t').replace(' __eou__ ', '\t').strip()
                utterances = [i.replace('__eou__', '\t').strip() for i in utterances]
                assert len(utterances) == 10
                for idx, utterance in enumerate(utterances):
                    label = 1 if idx == 0 else 0
                    f.write(f'{label}\t{context}\t{utterance}\n')
    print(f'[!] transform {args["mode"]} dataset over ...')

if __name__ == "__main__":
    args = {}
    args['mode'] = 'train'
    transform(f'{args["mode"]}.csv', f'{args["mode"]}.txt')
    
    args['mode'] = 'test'
    transform(f'{args["mode"]}.csv', f'{args["mode"]}.txt')
    
    args['mode'] = 'dev'
    transform(f'{args["mode"]}.csv', f'{args["mode"]}.txt')
