from .header import *

'''
Fasttext model for short response classification
In root path:
    ```bash
    python -m multiview.topic --mode train
    python -m multiview.topic --mode predict 
    ```
'''

def train_model():
    topic_clf = ff.train_supervised('data/topic/train.txt')
    model = topic_clf.save_model('ckpt/fasttext/model.bin')

def predict_model(model, s):
    '''
    s is already tokenized by jieba
    model: topic_clf = ff.load_model('fasttext/model')
    '''
    label, prob = model.predict(s)
    label = label[0].replace('__label__', '')
    prob = prob[0]
    return label, prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str)
    args = vars(parser.parse_args())

    if args['mode'] == 'train':
        train_model()
    elif args['mode'] == 'predict':
        model = ff.load_model('ckpt/fasttext/model.bin')
        label, prob = predict_model(model, '我 喜欢 爱情 片')
        print(label, prob)
    else:
        raise Exception(f'[!] unknow mode {args["mode"]}')
