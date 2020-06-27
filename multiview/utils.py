from .header import *
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist

def load_corpus(path):
    cutter = thulac.thulac(seg_only=True)
    with open(path) as f:
        data = f.read().split('\n\n')
        data = [i.replace('\n', '') for i in data if i.strip()]
    # tokenize
    print(f'[!] collect {len(data)} utterances')
    data_ = []
    for u in tqdm(data):
        u = [i[0] for i in cutter.cut(u)]
        u = ' '.join(u)
        data_.append(u)
    return data_

def load_stopwords(path):
    with open(path) as f:
        data = [i.strip() for i in f.readlines() if i.strip()]
    return data

def obtain_word_idf(corpus, sw=None):
    '''
    data = [
        '我 来到 北京 清华大学',
        '他 来到 了 网易 杭研 大厦',
        '小明 硕士 毕业 于 中国 科学院',
        '我 爱 北京 天安门'
    ]
    '''
    vectorizer = CountVectorizer(min_df=4, max_df=50000, stop_words=sw)
    data = vectorizer.fit_transform(corpus).toarray()
    idf_count = np.sum(data, axis=0) + 1
    whole_tokens = sum(idf_count)
    tf_count = idf_count / whole_tokens
    return vectorizer.get_feature_names(), idf_count, tf_count

def cal_distinct(corpus):
    bigram_finder = BigramCollocationFinder.from_words(corpus)
    dist = FreqDist(corpus)
    try:
        bi_diversity = len(bigram_finder.ngram_fd) / bigram_finder.N
        uni_diversity = len(dist) / len(corpus)
        return (bi_diversity + uni_diversity) / 2
    except:
        return 0.0

if __name__ == "__main__":
    data = load_corpus('data/zh50w/train_.txt')
