from header import *
from metrics import *
from utils import *

def cal_ir_metric(rest):
    # P@1
    p_1 = np.mean([precision_at_k(y_true, y_pred, k=1) for y_true, y_pred in rest])
    # R10@1, R10@2, R10@5
    recalls = [recall(y_true, y_pred, cutoffs=[1, 2, 5]) for y_true, y_pred in rest]
    r10_1 = np.mean([i[0][1] for i in recalls])
    r10_2 = np.mean([i[1][1] for i in recalls])
    r10_5 = np.mean([i[2][1] for i in recalls])
    # R2@1, need to fix the rest
    new_rest = []
    for l, p in deepcopy(rest):
        candidate = random.sample(p, 1)[0]
        while candidate == 0:
            candidate = random.sample(p, 1)[0]
        if p.index(candidate) > p.index(0):
            new_rest.append((l, [0, candidate]))
        else:
            new_rest.append((l, [candidate, 0]))
    recalls = [recall(y_true, y_pred, cutoffs=[1]) for y_true, y_pred in new_rest]
    r2_1 = np.mean([i[0][1] for i in recalls])
    # MAP and MRR
    y_true = [i[0] for i in rest]
    y_pred = [i[1] for i in rest]
    MAP = mean_avg_precision_at_k(y_true, y_pred)
    MRR = mean_reciprocal_rank(y_true, y_pred)
    # 
    p_1 = round(p_1, 4)
    r2_1 = round(r2_1, 4)
    r10_1 = round(r10_1, 4)
    r10_2 = round(r10_2, 4)
    r10_5 = round(r10_5, 4)
    MAP = round(MAP, 4)
    MRR = round(MRR, 4)
    return p_1, r2_1, r10_1, r10_2, r10_5, MAP, MRR

def cal_generative_metric(path, batch_size=16, lang='zh'):
    # read the generated rest
    with open(path) as f:
        data = f.read().split('\n\n')
        data = [i.split('\n') for i in data if i.strip()]
    # filter the prefix, collect the refs and tgts
    rest, refs, tgts = [], [], []
    for example in data:
        example = [i[5:].replace('[SEP]', '').replace('[CLS]', '') for i in example]
        rest.append(example)
        if lang == 'en':
            refs.append(example[1].split())
            tgts.append(example[2].split())
        else:
            # use the jieba to tokenize for chinese
            refs.append(list(jieba.cut(example[1])))
            tgts.append(list(jieba.cut(example[2])))
            # refs.append(list(example[1]))
            # tgts.append(list(example[2]))
    # performance: (bleu4, dist-1/2, average, extrema, greedy)
    # length
    r_max_l, r_min_l, r_avg_l = cal_length(refs)
    c_max_l, c_min_l, c_avg_l = cal_length(tgts)
    # BLEU-4
    b_refs, b_tgts = [' '.join(i) for i in refs], [' '.join(i) for i in tgts]
    bleu1, bleu2, bleu3, bleu4 = cal_BLEU(b_refs, b_tgts)
    # Dist-1/2
    candidates, references = [], []
    for t, r in zip(tgts, refs):
        candidates.extend(t)
        references.extend(r)
    dist1, dist2 = cal_Distinct(candidates)
    r_dist1, r_dist2 = cal_Distinct(references)
    # embedding-based (average, extrema, greedy), using the character instead of the word
    # word embeddings from: https://github.com/Embedding/Chinese-Word-Vectors
    if lang == 'zh':
        w2v = gensim.models.KeyedVectors.load_word2vec_format('data/chinese_w2v.txt', binary=False)
    else:
        w2v = gensim.models.KeyedVectors.load_word2vec_format('data/english_w2v.bin', binary=True)
        print(f'[!] load english word2vec by gensim; GoogleNews WordVector: data/vocab/english_w2v.bin')
    es, vs, gs = [], [], []
    for r, c in tqdm(list(zip(refs, tgts))):
        es.append(cal_embedding_average(r, c, w2v))
        vs.append(cal_vector_extrema(r, c, w2v))
        gs.append(cal_greedy_matching_matrix(r, c, w2v))
    average = np.mean(es)
    extrema = np.mean(vs)
    greedy = np.mean(gs)
    # round, 4
    bleu1, bleu2, bleu3, bleu4 = round(bleu1, 4), round(bleu2, 4), round(bleu3, 4), round(bleu4, 4)
    dist1, dist2, r_dist1, r_dist2 = round(dist1, 4), round(dist2, 4), round(r_dist1, 4), round(r_dist2, 4)
    average, extrema, greedy = round(average, 4), round(extrema, 4), round(greedy, 4)
    return (bleu1, bleu2, bleu3, bleu4), ((r_max_l, r_min_l, r_avg_l), (c_max_l, c_min_l, c_avg_l)), (dist1, dist2, r_dist1, r_dist2), (average, extrema, greedy)
