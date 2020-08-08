import numpy as np
# import math


def to_relevance_scores(y_true, y_pred):
    """
    Returns a list of relevance scores (binary), which can be used in various
    information retrieval metrics

    :param y_true: a list of relevant items (such as document ids)
    :param y_pred: a list of predicted items
    :return: a list of binary relevance scores, ex. [1, 0, 1, 1, 0, ...]
    """
    rel_scores = [0] * len(y_pred)
    for i, d in enumerate(y_pred):
        if d in y_true and d not in y_pred[:i]:
            rel_scores[i] = 1
    return rel_scores


def precision_at_k(y_true, y_pred, k=None):
    """
    The fraction of the documents retrieved that are relevant at position k;

    :param y_true: list
        A list of ground truth elements (order is not counted)
    :param y_pred: list
        A list of predicted elements (order does matter)
    :param k: int (optional)
        Number of results to consider
        If k is none, k is the length of the given list of relevance scores.
    :return: double
        The precision score over the input lists at position k
    """
    rel_scores = to_relevance_scores(y_true, y_pred)
    if k is None:
        k = len(rel_scores)
    rel_scores = np.asarray(rel_scores)[:k]
    return np.mean(rel_scores) if len(rel_scores) > 0 else 0


def recall(y_true, y_pred, cutoffs=None, rel_scores=None, min_rel_level=1):
    """
    The fraction of the relevant documents that are successfully retrieved
    measured at various cutoffs.

    :param y_true: list
        A list of ground truth elements (order is not counted)
    :param y_pred: list
        A list of predicted elements (order does matter)
    :param cutoffs: list (optional)
        A list of cutoff positions at where the recall is calculated
    :param rel_scores: list (optional)
        A list of relevance scores; If provided, this is used instead of
        binary relevance scores calculated from y_true and y_pred
    :param min_rel_level: int (optional)
        Relevance score level that is used to count items which can be
        considered as relevant
    :return: list
        A list of pairs such that (cutoff position, recall)
    """
    if rel_scores is None:
        rel_scores = to_relevance_scores(y_true, y_pred)
    if cutoffs is None:
        cutoffs = [1, 2, 5]
    recalls = [[c, 0.] for c in cutoffs]
    recall_count = 0
    cutoff_idx = 0
    for i, rel in enumerate(rel_scores):
        if i == cutoffs[cutoff_idx]:
            recalls[cutoff_idx][1] = recall_count / len(y_true)
            cutoff_idx += 1
            if cutoff_idx == len(cutoffs):
                break
        if rel >= min_rel_level:
            recall_count += 1
    # Assign recalls for the rest of the cutoffs if exist
    while cutoff_idx < len(cutoffs):
        recalls[cutoff_idx][1] = recall_count / len(y_true)
        cutoff_idx += 1
    return recalls


def f_measure(y_true, y_pred, beta=1):
    """
    The weighted harmonic mean of precision and recall.

    :param y_true: list
        A list of ground truth elements (order is not counted)
    :param y_pred: list
        A list of predicted elements (order does matter)
    :param beta: non-negative real number, balancing factor
    :return: (double, double, double)
        Three measures; precision, recall, and f-measure

    """
    p = precision_at_k(y_true, y_pred)
    recalls = recall(y_true, y_pred)
    r = recalls[-1][1]
    b2 = beta ** 2
    if p == 0 and r == 0:
        f = 0
    else:
        f = (1 + b2) * p * r / b2 * (p + r)
    return f


def avg_precision_at_k(y_true, y_pred, k=None):
    """
    Average Precision at k between two lists

    :param y_true: list
        A list of ground truth elements (order is not counted)
    :param y_pred: list
        A list of predicted elements (order does matter)
    :param k: int, optional
        The maximum number of predicted elements
    :return: double
        The average precision at k over the input lists
    """
    rel_scores = to_relevance_scores(y_true, y_pred)
    if k is None:
        k = len(rel_scores)
    rel_scores = np.asarray(rel_scores)[:k]

    out = [precision_at_k(y_true, y_pred, k + 1) for k in range(rel_scores.size) if rel_scores[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_avg_precision_at_k(y_true, y_pred, k=10):
    """
    Mean Average Precision at k between two lists

    :param y_true: list
        A list of lists of ground truth elements
    :param y_pred: list
        A list of lists of predicted elements
    :param k: int, optional
        The maximum number of predicted elements
    :return: double
        The mean average precision at k (MAP) over the input lists
    """
    return np.mean([avg_precision_at_k(t, p, k)
                    for t, p in zip(y_true, y_pred)])


def dcg(y_true, y_pred, rel_scores=None, k=None):
    """
    DCG (Discounted Cumulative Gain) -- measures the gain ("usefulness")
    based on the position of a document in the result list. Here, we are
    using the alternative formula which places stronger emphasis on retrieving
    relevant documents.
    description (https://goo.gl/5yPVU7)

    DCG_p = \sum_{i=1}^p \frac{2^{rel_i}-1}{\log_2(i+1)}
    """
    if rel_scores is None:
        rel_scores = to_relevance_scores(y_true, y_pred)
    if k is None:
        k = len(rel_scores)
    rel_scores = np.asarray(rel_scores)[:k]

    assert len(rel_scores) == len(y_pred)
    if len(rel_scores) > 0:
        return np.sum((np.exp2(rel_scores) - 1) /
                      np.log2(np.arange(2, rel_scores.size + 2)))
    else:
        return 0.


def ndcg(y_true, y_pred, rel_scores=None, k=None):
    if rel_scores is None:
        rel_scores = to_relevance_scores(y_true, y_pred)
    if k is None:
        k = len(rel_scores)
    rel_scores = np.asarray(rel_scores)[:k]

    if len(rel_scores) > 0:
        rel_scores_r = sorted(rel_scores, reverse=True)

        idcg = np.sum((np.exp2(rel_scores_r) - 1) / np.log2(np.arange(2, len(rel_scores_r) + 2)))

    return dcg(y_true, y_pred, rel_scores=None, k=None) / idcg


def reciprocal_rank(y_true, y_pred):
    '''
    Reciprocal rank between two list
    '''
    rank = 0
    for idx, i in enumerate(y_pred):
        if i in y_true:
            rank = idx
            break
    rank = 0 if rank == 0 else 1 / rank
    return rank

def mean_reciprocal_rank(y_true, y_pred):
    return np.mean([reciprocal_rank(t, p) for t, p in zip(y_true, y_pred)])



if __name__ == "__main__":
    # y_true = list('a')
    # y_pred = list('czbajepgard')
    y_true = list('a')
    y_pred = list('bacd')
    rel_scores = to_relevance_scores(y_true, y_pred)
    print(f'rel_scores: {rel_scores}')

    print(f'P@K: {precision_at_k(y_true, y_pred)}')
    print(f'recall: {recall(y_true, y_pred, cutoffs=[1])}')
    print(f'avg_prec at 1: {avg_precision_at_k(y_true, y_pred, k=1)}')
    print(f'avg_prec at 3: {avg_precision_at_k(y_true, y_pred, k=3)}')
    print(f'avg_prec at 5: {avg_precision_at_k(y_true, y_pred, k=5)}')
    print(f'avg_prec: {avg_precision_at_k(y_true, y_pred)}')

    y_true = [list('a'), list('n')]
    y_pred = [list('czbajepgard'), list('ahbcnfgmdek')]
    print(f'mean avg_prec: {mean_avg_precision_at_k(y_true, y_pred)}')
    print(f'MRR: {mean_reciprocal_rank(y_true, y_pred)}') 
    
