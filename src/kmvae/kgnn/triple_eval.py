import torch


def get_triple_ranks(triple_scores, descending=False, method='average'):
    if descending:
        triple_scores = torch.nan_to_num(triple_scores, nan=-float('inf'))
    else:
        triple_scores = torch.nan_to_num(triple_scores, nan=float('inf'))

    ranks = torch.empty(len(triple_scores))
    for i in range(len(triple_scores)):
        ranks[i] = get_triple_rank(
            triple_scores[i],
            triple_scores[i, 0],
            descending=descending,
            method=method)
    return ranks


def get_triple_rank(
        triple_scores,
        true_score,
        descending=False,
        method='average'):
    sorted_scores, _ = torch.sort(triple_scores, descending=descending)
    tied_ranks = (sorted_scores == true_score).cpu().nonzero().float() + 1

    if method == 'average':
        rank = torch.mean(tied_ranks)
    elif method == 'min':
        rank = torch.min(tied_ranks)
    elif method == 'max':
        rank = torch.max(tied_ranks)
    else:
        assert 'method must be one of (average, min, max)'

    return rank
