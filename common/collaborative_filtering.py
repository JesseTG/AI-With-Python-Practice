import numpy
from .compute_scores import pearson

def find_similar_users(data, user, num_users):
    scores = numpy.array([[x, pearson(data, user, x)] for x in data if x != user])
    
    # Sort the scores in descending order
    scores_sorted = numpy.argsort(scores[:, 1])[::-1]    
    top_users = scores_sorted[:num_users]
    return scores[top_users]