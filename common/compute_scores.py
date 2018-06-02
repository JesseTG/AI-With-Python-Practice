import numpy

def euclidean(data, user1, user2):
    # Note: Assuming user1 and user2 are in the data set since we're using widgets
    
    movies1, movies2 = data[user1], data[user2]
    common_movies = frozenset(movies1) & frozenset(movies2)
    
    if not common_movies:
        # If there's no movie that both of these users have seen...
        return 0

    squared_diff = []
    for movie in movies1:
        if movie in movies2:
            squared_diff.append(numpy.square(movies1[movie] - movies2[movie]))
    
    return 1.0 / (1 + numpy.sqrt(numpy.sum(squared_diff)))

def pearson(data, user1, user2):
    movies1, movies2 = data[user1], data[user2]
    common_movies = frozenset(movies1) & frozenset(movies2)
    num_ratings = len(common_movies)
    
    if not common_movies:
        # If there's no movie that both of these users have seen...
        return 0

    # Sum of ratings
    sum1 = numpy.sum(movies1[m] for m in common_movies)
    sum2 = numpy.sum(movies2[m] for m in common_movies)
    
    # Sum of squares of ratings
    sumsquared1 = numpy.sum(numpy.square([movies1[m] for m in common_movies]))
    sumsquared2 = numpy.sum(numpy.square([movies2[m] for m in common_movies]))
    
    # Sum of products of ratings
    sum_of_products = numpy.sum(movies1[m] * movies2[m] for m in common_movies)
    
    # Correlation Scores
    Sxy = sum_of_products - (sum1 * sum2 / num_ratings)
    Sxx = sumsquared1 - numpy.square(sum1) / num_ratings
    Syy = sumsquared2 - numpy.square(sum2) / num_ratings
    
    if Sxx * Syy == 0:
        # If there's no deviation...
        return 0

    return Sxy / numpy.sqrt(Sxx * Syy)