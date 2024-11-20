import argparse
import json
import numpy as np

def pearson_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')
    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    num_ratings = len(common_movies)

    if num_ratings == 0:
        return 0

    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])

    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])
    Sxx = user1_squared_sum - (np.square(user1_sum) / num_ratings)
    Syy = user2_squared_sum - (np.square(user2_sum) / num_ratings)
    Sxy = sum_of_products - ((user1_sum * user2_sum) / num_ratings)

    if Sxx * Syy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy)

def find_similar_users(dataset, user, num_users):
    if user not in dataset:
        raise TypeError('Cannot find ' + user + ' in the dataset')

    scores = np.array([[x, pearson_score(dataset, user, x)] for x in dataset if x != user])
    scores_sorted = np.argsort(scores[:, 1])[::-1]
    top_users = scores_sorted[:num_users]
    return scores[top_users]

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find the movie recommenda-tions for the given user')
    parser.add_argument('--user', dest='user', required=True, help='Input user')
    return parser

def get_recommendations(dataset, input_user):
    if input_user not in dataset:
        raise TypeError('Cannot find ' + input_user + ' in the dataset')

    overall_scores = {}
    similarity_scores = {}

    for user in [x for x in dataset if x != input_user]:
        similarity_score = pearson_score(dataset, input_user, user)

        if similarity_score <= 0:
            continue

        filtered_list = [x for x in dataset[user] if x not in da-taset[input_user] or dataset[input_user][x] == 0]

        for item in filtered_list:
            if item not in overall_scores:
                overall_scores[item] = 0
                similarity_scores[item] = 0
            overall_scores[item] += dataset[user][item] * similarity_score
            similarity_scores[item] += similarity_score

    if len(overall_scores) == 0:
        return ['No recommendations possible']

    movie_scores = np.array([[score / similarity_scores[item], item] for item, score in overall_scores.items()])
    movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]
    movie_recommendations = [movie for _, movie in movie_scores]

    return movie_recommendations

args = build_arg_parser().parse_args()
user = args.user

ratings_file = 'ratings.json'
with open(ratings_file, 'r') as f:
    data = json.loads(f.read())

print("\nMovie recommendations for " + user + ":")
movies = get_recommendations(data, user)

for i, movie in enumerate(movies):
    print(str(i + 1) + '. ' + movie)
