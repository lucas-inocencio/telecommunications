import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from statistics import mean
from random import shuffle

positive_review_ids = nltk.corpus.movie_reviews.fileids(categories=["pos"])
negative_review_ids = nltk.corpus.movie_reviews.fileids(categories=["neg"])
all_review_ids = positive_review_ids + negative_review_ids
sia = SentimentIntensityAnalyzer()

def get_mean_scores(review_id: str) -> bool:
    """True if the average of all sentence compound scores is positive."""
    text = nltk.corpus.movie_reviews.raw(review_id)
    scores = [
        sia.polarity_scores(sentence)["compound"]
        for sentence in nltk.sent_tokenize(text)
    ]
    return f"{review_id}: {mean(scores)}"

shuffle(all_review_ids)

for i, review_id in enumerate(all_review_ids):
    if i == 3:
        break
    print(get_mean_scores(review_id))
