import tweepy
from textblob import TextBlob

consumer_key = 'nC0AcbetE6KvIAAP9djJYn0jb'
consumer_key_secret = 'qtOj4cWm50ZwScY0tPmmECudQBAlxFwFuy20TIj1od1lrUSc85'
access_token = '1274916618657619969-cwNaLECcH8fZsDQNMWwU3zjoCZvhml'
access_token_secret = '46ST4PzGnU47rgR8nPVJGajbcHHRmpx6hAetWLngdDVkz'

auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
public_tweets = api.search('Donald Trump')

for tweet in public_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
