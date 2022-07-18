import requests
from requests import HTTPError
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import os
from dotenv import load_dotenv
from typing import List
import argparse
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm

load_dotenv()
TOKEN = os.getenv('TWITTER_API_BEARER_TOKEN')
API_ENDPOINT = os.getenv('TWITTER_API_URL', default='https://api.twitter.com/2/')
API_ENDPOINT = API_ENDPOINT + '/' if API_ENDPOINT[-1] != '/' else API_ENDPOINT


AUTH_HEADERS = {
    'Accept': 'application/json',
    'Authentication': 'Basic',
    'Authorization': f'Bearer {TOKEN}'
}

MAX_TWEETS_PER_REQUEST = 100
RATE_LIMIT_15_MINUTES = 450

def get_session(retries=5, backoff_factor=2):
    session = requests.Session()
    retries = Retry(total=retries, backoff_factor=backoff_factor)
    session.mount('https://', HTTPAdapter(max_retries=retries))

    return session


def get_user(username: str) -> dict:
    url = API_ENDPOINT + f'users/by/username/{username}'
    query = {
        'user.fields': 'id,username,name,profile_image_url,verified',
    }

    s = get_session(5, 5)
    r = s.get(url, params=query, headers=AUTH_HEADERS)
    r = r.json()

    if 'data' in r:
        user = r['data']
    else:
        status = r['status']
        raise HTTPError(f'Error {status} fetching user.')

    return user


def get_all_tweets(
        username: str,
        n_tweets: int = 10,
        exclude: List[str] = None
) -> List:
    n_tweets = max(5, n_tweets)

    url = API_ENDPOINT + f'tweets/search/all'

    n_iterations = 1 + (n_tweets // (MAX_TWEETS_PER_REQUEST + 1))

    if exclude is not None:
        exclude_param = ','.join(exclude)
    else:
        exclude_param = None

    query = {
        'max_results': min(MAX_TWEETS_PER_REQUEST, n_tweets),
        #'exclude': exclude_param,
        'query': f'from:{username}',
        'tweet.fields': 'public_metrics,created_at,conversation_id',
        'pagination_token': None
    }

    tweets = []

    s = get_session(5, 10)
    for it in tqdm(range(n_iterations)):
        if it+1 % RATE_LIMIT_15_MINUTES-10 == 0:
            print("Rate limit!")
            time.sleep(60*15) #15 minutes
            #break
        r = s.get(url, params=query, headers=AUTH_HEADERS)
        o_r = r
        r = r.json()
        if 'data' in r:
            tweets += r['data']
        else:
            status = o_r.status_code
            raise HTTPError(f'Error {status} fetching tweets from user with {username}')

        if 'next_token' in r['meta']:
            query['pagination_token'] = r['meta']['next_token']
        else:
            break

    return tweets


def add_user_info(user: dict, data: pd.DataFrame) -> pd.DataFrame:
    data['user_id'] = user['id']
    data['username'] = user['username']
    data['user_name'] = user['name']
    data['user_picture'] = user['profile_image_url']
    data['verified'] = user['verified']

    return data


def parse_tweets_to_dataframe(user: dict, tweets: List) -> pd.DataFrame:
    data = pd.DataFrame(tweets)
    if 'public_metrics' in data.columns:
        data = data.join(pd.json_normalize(data['public_metrics'])).drop('public_metrics', axis='columns')

    data = add_user_info(user, data)

    return data


def group_threads(user: dict, tweets: pd.DataFrame) -> pd.DataFrame:
    grouped = tweets.sort_values('created_at').groupby('conversation_id')
    grouped = grouped.agg(
        tweets_id=('id', lambda x: list(x)),
        tweets_in_thread=('id', lambda x: len(x.unique())),
        created_at=('created_at', min),
        text=('text', lambda x: '\n'.join(x)),
        retweet_count=('retweet_count', sum),
        like_count=('like_count', sum),
        reply_count=('reply_count', sum),
        quote_count=('quote_count', sum)
    )
    grouped = add_user_info(user, grouped)

    return grouped


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Gera datasets com todos os tweets disponíveis para os usuários fornecidos.")

    parser.add_argument('usernames', nargs='*')
    parser.add_argument('--group-threads', dest='group_threads', action='store_true')
    args = parser.parse_args()
    
    for username in args.usernames:
        tic = datetime.now()
        current_user = get_user(username)
        current_tweets = get_all_tweets(username, n_tweets=14200, exclude=['retweets', 'replies'])
        print("Tempo:",datetime.now() - tic)
        current_df = parse_tweets_to_dataframe(current_user, current_tweets)
        current_df.to_csv(f"{datetime.now().strftime(r'%Y%m%dT%H%M%S%z')}_{username}_tweets.csv", index=False)
        if args.group_threads:
            grouped_df = group_threads(current_user, current_df)
            grouped_df.to_csv(f"{datetime.now().strftime(r'%Y%m%dT%H%M%S%z')}_{username}_tweets_grouped.csv", index=False)
