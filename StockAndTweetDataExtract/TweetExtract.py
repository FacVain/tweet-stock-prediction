import sys
import snscrape.modules.twitter as sntwitter
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd
import multiprocessing
from pathlib import Path


def extract_tweet(stock, since, until, thread_num):
    t_query = f'{stock} since:{since.strftime("%Y-%m-%d")} until:{until.strftime("%Y-%m-%d")}'
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(t_query).get_items():
        tweets.append([tweet.date, tweet.user.username, tweet.content])
    t_df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet']).set_index('Date')
    filepath = Path(f'./{stock}/{since.year}/tweet_part{thread_num}.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    t_df.to_csv(filepath)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit('Enter stock and year as an argument.')
    stock = sys.argv[1]
    year = int(sys.argv[2])
    init_date = date(year, 1, 1)
    end_date = date(year, 12, 1)
    pnum = 0
    processes = []
    while init_date < end_date:
        s_date = init_date + relativedelta(months=2)
        processes.append(multiprocessing.Process(target=extract_tweet, args=(stock, init_date, s_date, pnum)))
        processes[pnum].start()
        print(f'Process {init_date.strftime("%Y-%m-%d")} started')
        init_date = s_date
        pnum += 1

    for process in processes:
        process.join()
