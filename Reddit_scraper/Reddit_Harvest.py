import praw

reddit = praw.Reddit(client_id='PurljHA0fr4yYQ',
                    client_secret='URh00WvOlXQz5O_06VhkBhIXtus',
                    user_agent='Maybeatestytestmaybeatroll')
filename = "nosleep.txt"
with open(filename,"a+",encoding="utf-8") as fileboi:
    for submission in reddit.subreddit('nosleep').top(time_filter='all', limit=10000):
        fileboi.write(submission.selftext)


