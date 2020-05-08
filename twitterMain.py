import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tweepy
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from textblob import TextBlob
from tweepy import OAuthHandler


warnings.filterwarnings('ignore')


class TwitterClient():
    def __init__(self):
        try:
            # Access Credentials
            consumer_key = ""
            consumer_secret = ""
            access_token = ""
            access_token_secret = ""

            # OAuthHandler object
            auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

        except tweepy.TweepError as e:
            print(f"Error: Twitter Authentication Failed - \n{str(e)}")

            # Function to fetch tweets

    def ScrapData(self, query, limit):
        # empty list to store parsed tweets
        tweets = []
        sinceId = None
        max_id = -1
        tweetCount = 0
        tweetsPerQry = 500
        tempList = []
        tweetList = []
        limit = limit
        labelList = []
        searchTerm = query
        userNameList = []

        while tweetCount < limit:
            try:
                if max_id <= 0:
                    if not sinceId:
                        new_tweets = self.api.search(q=query, count=tweetsPerQry)
                    else:
                        new_tweets = self.api.search(q=query, count=tweetsPerQry,
                                                     since_id=sinceId)
                else:
                    if not sinceId:
                        new_tweets = self.api.search(q=query, count=tweetsPerQry,
                                                     max_id=str(max_id - 1))
                    else:
                        new_tweets = self.api.search(q=query, count=tweetsPerQry,
                                                     max_id=str(max_id - 1),
                                                     since_id=sinceId)
                if not new_tweets:
                    print("No more tweets found")
                    break

                print(type(new_tweets))

                for tweet in new_tweets:
                    userList = tweet._json['entities']['user_mentions']

                    if len(userList) != 0:
                        userNameDict = {'username': userList[0]['name']}

                    parsed_tweet = {'tweet': tweet.text}

                    # appending parsed tweet to tweets list
                    if tweet.retweet_count > 0:
                        # if tweet has retweets, ensure that it is appended only once
                        tempList.append(parsed_tweet)
                        if userNameDict not in userNameList:
                            userNameList.append(userNameDict)
                    else:
                        tempList.append(parsed_tweet)
                        userNameList.append(userNameDict)

                tweetCount += len(new_tweets)
                print("Downloaded {0} tweets".format(tweetCount))
                max_id = new_tweets[-1].id

            except tweepy.TweepError as e:
                print("Tweepy error : " + str(e))
                break

        print("tempList>>>>>>>>>>>>>>>>>>>", tempList)

        for i in tempList:
            tweets.append(i['tweet'])

        print(">>>>>>>>", tweets)

        polarity = 0
        positive = 0
        wpositive = 0
        spositive = 0
        negative = 0
        wnegative = 0
        snegative = 0
        neutral = 0

        # iterating through tweets fetched
        for tweet in tweets:
            # Append to temp so that we can store in csv later. I use encode UTF-8
            new_text = str((self.cleanTweet(tweet).encode('utf-8')), 'utf-8')
            tweetList.append(new_text)

            analysis = TextBlob(tweet)
            # print(analysis.sentiment)  # print tweet's polarity
            polarity += analysis.sentiment.polarity  # adding up polarities to find the average later

            if analysis.sentiment.polarity == 0:
                labelDict = {'label': 0}
                labelList.append(labelDict)
            elif 0 < analysis.sentiment.polarity <= 0.3 or 0.3 < analysis.sentiment.polarity <= 0.6 or 0.6 < analysis.sentiment.polarity <= 1:
                labelDict = {'label': 1}
                labelList.append(labelDict)
            elif -0.3 < analysis.sentiment.polarity <= 0 or -0.6 < analysis.sentiment.polarity <= -0.3 or -1 < analysis.sentiment.polarity <= -0.6:
                labelDict = {'label': 2}
                labelList.append(labelDict)

            if analysis.sentiment.polarity == 0:  # adding reaction of how people are reacting to find average later
                neutral += 1
            elif 0 < analysis.sentiment.polarity <= 0.3:
                wpositive += 1
            elif 0.3 < analysis.sentiment.polarity <= 0.6:
                positive += 1
            elif 0.6 < analysis.sentiment.polarity <= 1:
                spositive += 1
            elif -0.3 < analysis.sentiment.polarity <= 0:
                wnegative += 1
            elif -0.6 < analysis.sentiment.polarity <= -0.3:
                negative += 1
            elif -1 < analysis.sentiment.polarity <= -0.6:
                snegative += 1

        df1 = pd.DataFrame(userNameList, columns=['username'])
        df2 = pd.DataFrame(tweetList, columns=['tweet'])
        df3 = pd.DataFrame(labelList, columns=['label'])

        df = pd.concat([df1, df2, df3], axis=1)

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>\n", df)

        print(type(df))
        df.to_excel('Scrapping.xlsx', index=False)

        percentage_positive = self.percentage(positive, limit)
        percentage_wpositive = self.percentage(wpositive, limit)
        percentage_spositive = self.percentage(spositive, limit)
        percentage_negative = self.percentage(negative, limit)
        percentage_wnegative = self.percentage(wnegative, limit)
        percentage_snegative = self.percentage(snegative, limit)
        percentage_neutral = self.percentage(neutral, limit)

        polarity = polarity / limit

        print("How people are reacting on " + searchTerm + " by analyzing " + str(limit) + " tweets.")

        if polarity == 0:
            print("Neutral")
        elif 0 < polarity <= 0.3:
            print("Weakly Positive")
        elif 0.3 < polarity <= 0.6:
            print("Positive")
        elif 0.6 < polarity <= 1:
            print("Strongly Positive")
        elif -0.3 < polarity <= 0:
            print("Weakly Negative")
        elif -0.6 < polarity <= -0.3:
            print("Negative")
        elif -1 < polarity <= -0.6:
            print("Strongly Negative")

        print("neutral>>>>>>>>>>>>>", neutral)
        print("positive>>>>>>>>>>>>>", positive)
        print("wpositive>>>>>>>>>>>>>", wpositive)
        print("spositive>>>>>>>>>>>>>", spositive)
        print("negative>>>>>>>>>>>>>", negative)
        print("wnegative>>>>>>>>>>>>>", wnegative)
        print("snegative>>>>>>>>>>>>>", snegative)

        print(str(percentage_neutral) + "% people thought it was neutral")
        print(str(percentage_positive) + "% people thought it was positive")
        print(str(percentage_wpositive) + "% people thought it was weakly positive")
        print(str(percentage_spositive) + "% people thought it was strongly positive")
        print(str(percentage_negative) + "% people thought it was negative")
        print(str(percentage_wnegative) + "% people thought it was weakly negative")
        print(str(percentage_snegative) + "% people thought it was strongly negative")

        self.plotPieChart(percentage_positive, percentage_wpositive, percentage_spositive, percentage_negative,
                          percentage_wnegative, percentage_snegative, percentage_neutral, searchTerm,
                          limit)

    def cleanTweet(self, tweet):
        # Remove Links, Special Characters etc from tweet
        return re.sub(r"[^a-zA-Z0-9]+", ' ', tweet)

        # function to calculate percentage

    def percentage(self, part, whole):
        temp = 100 * float(part) / float(whole)
        return format(temp, '.2f')

    def plotPieChart(self, percentage_positive, percentage_wpositive, percentage_spositive, percentage_negative,
                     percentage_wnegative, percentage_snegative, percentage_neutral, searchTerm,
                     noOfSearchTerms):
        labels = ['Positive [' + str(percentage_positive) + '%]',
                  'Weakly Positive [' + str(percentage_wpositive) + '%]',
                  'Strongly Positive [' + str(percentage_spositive) + '%]',
                  'Neutral [' + str(percentage_neutral) + '%]',
                  'Negative [' + str(percentage_negative) + '%]',
                  'Weakly Negative [' + str(percentage_wnegative) + '%]',
                  'Strongly Negative [' + str(percentage_snegative) + '%]']
        sizes = [percentage_positive, percentage_wpositive, percentage_spositive, percentage_neutral,
                 percentage_negative,
                 percentage_wnegative, percentage_snegative]
        colors = ['yellowgreen', 'lightgreen', 'darkgreen', 'gold', 'red', 'lightsalmon', 'darkred']
        patches, texts = plt.pie(sizes, colors=colors, startangle=90)
        plt.legend(patches, labels, loc="best")
        plt.title('How people are reacting on ' + searchTerm + ' by analyzing ' + str(noOfSearchTerms) + ' Tweets.')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def train_test(self, tweetType):
        tweetType = tweetType.rstrip(' ')

        df = pd.read_excel('Scrapping.xlsx')

        le1 = LabelEncoder()

        X = le1.fit_transform(df['tweet'].astype(str))

        y = le1.fit_transform(df['label'].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=75)

        X_train = np.array(X_train.reshape(-1, 1))
        X_test = np.array(X_test.reshape(-1, 1))
        y_train = np.array(y_train.reshape(-1, 1))
        y_test = np.array(y_test.reshape(-1, 1))

        print('----------------Train Model----------------------')

        classifier_svm = SVC(probability=True, kernel='rbf')  # radial basis function
        classifier_svm.fit(X_train, y_train)

        filename = 'Twitter_Model.sav'
        joblib.dump(classifier_svm, filename)

        predicted_y = classifier_svm.predict(X_test)
        # print('prediction:::', prediction)

        print("MAE: %.2f" % mean_absolute_error(y_test, predicted_y))

        score = classifier_svm.score(X_test, y_test)
        print('Training Accuracy::%.2f', score)

        print('------------------Test Model--------------------')

        model_dump = joblib.load('Twitter_Model.sav')

        testSample = le1.fit_transform(df['tweet'].astype(str))

        testSample = np.array(testSample.reshape(-1, 1))

        prediction = model_dump.predict(testSample)

        print('Test Accuracy:::: %.2f', model_dump.score(testSample, prediction))

        for i in range(len(prediction)):
            if prediction[i] > 1:
                prediction[i] = 2

        # --------------------------------------------------------------------------
        tweet = list(df['tweet'])
        userName = list(df['username'])

        neutralCount = 0
        positiveCount = 0
        negativeCount = 0
        label = []
        for j in prediction:
            if j == 0:
                neutralCount += 1
                label.append('Neutral')
            elif j == 1:
                positiveCount += 1
                label.append('Non Suspicious')
            else:
                negativeCount += 1
                label.append('Suspicious')

        print('neutralCount::::::', neutralCount)
        print('positiveCount:::::', positiveCount)
        print('negativeCount:::::', negativeCount)

        columnName = ['Username', 'Tweet', 'Result']

        df1 = pd.DataFrame({'Username': userName, 'Tweet': tweet, 'Result': label},
                           columns=columnName)

        df1.to_excel('{}.xlsx'.format(tweetType), index=False)


twitterClient = TwitterClient()

tweetType = input('Enter TweetType::')
noOfTweet = int(input('Enter no of tweet::'))

twitterClient.ScrapData(tweetType, noOfTweet)
twitterClient.train_test(tweetType)
