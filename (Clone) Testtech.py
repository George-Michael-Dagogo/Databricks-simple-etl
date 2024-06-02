# Databricks notebook source
# MAGIC %md
# MAGIC ## Extracting and Transforming data from News API

# COMMAND ----------

import os
os.system("pip install -r https://raw.githubusercontent.com/George-Michael-Dagogo/World_news_tutorial/main/requirements.txt")


from newsapi.newsapi_client import NewsApiClient
import pandas as pd
from newspaper import Article, Config
from nltk.corpus import stopwords
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import date, timedelta



def extract_transform_function():

    today = date.today()
    yesterday = today - timedelta(days = 1)
    day_before_yesterday = today - timedelta(days = 2)
    # Init
    newsapi = NewsApiClient(api_key='ff4373852c2343a98303951439854f8c')

    # /v2/top-headlines
    top_headlines = newsapi.get_top_headlines(   
                                            category='entertainment',
                                            language='en',
                                            page_size = 90,
                                            page= 1)

    articles = top_headlines.get('articles',[])

    init_df = pd.DataFrame(articles, columns = ['source','title','publishedAt','author','url'])

    init_df['source'] = init_df['source'].apply(lambda x: x['name'] if pd.notna(x) and 'name' in x else None)

    init_df['publishedAt'] = pd.to_datetime(init_df['publishedAt'])



    filtered_df = init_df[(init_df['publishedAt'].dt.date == day_before_yesterday) | (init_df['publishedAt'].dt.date == yesterday)]
    filtered_df.rename(columns={'publishedAt': 'date_posted'}, inplace=True)


    df = filtered_df.copy()

    def full_content(url):
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
        config = Config()
        config.browser_user_agent = user_agent
        page = Article(url, config = config)

        try:
            page.download()
            page.parse()
            return page.text
        except Exception as e:
            print(f"Error retrieving content from {url}: {e}")
            return 'couldnt retrieve'


    df['content'] = df['url'].apply(full_content)
    df['content'] = df['content'].str.replace('\n', ' ')
    df = df[df['content'] != 'couldnt retrieve']



    # Download the stopwords dataset
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')


    def count_words_without_stopwords(text):
        if isinstance(text, (str, bytes)):
            words = nltk.word_tokenize(str(text))
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word.lower() not in stop_words]
            return len(filtered_words)
        else:
            0

    df['word_count'] = df['content'].apply(count_words_without_stopwords)

    nltk.download('vader_lexicon')

    sid = SentimentIntensityAnalyzer()

    def get_sentiment(row):
        sentiment_scores = sid.polarity_scores(row)
        compound_score = sentiment_scores['compound']

        if compound_score >= 0.05:
            sentiment = 'Positive'
        elif compound_score <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return sentiment, compound_score

    df[['sentiment', 'compound_score']] = df['content'].astype(str).apply(lambda x: pd.Series(get_sentiment(x)))

    return df

    

dataframe = extract_transform_function()




# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Database and Table If Not Exists

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS the_news;
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS the_news.news_table (
# MAGIC   source STRING,
# MAGIC   title STRING,
# MAGIC   date_posted DATE,
# MAGIC   author STRING,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   word_count INT,
# MAGIC   sentiment STRING,
# MAGIC   compound_score DOUBLE
# MAGIC )

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the schema explicitly to avoid datatype errors

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType, IntegerType, DoubleType

spark = SparkSession.builder.appName("CreateTableExample").getOrCreate()
schema = StructType([
    StructField("source", StringType(), True),
    StructField("title", StringType(), True),
    StructField("date_posted", DateType(), True), 
    StructField("author", StringType(), True),
    StructField("url", StringType(), True),
    StructField("content", StringType(), True),
    StructField("word_count", IntegerType(), True),
    StructField("sentiment", StringType(), True),
    StructField("compound_score", DoubleType(), True)
])

spark_df = spark.createDataFrame(dataframe, schema=schema)
spark_df.write.mode('append').saveAsTable('the_news.news_table')



