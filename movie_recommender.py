

import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer


def discover_movie(movie_merge):
 
    """The function extracts the information of movie genres, keywords, directors, actors, and titles, and returns a matrix of cosine similarity.

    Args:
        NULL

    Returns:
        matrix(pandas.core.frame.DataFrame): The cosine similarity matrix for recommender system.
    """
    '''
    # load data tables
    movie = pd.read_csv("movies_metadata.csv")
    credit = pd.read_csv("credits.csv")
    keyword = pd.read_csv("keywords.csv")

    # deal with NAs in movie 
    movie = movie.drop(movie.index[19730])
    movie = movie.drop(movie.index[29502])
    movie = movie.drop(movie.index[35585])
    pd.to_numeric(movie['id'])

    # select a sample of data
    movie = movie.head(5000)

    # genre columns
    movie['genres'] = movie['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    # transform the data type of id column
    keyword['id'] = keyword['id'].astype('int')
    credit['id'] = credit['id'].astype('int')
    movie['id'] = movie['id'].astype('int')

    # merge movie, keyword, credit
    movie_merge = movie.merge(keyword, on='id')
    movie_merge = movie_merge.merge(credit, on='id')

    # for the credits table:
    # we only take directors from crew
    # and pick the top2 actors from cast
    movie_merge['cast'] = movie_merge['cast'].apply(literal_eval)
    movie_merge['crew'] = movie_merge['crew'].apply(literal_eval)
    movie_merge['keywords'] = movie_merge['keywords'].apply(literal_eval)
    movie_merge['cast_size'] = movie_merge['cast'].apply(lambda x: len(x))
    movie_merge['crew_size'] = movie_merge['crew'].apply(lambda x: len(x))

    # extract director from crew
    def get_director(input):
        """read crew table

        Args:
            param1 (dictionary): crew table

        Returns:
            the director name 
        """
        for item in input:
            if (item['job'] == 'Director'):
                return item['name']
        return np.nan


    movie_merge['director'] = movie_merge['crew'].apply(get_director)

    # extract actors from cast
    movie_merge['cast'] = movie_merge['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    movie_merge['cast'] = movie_merge['cast'].apply(lambda x: x[:2] if len(x) >=2 else x)

    # deal with keywords
    movie_merge['keywords'] = movie_merge['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    # tranform actors into lower case
    movie_merge['cast'] = movie_merge['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

    # transform directors into lower case
    movie_merge['director'] = movie_merge['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
    movie_merge['director'] = movie_merge['director'].apply(lambda x: [x,x, x])

    # count the frequency of each keywords
    s = movie_merge.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'keyword'
    s = s.value_counts()

    # As i am doing frequency count, the keyword with frequency as 1 should be dropped.
    s = s[s>1]

    # we have to make sure singular and plural forms of a word can be recognized as one single word
    stemmer = SnowballStemmer('english')


    # filter keywords
    def filter_keywords(x):
        """extract keywords

        Args:
            param1 (String): keywords string

        Returns:
            a list of keywords
        """
        words = []
        for item in x:
            if item in s:
                words.append(item)
        return words



    movie_merge['keywords'] = movie_merge['keywords'].apply(filter_keywords)
    movie_merge['keywords'] = movie_merge['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    movie_merge['keywords'] = movie_merge['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    
    movie_merge['criteria'] = movie_merge['keywords'] + movie_merge['cast'] + movie_merge['director'] + movie_merge['genres']
    movie_merge['criteria'] = movie_merge['criteria'].apply(lambda x: ' '.join(x))
    '''
    count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    count_matrix = count.fit_transform(movie_merge['criteria'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    titles = movie_merge['title']
    indices = pd.Series(movie_merge.index, index=movie_merge['title'])

    return (indices, cosine_sim, titles)







