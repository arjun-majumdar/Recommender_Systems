

"""
Using Cosine Similarity (the dot product for normalized vectors) to build a Movie Recommender System.


Recommender systems are an important class of ML algorithms that offer 'relevant' suggestions to users.
Youtube, Amazon, Netflix, all function on recommendation systems where the system recommends you the next
video or product:

1. based on your past activity -> Content-based Filtering (or)
2. based on activities and preferences of other users similar to you -> Collaborative Filtering

Likewise, Facebook also uses a recommendation system to suggest Facebook users you may know offline.


Recommendation Systems work based on the 'similarity' between either the content or the users who access
the content.

There are several ways to measure the similarity between two items. The recommendation systems use this
'similarity matrix' to recommend the next most similar product to the user.

Here, we will build a ML algorithm that would recommend movies based on a movie the user likes. This ML
model would be based on 'Cosine Similarity'.


Refer-
https://towardsdatascience.com/using-cosine-similarity-to-build-a-movie-recommendation-system-ae7f20842599
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("movie_dataset.csv")

data.shape
# (4803, 24)

data.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4803 entries, 0 to 4802
Data columns (total 24 columns):
 #   Column                Non-Null Count  Dtype
---  ------                --------------  -----
 0   index                 4803 non-null   int64
 1   budget                4803 non-null   int64
 2   genres                4775 non-null   object
 3   homepage              1712 non-null   object
 4   id                    4803 non-null   int64
 5   keywords              4391 non-null   object
 6   original_language     4803 non-null   object
 7   original_title        4803 non-null   object
 8   overview              4800 non-null   object
 9   popularity            4803 non-null   float64
 10  production_companies  4803 non-null   object
 11  production_countries  4803 non-null   object
 12  release_date          4802 non-null   object
 13  revenue               4803 non-null   int64
 14  runtime               4801 non-null   float64
 15  spoken_languages      4803 non-null   object
 16  status                4803 non-null   object
 17  tagline               3959 non-null   object
 18  title                 4803 non-null   object
 19  vote_average          4803 non-null   float64
 20  vote_count            4803 non-null   int64
 21  cast                  4760 non-null   object
 22  crew                  4803 non-null   object
 23  director              4773 non-null   object
dtypes: float64(3), int64(5), object(16)
memory usage: 900.7+ KB
'''

data.isna().values.any()
# True

data.isna().sum()
'''
index                      0
budget                     0
genres                    28
homepage                3091
id                         0
keywords                 412
original_language          0
original_title             0
overview                   3
popularity                 0
production_companies       0
production_countries       0
release_date               1
revenue                    0
runtime                    2
spoken_languages           0
status                     0
tagline                  844
title                      0
vote_average               0
vote_count                 0
cast                      43
crew                       0
director                  30
dtype: int64
'''

# Get column names-
data.columns
'''
Index(['index', 'budget', 'genres', 'homepage', 'id', 'keywords',
       'original_language', 'original_title', 'overview', 'popularity',
       'production_companies', 'production_countries', 'release_date',
       'revenue', 'runtime', 'spoken_languages', 'status', 'tagline', 'title',
       'vote_average', 'vote_count', 'cast', 'crew', 'director'],
      dtype='object')
'''


# The features/columns we are interested to compute cosine similarity for making new recommendations are:
# keywords, cast, genres & director

'''
A user who likes a horror movie will most probably like another horror movie. Some users may like seeing
their favorite actors in the cast of the movie. Others may love movies directed by a particular person.
Combining all of these aspects, our shortlisted 4 features are sufficient to train our recommendation
algorithm.
'''

'''
We will import Scikit-learn’s 'CountVectorizer()' which is used to convert a collection of text documents
to a vector of term/token counts.

We will also import the 'cosine_similarity()' from sklearn, as the metric of our similarity matrix.
'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Choose relevant columns/features-
features = ['keywords', 'cast', 'genres', 'director']

# Check for missing values for relevant columns-
for col in features:
    print(f"col = {col} has = {data[col].isna().sum()} missing values")
'''
col = keywords has = 412 missing values
col = cast has = 43 missing values
col = genres has = 28 missing values
col = director has = 30 missing values
'''

# Data preprocessing: replace any rows having NaN values with a space/empty string, so it does not
# generate an error while running the code-
for col in features:
    data[col] = data[col].fillna('')


'''
Combining Relevant Features into a Single Feature:

Next, we will define a function called 'combined_features()'. This function will combine all of our
useful features (keywords, cast, genres & director) from their respective rows, and return a row with
all 'the combined features in a single string'.
'''
def combined_features(row):
    '''
    Function to combine relevant features- keywords, cast, genres and director from their respective
    rows and return a row with all of the combined columns/features as a single string.
    '''
    # separate columns/features with space-
    return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']


# Add a new column 'combined_features' to our existing pd DataFrame and apply 'combined_features()'
# function to each row (axis = 1)-
data['combined_features'] = data.apply(combined_features, axis = 1)

# Sanity check-
data.loc[:5, 'combined_features']
'''
0    culture clash future space war space colony so...
1    ocean drug abuse exotic island east india trad...
2    spy based on novel secret agent sequel mi6 Dan...
3    dc comics crime fighter terrorist secret ident...
4    based on novel mars medallion space travel pri...
5    dual identity amnesia sandstorm love of one's ...
Name: combined_features, dtype: object
'''


'''
Extracting Features:

Next, we will extract features from our data.

'sklearn.feature_extraction' module can be used to extract features in a format supported by machine
learning algorithms from datasets consisting of formats such as text and image. We will use
CountVectorizer’s 'fit.tranform' to count the number of texts and we will print the transformed matrix
count_matrix into an array for better understanding.
'''
cv = CountVectorizer()
count_matrix = cv.fit_transform(data['combined_features'])

type(count_matrix)
# scipy.sparse.csr.csr_matrix

count_matrix.shape
# (4803, 14845)

print(f"Count Matrix: {count_matrix.toarray()}")
'''
Count Matrix: [[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]]
'''


# Convert Sparse Matrix to pd DataFrame to see words-
doc_term_matrix = count_matrix.todense()

type(doc_term_matrix)
# numpy.matrix

doc_term_matrix_df = pd.DataFrame(data = doc_term_matrix, columns = cv.get_feature_names())

doc_term_matrix_df.head()
'''
   11  15th  17th  18th  1910s  1917  1920s  1930s  ...  zuleikha  zuniga  zurer  zvyagintsev  zwart  zwick  zwigoff  zylka
0   0     0     0     0      0     0      0      0  ...         0       0      0            0      0      0        0      0
1   0     0     0     0      0     0      0      0  ...         0       0      0            0      0      0        0      0
2   0     0     0     0      0     0      0      0  ...         0       0      0            0      0      0        0      0
3   0     0     0     0      0     0      0      0  ...         0       0      0            0      0      0        0      0
4   0     0     0     0      0     0      0      0  ...         0       0      0            0      0      0        0      0

[5 rows x 14845 columns]
'''


# Use TF-IDF Vectorizer:
tfidf_vec = TfidfVectorizer()
count_matrix_tfidf = tfidf_vec.fit_transform(data['combined_features'])

count_matrix_tfidf.shape
# (4803, 14845)

# Convert to dense matrix-
doc_term_tfidf_matrix = count_matrix_tfidf.todense()

# Create a pd DataFrame-
doc_term_tfidf_matrix_df = pd.DataFrame(data = doc_term_tfidf_matrix, columns = tfidf_vec.get_feature_names())

doc_term_tfidf_matrix_df.head()
'''
    11  15th  17th  18th  1910s  1917  1920s  1930s  ...  zuleikha  zuniga  zurer  zvyagintsev  zwart  zwick  zwigoff  zylka
0  0.0   0.0   0.0   0.0    0.0   0.0    0.0    0.0  ...       0.0     0.0    0.0          0.0    0.0    0.0      0.0    0.0
1  0.0   0.0   0.0   0.0    0.0   0.0    0.0    0.0  ...       0.0     0.0    0.0          0.0    0.0    0.0      0.0    0.0
2  0.0   0.0   0.0   0.0    0.0   0.0    0.0    0.0  ...       0.0     0.0    0.0          0.0    0.0    0.0      0.0    0.0
3  0.0   0.0   0.0   0.0    0.0   0.0    0.0    0.0  ...       0.0     0.0    0.0          0.0    0.0    0.0      0.0    0.0
4  0.0   0.0   0.0   0.0    0.0   0.0    0.0    0.0  ...       0.0     0.0    0.0          0.0    0.0    0.0      0.0    0.0

[5 rows x 14845 columns]
'''

# https://www.machinelearningplus.com/cosine-similarity/
# https://www.sciencedirect.com/topics/computer-science/cosine-similarity




"""
Using the Cosine Similarity:

We will use the Cosine Similarity from Sklearn, as the metric to compute the similarity between two movies.

Cosine similarity is a metric used to measure how similar two items are. Mathematically, it measures the
cosine of the angle between two vectors projected in a multi-dimensional space. The output value ranges
from 0 to 1.

0 means no similarity, where as 1 means that both the items are 100% similar.

The python Cosine Similarity or cosine kernel, computes similarity as the normalized dot product of
input samples X and Y. We will use the sklearn 'cosine_similarity' to find the cosθ for the two vectors
in the count matrix-
"""
cosine_sim = cosine_similarity(count_matrix)

type(cosine_sim)
# numpy.ndarray

cosine_sim.shape
# (4803, 4803)


'''
'cosine_sim' matrix is a numpy array with calculated cosine similarity between each movies. As you can
see below, the cosine similarity of movie 0 with movie 0 is 1; they are 100% similar (as should be).

Similarly the cosine similarity between movie 0 and movie 1 is 0.105409 (the same score between movie 1
and movie 0 — order does not matter).

Movies 0 and 4 are more similar to each other (with a similarity score of 0.23094) than movies 0 and 3
(score = 0.0377426).

The diagonal with 1s suggests what the case is, each movie ‘x’ is 100% similar to itself.
'''
cosine_sim[:5, :5]
'''
array([[1.        , 0.10540926, 0.12038585, 0.03774257, 0.23094011],
       [0.10540926, 1.        , 0.0761387 , 0.03580574, 0.07302967],
       [0.12038585, 0.0761387 , 1.        , 0.16357216, 0.20851441],
       [0.03774257, 0.03580574, 0.16357216, 1.        , 0.03922323],
       [0.23094011, 0.07302967, 0.20851441, 0.03922323, 1.        ]])
'''


'''
Content User likes:

The next step is to take as input a movie that the user likes in the 'movie_user_likes' variable.

Since we are building a content based filtering system, we need to know the users’ likes in order to
predict a similar item.
'''
movie_user_likes = "Dead Poets Society"

def get_index_from_title(title):
    return data[data['title'] == title]['index'].values[0]
    # return data[data.title == title]["index"].values[0]


movie_index = get_index_from_title(movie_user_likes)

movie_index
# 2453

data.loc[movie_index, 'title']
# 'Dead Poets Society'


'''
Generating the Similar Movies Matrix:

Next we will generate a list of similar movies. We will use the 'movie_index' of the movie we have given
as input 'movie_user_likes'. The 'enumerate()' method will add a counter to the iterable list
'cosine_sim' and return it in a form of a list 'similar_movies' with the similarity score of each index.
'''
similar_movies = list(enumerate(cosine_sim[movie_index]))

#len(similar_movies)
# 4803

similar_movies[:5]
# [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.04499212706658475), (4, 0.0)]
# counter, similarity score of each index


'''
Sorting the Similar Movies List in Descending Order:

The next step is to sort the movies in the list 'similar_movies'. We have used the parameter
'reverse=True' since we want the list in the descending order, with the most similar item at the top.
'''
sorted_similar_movies = sorted(similar_movies, key = lambda x: x[1], reverse = True)
# 'key' parameter uses the second argument instead of first argument which is counter

'''
'sorted_similar_movies' will be a list of all the movies sorted in descending order with respect to their
similarity score with the input movie- 'movie_user_likes'.

As can be seen below, the most similar one with a similarity score of 0.9999999999999993 is at the top
with its index number 2453 (the movie is 'Dead Poets Society' which we gave as input).
'''
sorted_similar_movies[:5]
'''
[(2453, 0.9999999999999993),
 (3250, 0.21677749238102995),
 (905, 0.21052631578947364),
 (2975, 0.2051956704170308),
 (825, 0.19564639521780736)]
'''

data.loc[2453, 'title']
# 'Dead Poets Society'

data.loc[3250, 'title']
# 'Much Ado About Nothing'

data.loc[905, 'title']
# 'Patch Adams'


def get_title_from_index(index):
    return data[data['index'] == index]['title'].values[0]


# Sanity check-
get_title_from_index(3250)
# 'Much Ado About Nothing'

get_title_from_index(905)
# 'Patch Adams'


# Print top k = 5 similar movies-
k = 5
print(f"\nTop {k} similar movies are:\n")

c = 0
for movie in sorted_similar_movies[1: k + 1]:
    print(f"Movie: {get_title_from_index(movie[0])}, similarity score = {movie[1]:.4f}")
    c += 1
    
    if c > 5:
        break

'''
Top 5 similar movies are:

Movie: Much Ado About Nothing, similarity score = 0.2168
Movie: Patch Adams, similarity score = 0.2105
Movie: Good Will Hunting, similarity score = 0.2052
Movie: Flightplan, similarity score = 0.1956
Movie: Alive, similarity score = 0.1913
'''


def get_top_k_similar_movies(sorted_similar_movies, k = 5):
    # print(f"Top {k} similar movies are:\n")
    
    # Python3 dict to hold: key = movie name, value = similarity score
    top_k_similar_movies = {}
    
    c = 0
    for movie in sorted_similar_movies[1: k + 1]:
        # print(f"Movie: {get_title_from_index(movie[0])}, similarity score = {movie[1]:.4f}")
        top_k_similar_movies[get_title_from_index(movie[0])] = movie[1]
        c += 1
    
        if c > 5:
            break

    return top_k_similar_movies


top_k_similar_movies = get_top_k_similar_movies(sorted_similar_movies, k = 6)

top_k_similar_movies
'''
{'Much Ado About Nothing': 0.21677749238102995,
 'Patch Adams': 0.21052631578947364,
 'Good Will Hunting': 0.2051956704170308,
 'Flightplan': 0.19564639521780736,
 'Alive': 0.19134594929397594,
 'The Basket': 0.1908854288927333}
'''

