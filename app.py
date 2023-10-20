import streamlit as st
import pandas as pd
import numpy as np
import gensim

st.title('Movie Recommendation')

# read movie info
movies = pd.read_csv("data/movies.tsv", sep="\t")

# load item2vec model
model = gensim.models.word2vec.Word2Vec.load("data/item2vec.model")

# change df into dict 
movie_titles = movies["title"].tolist()
movie_ids = movies["movie_id"].tolist()
movie_id_to_title = dict(zip(movie_ids, movie_titles))
movie_title_to_id = dict(zip(movie_titles, movie_ids))

st.markdown("## Show recommended movies toward your preference")
selected_movie = st.selectbox("Select a movie", movie_titles)
selected_movie_id = movie_title_to_id[selected_movie]
st.write(f"Your selected movie title is: {selected_movie}(id={selected_movie_id}) ")

# 似ている映画を表示
st.markdown(f"### {selected_movie} are similar to the movie")
results = []
for movie_id, score in model.wv.most_similar(selected_movie_id):
    title = movie_id_to_title[movie_id]
    results.append({"movie_id":movie_id, "title": title, "score": score})
results = pd.DataFrame(results)
st.write(results)


st.markdown("## Other movie recommendation")

selected_movies = st.multiselect("Select your movie titles", movie_titles)
selected_movie_ids = [movie_title_to_id[movie] for movie in selected_movies]
vectors = [model.wv.get_vector(movie_id) for movie_id in selected_movie_ids]
if len(selected_movies) > 0:
    user_vector = np.mean(vectors, axis=0)
    st.markdown(f"### Movies Recommendation")
    recommend_results = []
    for movie_id, score in model.wv.most_similar(user_vector):
        title = movie_id_to_title[movie_id]
        recommend_results.append({"movie_id":movie_id, "title": title, "score": score})
    recommend_results = pd.DataFrame(recommend_results)
    st.write(recommend_results)
