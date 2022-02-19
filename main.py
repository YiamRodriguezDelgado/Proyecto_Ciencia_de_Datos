import streamlit as st
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import preprocessor as p
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from PIL import Image
import re
import plotly.express as px
from user import df_tweet
from datetime import date
from datetime import datetime
import streamlit.components.v1 as components
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import keras
from keras.preprocessing.sequence import pad_sequences
from PIL import Image
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import Bidirectional
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
nltk.download('stopwords')
from nltk.corpus import stopwords


import joblib
import numpy as np
# nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Removing stopwords
stop = nltk.corpus.stopwords.words('spanish')
plt.style.use('fivethirtyeight')

consumer_key = 'UjYQZT9P9vdYq6OzEFPsqSLQB'
consumer_secret = 'LWKnd023OSmb6kfTbDd5zPjo4P0LcQodvg1wRf7LynWCgOH8Nb'
access_token = '384431766-SdZnMYaETCYPiI6NyogMtZSEZq95dAYqZyDNkWhU'
access_token_secret = '5GD2CpP3Bt3ZvsCjiMV0LvkNnh5oBiJiEeJNW9lzdQ38o'

st.set_option('deprecation.showPyplotGlobalUse', False)
p.set_options(p.OPT.URL, p.OPT.RESERVED, p.OPT.NUMBER)

# Create the authentication object
authenticate = tweepy.OAuthHandler(consumer_key, consumer_secret)

# Set the access token and access token secret
authenticate.set_access_token(access_token, access_token_secret)

# Creating the API object while passing in auth information
geo='-2.288137963036328,-80.08945470951159,50km'
api = tweepy.API(authenticate, wait_on_rate_limit=True)


n = 1000


def app():
    st.set_page_config(layout="centered", page_icon="💬", page_title="Tweets Indicadores", )
    st.title("💬 Indicadores de calidad  de servicios publicos")



    st.info('Electricidad, Agua, Transporte, Recolección y Alcantarillado')
    #st.title("Los servicios públicos analizados fueron: agua, luz, transporte y recolección de basura")
    # Select
    flowers = ["Seleccione una Opción", "Electricidad", "Agua", "Transporte", "Recolección", "Alcantarillado"]
    choice = st.selectbox("Seleccione el tipo de servicio que desea analizar", flowers)
    col1, col2 = st.columns(2)




    if(choice=="Seleccione una Opción"):
            image1 = Image.open('malecon.jpg')
            image1 = image1.resize((800, 400))
    if (choice == "Electricidad"):
            image1 = Image.open('luz.jpg')
            image1 = image1.resize((800, 410))

    if (choice == "Agua"):
            image1 = Image.open('agua.jpg')
            image1 = image1.resize((800, 400))

    if (choice == "Transporte"):
            image1 = Image.open('transporte.jpg')
            image1 = image1.resize((800, 400))

    if (choice == "Recolección"):
            image1 = Image.open('recoleccion.jpg')
            image1 = image1.resize((800, 400))

    if (choice == "Alcantarillado"):
            image1 = Image.open('alcantarillado.jpg')
            image1 = image1.resize((800, 400))
    st.image(image1)


    print(choice)

            # Usamos el modelo para hacer predicciones - Kevin
           # print("El porcentaje de fallas segun los Tweets", predicted_porcentaje_fallos)
           # mensaje = "El porcentaje de fallas segun los Tweets:" + str(predicted_porcentaje_fallos)
            #st.success(mensaje)


    if st.button("Analizar Servicio"):
        texto = "Analizando los tweets de los ultimos 7 días"
        st.success(texto)
        if(choice=="Seleccione una Opción"):
            texto = "No has seleccionado un servicio"
            st.warning(texto)
        else:




            def Show_Recent_Tweets():
                # Extract 3200 tweets from the twitter user
                with st.spinner('Analizando los Tweets...'):
                    posts = tweepy.Cursor(api.search_tweets, q=choice, include_rts=False,
                                          tweet_mode="extended", geocode=geo).items(n)
                    usuario = df_tweet(posts)
                return usuario

            df = Show_Recent_Tweets()
            #print(len(df))
            #print(df.head(10))

            # Parte Nueva
            def limpiar_tokenizar(texto):
                '''
                Esta función limpia y tokeniza el texto en palabras individuales.
                El orden en el que se va limpiando el texto no es arbitrario.
                El listado de signos de puntuación se ha obtenido de: print(string.punctuation)
                y re.escape(string.punctuation)
                '''

                # Se convierte todo el texto a minúsculas
                nuevo_texto = texto.lower()
                # Eliminación de páginas web (palabras que empiezan por "http")
                nuevo_texto = re.sub('http\S+', ' ', nuevo_texto)
                # Eliminación de signos de puntuación
                regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
                nuevo_texto = re.sub(regex, ' ', nuevo_texto)
                # Eliminación de números
                nuevo_texto = re.sub("\d+", ' ', nuevo_texto)
                # Eliminación de espacios en blanco múltiples
                nuevo_texto = re.sub("\\s+", ' ', nuevo_texto)
                # Tokenización por palabras individuales
                nuevo_texto = nuevo_texto.split(sep=' ')
                # Eliminación de tokens con una longitud < 2
                nuevo_texto = [token for token in nuevo_texto if len(token) > 1]

                return (nuevo_texto)

            stop_words = list(stopwords.words('spanish'))
            stop_words.remove("no")
            # Creación de la matriz tf-idf
            # ==============================================================================
            tfidf_vectorizador = TfidfVectorizer(
                tokenizer=limpiar_tokenizar,
                min_df=2,
                stop_words=stop_words
            )
            loaded_model = keras.models.load_model("TwitterServPublico.h5")
            X_test = df["Tweets"]

            tokenizer = Tokenizer(num_words=5029)
            tokenizer.fit_on_texts(X_test)

            X_test_rn = tokenizer.texts_to_sequences(X_test)

            maxlen = 30  ## Esto va acorde a una analisis de percentiles

            X_test_seq = pad_sequences(X_test_rn, padding='post', maxlen=maxlen)
            X_vector = tfidf_vectorizador.fit_transform(X_test)

            vocabulario_size = len(tfidf_vectorizador.get_feature_names())
            #vocabulario_size
            y_predict = loaded_model.predict(X_test_seq) > 0.5
            true_count = sum(y_predict)
            false_count=len(y_predict)-true_count
            print("Los True son" + str(true_count))
            print("Los False son" + str(len(y_predict)-true_count))
            print(y_predict[:5])

            df["fecha"] = df['created_at'].dt.strftime('%m/%d/%Y')
            df["anio"] = df['created_at'].dt.strftime('%Y')
            df["mesnombre"] = df['created_at'].dt.strftime('%b')
            df["mes"] = df['created_at'].dt.strftime('%m')

            df = df[df["anio"].astype(int) >= 2021].copy()
            st.success("¡LISTO!")

            #st.subheader("Estos son tus datos")





            st.subheader("Palabras más usadas")

            def gen_wordcloud():
                # word cloud visualization
                allWords = ' '.join([twts for twts in df['Tweets']])
                allWords = p.clean(allWords)
                wordCloud = WordCloud(width=700, height=500, random_state=21, max_font_size=110, stopwords=stop).generate(
                    allWords)
                plt.imshow(wordCloud, interpolation="bilinear")
                plt.axis('off')
                plt.savefig('WC.jpg')
                img = Image.open("WC.jpg")
                return img


            try:
                img = gen_wordcloud()
                st.image(img, width=700)
            except:
                st.write("Parece que el trabajo te ha tenido ocupado y no tenemos tweets !!")

            def grafico():
                # Pie chart, where the slices will be ordered and plotted counter-clockwise:
                labels = 'Quejas', 'Normal'

                total = true_count + false_count
                porTrue = (true_count / total) * 100
                num1 = int(porTrue)
                porFalse = (false_count / total) * 100
                num2 = int(porFalse)
                print(num1)
                print(num2)

                col1, col2, col3 = st.columns(3)
                col1.metric("Total de Tweets Analizados ", total, "")

                sizes = [num1, num2]
                explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                        shadow=True, startangle=90)
                ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

                return fig1
            try:
                img = grafico()
                st.pyplot(img)
            except:
                st.write("Parece que el trabajo te ha tenido ocupado y no tenemos tweets !!")


            st.subheader("Hashtag más utilizados")

            try:
                hashtags = df['Tweets'].apply(lambda x: pd.value_counts(re.findall('(#\w+)', x.lower()))) \
                    .sum(axis=0).to_frame().reset_index().sort_values(by=0, ascending=False)
                hashtags.columns = ['hashtag', 'occurences']
                fig = px.bar(hashtags, x='hashtag', y='occurences')
                st.plotly_chart(fig)
            except:
                st.write("No tenemos hashtag")





if __name__ == "__main__":
    app()