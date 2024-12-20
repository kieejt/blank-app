import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
# Load the .pkl files
@st.cache_resource
def load_models():
    random_forest_model = joblib.load("random_forest_model.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    lsa_model = joblib.load("lda_model.pkl")
    return random_forest_model, tfidf_vectorizer, lsa_model

random_forest_model, tfidf_vectorizer, lsa_model = load_models()

stopwords = [
    'và', 'là', 'của', 'cho', 'có', 'như', 'với', 'từ', 'để', 'đến', 'một',
    'nhưng', 'cũng', 'thì', 'này', 'đó', 'được', 'trên', 'dưới', 'khi', 'ở',
    'nơi', 'vì', 'sao', 'cùng', 'rằng', 'ra', 'vẫn', 'đang', 'hãy', 'đã', 'nếu'
    # Bạn có thể mở rộng danh sách này tùy ý
]
# Define text-cleaning function
def clean_text(text):
    # 1. Loại bỏ ký tự đặc biệt
    text = re.sub(r'[^\w\s]', ' ', text)  # Thay thế ký tự đặc biệt bằng khoảng trắng
    text = re.sub(r'\d+', ' ', text)  # Loại bỏ chữ số
    text = re.sub(r'\s+', ' ', text).strip()  # Loại bỏ khoảng trắng thừa

    # 2. Chuyển về chữ thường
    text = text.lower()

    # 3. Loại bỏ stopword
    words = text.split()
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)

# Streamlit app interface
st.title("Emotion Classification for Vietnamese Song Lyrics")

st.markdown("""
This app predicts the **emotion** of a Vietnamese song based on its lyrics.  
Enter the lyrics of a song below and click **Predict** to see the result.
""")

# Input field for song lyrics
input_text = st.text_area("Enter song lyrics:", height=200)

df_test = pd.DataFrame([[1001,'Tên',input_text]], columns=['index','name_song', 'lyric'])

df_test['lyric'] = df_test['lyric'].astype(str).apply(clean_text)
df_test.columns = ['index','name_song','lyric']

if st.button("Predict"):
    if input_text.strip() == "":
        st.error("Please enter some text before predicting!")
    else:
        
        
        # Transform text using TF-IDF and LSA
        tfidf_features = tfidf_vectorizer.transform(df_test['lyric'])
        lsa_features = lsa_model.transform(tfidf_features)
        
        # Predict emotion
        prediction = random_forest_model.predict(lsa_features)
        
        # Display the result
        st.success(f"The predicted emotion is: **{prediction[0]}**")
