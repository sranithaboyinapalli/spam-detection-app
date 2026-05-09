
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

data = pd.read_csv("spam mail.csv")


data['Category'] = data['Category'].replace(
    ['ham', 'spam'],
    ['Not Spam', 'Spam']
)


data.drop_duplicates(inplace=True)


mess = data['Masseges']
cat = data['Category']


mess_train, mess_test, cat_train, cat_test = train_test_split(
    mess, cat, test_size=0.2, random_state=42
)


cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)


model = MultinomialNB()
model.fit(features, cat_train)


def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result[0]


st.title("📩 Spam Message Detector")

input_mess = st.text_area("Enter Message")

if st.button("Check Message"):
    output = predict(input_mess)

    if output == "Spam":
        st.error("🚫 This message is Spam")
    else:
        st.success("✅ This message is Not Spam")