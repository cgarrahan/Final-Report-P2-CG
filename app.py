import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotnine as p9
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#import code needed for prediction
s = pd.read_csv("social_media_usage.csv")
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x
#copy to new dataset
ss = s.copy()
#make binary
ss["sm_li"]=clean_sm(s["web1h"])
ss["parent"]=clean_sm(s["par"])
ss["married"]=clean_sm(s["marital"])
ss["female"]=clean_sm(s["gender"])

#pull correct columns into a list for final dataset
selected_columns = ["sm_li", "income", "educ2", "parent", "married", "female", "age"]

# create a new df with the selected columns
ss = ss[selected_columns]

# replace values for income
ss['income'] = np.where(ss['income'] > 9, np.nan, ss['income'])

# replace values for age
ss['education'] = np.where(ss['educ2'] > 8, np.nan, ss['educ2'])

# replace values for age
ss['age'] = np.where(ss['age'] > 98, np.nan, ss['age'])

ss = ss.dropna()
#create variables for model
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=94526) # set for reproducibility

# Initialize algorithm 
lr = LogisticRegression(class_weight='balanced')
# Fit algorithm to training data
lr.fit(X_train, y_train)

# Make predictions using the model and the testing data
y_pred = lr.predict(X_test)

#will use predict data later based on the input

# Centered headers
st.markdown("<h1 style='text-align: center;'>Who are LinkedIn Users?</h1>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center;'><img src='LinkedIn.png' alt='LinkedIn Image'></div>",
    unsafe_allow_html=True
)

st.markdown("<h3 style='text-align: center;'>To determine who is likely a LinkedIn user, please enter demographic information below</h3>", unsafe_allow_html=True)

# Subheading
st.markdown("#### Income")

# Text with emphasis
st.write("*See Income level breakdown below*")

st.image("Income.png")
income = st.slider('Select a value', min_value=1, max_value=9, value=1)

st.markdown("#### Gender")
gender = st.radio("Select Female or Male", ["Female", "Male"])
if gender == "Female":
    gender1 = 1
else:
    gender1 = 0
st.markdown("#### Education")
st.image("Education.png")
education = st.slider('Select a value', min_value=1, max_value=8, value=1)
st.markdown("#### Parent")
parent = st.radio("Are you a Parent?", ["Yes", "No"])
if parent == "Yes":
    parent1 = 1
else:
    parent1 = 0
st.markdown("#### Married")
married = st.radio("Are you Married?", ["Yes", "No"])
if married == "Yes":
    married1 = 1
else:
    married1 = 0
st.markdown("#### Age")
age = st.number_input("Enter Age:", min_value=1, max_value=98, value=25)

# New data for predictions
predict_data = pd.DataFrame({
    "income": [income],  
    "education": [education],  
    "parent": [parent1],  
    "married": [married1], 
    "female": [gender1],  
    "age": [age] 
})

# Use model to make predictions
predict_data["sm_li"] = lr.predict(predict_data)

# Display predictions using Streamlit
if predict_data["sm_li"].iloc[0] == 1: 
    text = "LinkedIn User"
else:
    text = "Not a LinkedIn User"

st.markdown(f"# Result: {text}")