import streamlit as st
import pandas as pd
import sklearn
import pickle
import plotly.graph_objects as go
from PIL import Image

# loading the trained model
#pickle_in = open('RF_burnoutPredictor.pkl', 'rb') 
pickle_in = open('DT_burnoutPredictor.pkl', 'rb') 
model = pickle.load(pickle_in)
 
#@st.cache()

# defining the function which will make the prediction using the data which the user inputs 
def predict(Designation, Resource_Allocation, Mental_Fatigue_Score,Gender, Company_Type, WFH_Setup_Available):   
 
    # Pre-processing user input - one-hot encoding   
    if Gender == "Male":
        Gender_Male = 1
        Gender_Female = 0
    else:
        Gender_Male = 0
        Gender_Female = 1
 
    if Company_Type == "Product":
        Company_Type_Product = 1
        Company_Type_Service = 0
    else:
        Company_Type_Product = 0
        Company_Type_Service = 1
        
    if WFH_Setup_Available == "No":
        WFH_Setup_Available_No = 1
        WFH_Setup_Available_Yes = 0
    else:
        WFH_Setup_Available_No = 0
        WFH_Setup_Available_Yes = 1

    # Making predictions 
    BurnoutPrediction = model.predict([[Designation, 	Resource_Allocation,Mental_Fatigue_Score, Gender_Female, Gender_Male, Company_Type_Product, Company_Type_Service, WFH_Setup_Available_No, WFH_Setup_Available_Yes]])
    return BurnoutPrediction   

# App implementation ------------------------------------------------------------------------------------

st.write("""
# Simple Burnout Prediction App""")

image = Image.open('image.png')
st.image(image)

st.write("""
This simple tool predicts your **burnout** level by employing a pre-trained machine learning model. Please answer the question on the sidebar to receive a prediction."""
)



st.sidebar.header('Input Parameters')

def user_input_features():
    Designation = st.sidebar.slider('What is the designation of your role in your current organization? (0 = entry level, 5 = most senior level)', min_value=0.0, max_value=5.0, value=2.0, step=1.0)
    
    Resource_Allocation = st.sidebar.slider('How many daily working hours are required of you?',min_value=0.0, max_value=10.0, value=8.0, step=1.0)
    
    Mental_Fatigue_Score = st.sidebar.slider('How would you score your current level of mental fatigue? (0 = no fatigue, 10 = complete mental exhaustion)', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    
    Gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    
    Company_Type = st.sidebar.selectbox('What best describes your current working sector?', ('Service', 'Product'))
    
    WFH_Setup_Available = st.sidebar.selectbox('Do you feel you have an adequate remote working setup?', ('Yes', 'No'))
     
    return Designation, Resource_Allocation, Mental_Fatigue_Score, Gender,Company_Type, WFH_Setup_Available
    
Designation, Resource_Allocation, Mental_Fatigue_Score, Gender,Company_Type, WFH_Setup_Available = user_input_features()

predictions = predict(Designation, Resource_Allocation, Mental_Fatigue_Score, Gender,Company_Type, WFH_Setup_Available)

pred = float(predictions)*10

st.subheader('Burnout prediction:')

fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = pred,
    mode = "gauge+number",
    gauge = {'axis': {'range': [None, 10.0]},
    	     'bar': {'color': "steelblue"},
             'steps' : [
                 {'range': [0.0, 3.0], 'color': "lightgreen"},
                 {'range': [3.0, 6.0], 'color': "yellow"},
                 {'range': [6.0, 9.0], 'color': "orange"},
                 {'range': [9.0, 10.0], 'color': "red"}]}))


st.plotly_chart(fig, use_container_width=True)

if pred <= 3.0: st.write('Your burnout level is in the **low** range :relieved:')

elif pred > 3.0 and pred <= 6.0: st.write('Your burnout level is in the **medium** range :persevere:')

elif pred > 6.0 and pred <= 9.0: st.write('Your burnout level is in the **high** range :weary:')

elif pred > 9.0: st.write('Your burnout level is in the **very high** range :tired_face:')
	

