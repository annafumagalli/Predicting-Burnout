import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

# loading the trained model
pickle_in = open('RF_burnoutPredictor.pkl', 'rb') 
model = pickle.load(pickle_in)
 
#@st.cache()

# defining the function which will make the prediction using the data which the user inputs 
def predict(Designation, Resource_Allocation, Mental_Fatigue_Score,
       Gender, Company_Type, WFH_Setup_Available):   
 
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
    BurnoutPrediction = model.predict( 
        [[Designation, Resource_Allocation, Mental_Fatigue_Score,
       Gender_Female, Gender_Male, Company_Type_Product,
       Company_Type_Service, WFH_Setup_Available_No,
       WFH_Setup_Available_Yes]])
       
    return BurnoutPrediction   

# App implementation ------------------------------------------------------------------------------------

st.write("""
# Simple Burnout Prediction App
This app predicts your **burnout** level by employing a pre-trained machine learning model. Please answer the question in the sidebar to receive a prediction.
""")



st.sidebar.header('User Input Parameters')

def user_input_features():
    Designation = st.sidebar.slider('How would you rate the designation of your role in your current organization?', min_value=0.0, max_value=5.0, value=2.0, step=1.0)
    
    Resource_Allocation = st.sidebar.slider('How would you rate the amount of resources allocated to you/required of you? (i.e. number of daily working hours)',min_value=0.0, max_value=10.0, value=8.0, step=1.0)
    
    Mental_Fatigue_Score = st.sidebar.slider('How would you score your average mental fatigue level based on your current workload? (this should take into account your workplace and personal resposibilities and pressures)', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    
    Gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    
    Company_Type = st.sidebar.selectbox('What best describes your current working sector?', ('Service', 'Product'))
    
    WFH_Setup_Available = st.sidebar.selectbox('Do you feel you have an adequate remote working setup?', ('Yes', 'No'))
     
    return Designation, Resource_Allocation, Mental_Fatigue_Score, Gender,Company_Type, WFH_Setup_Available
    
Designation, Resource_Allocation, Mental_Fatigue_Score, Gender,Company_Type, WFH_Setup_Available = user_input_features()

data = {'Designation': Designation, 
	'Resource Allocation': Resource_Allocation, 
	'Mental Fatigue Score': Mental_Fatigue_Score, 
	'Gender': Gender,
	'Company Type': Company_Type, 
	'WFH Setup Available': WFH_Setup_Available}
	
data = pd.DataFrame(data, index=[0])	

st.subheader('User Input parameters')
st.write(data)

predictions = predict(Designation, Resource_Allocation, Mental_Fatigue_Score, Gender,Company_Type, WFH_Setup_Available)

st.subheader('Burnout prediction:')


fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = float(predictions),
    mode = "gauge+number",
    title = {'text': "Burnout level"},
    gauge = {'axis': {'range': [None, 1.0]},
    	     'bar': {'color': "steelblue"},
             'steps' : [
                 {'range': [0, 0.3], 'color': "lightgreen"},
                 {'range': [0.3, 0.6], 'color': "yellow"},
                 {'range': [0.6, 0.9], 'color': "orange"},
                 {'range': [0.9, 1.0], 'color': "red"}]}))

st.plotly_chart(fig, use_container_width=True)
