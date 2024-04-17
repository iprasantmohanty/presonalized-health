from dotenv import load_dotenv
load_dotenv() ## load all the environemnt variables

from utils import update_features,cont_update_features,find_best_sol,decode_results, MIN_VAL, MAX_VAL

import streamlit as st
import os
import sqlite3

import google.generativeai as genai
## Configure Genai Key

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#genai.configure(api_key=GOOGLE_API_KEY)

## Function To Load Google Gemini Model and provide queries as response

def get_gemini_response(question,prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content([prompt[0],question])
    return response.text

## Fucntion To retrieve query from the database

def read_sql_query(sql,db):
    conn=sqlite3.connect(db)
    cur=conn.cursor()
    cur.execute(sql)
    rows=cur.fetchall()
    conn.commit()
    conn.close()
    #for row in rows:
        #print(row)
    return rows

## Define Your Prompt
#features = ["Age", "Hypertension", "Heart Disease", "Gender", "Ever Married","Work Type","Residence_type", "smoking_status"]






prompt=[
    """
    You are an expert in converting English questions to SQL query!
    The SQL database has the name DISEASE and has the following columns - FEATURES,option1,option2,option3,option4,option5 
    \n\nFor example,\nExample 1 - Show me the features present?, 
    the SQL command will be something like this SELECT FEATURES FROM DISEASE ;
    \nExample 2 - Tell me all the features available to change?, 
    the SQL command will be something like this SELECT FEATURES FROM DISEASE ;
    also the sql code should not have ``` in beginning or end and sql word in output

    """


]
prompt1=[
    """
    You are an expert in converting English questions to SQL query!
    The SQL database has the name DISEASE and has the following columns - FEATURES,option1,option2,option3,option4,option5 
    \n\nFor example,\nExample 1 - features are Age, Gender
    the SQL command will be something like this SELECT * FROM DISEASE WHERE FEATURES LIKE '%' || LOWER('Age') || '%'
    OR FEATURES LIKE '%' || LOWER('Gender') || '%';

    \nExample 2 - f1, f2, f3?, 
    the SQL command will be something like this SELECT * FROM DISEASE WHERE FEATURES LIKE '%f1%' OR FEATURES LIKE '%f2%' OR FEATURES LIKE '%f3%';
    also there may be only part of each feature in there in question.
    And also the sql code should not have ``` in beginning or end and sql word in output

    """
]
## Streamlit App

st.set_page_config(page_title="🏥Personalized-Health Recommendation")
st.header("🏥Personalized Health Recommendation System😷")

question=st.text_input("Ask to show all the features: ",key="input")

submit=st.button("Show Features")

# if submit is clicked
if submit:
    try:
        query=get_gemini_response(question,prompt)
        print(query)
        response=read_sql_query(query,"multi_diseases.db")
        #st.text(f"SQL is {query}")
        st.subheader("The features are --")
        i=1
        for row in response:
            print(row)
            st.subheader(f"feature-{i} is {row[0]}")
            i=i+1
    except ValueError:
        st.error("Please enter a valid input")


question1=st.text_input("Among these, pick which are to make constant (fix) : ",key="input1")
submit1=st.button("Fix Features")

if submit1:
    try:
        query1=get_gemini_response(question1,prompt1)
        print(query1)
        #st.text(f"SQL is {query1}")
        response1=read_sql_query(query1,"multi_diseases.db")
        st.subheader("The options for all selected features are")
        i=1
        for row in response1:
            display_text = []
            for element in row:
                if element is not None:
                    display_text.append(str(element))
            
            if display_text:  # Check if there are any non-None elements to display
                print(display_text)
                if display_text[0].lower() == 'age':
                    st.subheader(f"feature-{i} is {display_text[0]}")
                else:
                    st.subheader(f"feature-{i} is {display_text[0]} having option(s) {display_text[1:]}")
                i += 1
    except ValueError:
        st.error("Please enter a valid input")


question2=st.text_input("Enter your option for these above features : ",key="input2")
submit2=st.button("Set Feature Options")
prompt2=[
    """
    You are expert in understanding context of the word.
    You need to convert the sentence in to dictionary as given in following example.
    For example - age = 33, gender is male (or m), ever married is yes
    it should return the dictonary as {'age':33,'gender':'male','ever married':'yes'}
    Another example - my age is 29, gender is F, residence type is urban, bmi = 23.5
    it should retuen the dictonary as {'age':29.,'gender':'female','residence type':'urban','bmi':23.5}
    it should return in the form of a dictonary
    """
]

if submit2:
    try:
        query2=get_gemini_response(question2,prompt2)
        feature_dict=eval(query2)
        print(feature_dict)
        #st.subheader("The Response is")
        #st.text(feature_dict)
        for feature, option in feature_dict.items():
            st.subheader(f"'{feature}' is set to '{option}'")
        feature_list=list(feature_dict.keys())
        MIN_VAL, MAX_VAL=update_features(feature_dict)
        print(feature_list)
        
    except ValueError:
        st.error("Please enter a valid input")

submit3=st.button("Show features having Continuous values")

if submit3:
    continuous_features = ['avg_glucose_level', 'bmi', 'Pregnancies', 'hypertension_cont', 'SkinThickness', 'Insulin',
                        'DiabetesPedigreeFunction']
    indices_to_edit = [3, 4, 15, 16, 17, 18, 19]  # Indices for which you want to edit min and max values

    for i, j in enumerate(indices_to_edit):
        st.subheader(f"Range of '{continuous_features[i]}' is [ {MIN_VAL[j]} --> {MAX_VAL[j]} ]")
    

question4=st.text_input("Among these feature ranges, enter your value for the feature : ",key="input4")
submit4=st.button("Save Features values")  

if submit4:
    try:
        query4=get_gemini_response(question4,prompt2)
        cont_feature_dict=eval(query4)
        print(cont_feature_dict)
        #st.subheader("The Response is")
        #st.text(cont_feature_dict)
        for feature, option in cont_feature_dict.items():
            st.subheader(f"'{feature}' is set to '{option}'")
        cont_feature_list=list(cont_feature_dict.keys())
        MIN_VAL, MAX_VAL=cont_update_features(cont_feature_dict)
        print(cont_feature_list)
        #st.write(MIN_VAL)
        #st.write(MAX_VAL)
    except ValueError:
        st.error("Please enter a valid input")

st.markdown("<h2>Enter number of counterfactuals needed:</h2>", unsafe_allow_html=True)
question5 = st.number_input("", key="input5")
#question5=st.text_input("Enter number of counterfactuals needed : ",key="input5")
submit5=st.button("Show all Counterfactuals") 


if submit5:
    no_cf=int(question5) #total number of counterfactuals needed
    all_best_solutions=[]
    fit_val=[]
    for i in range(no_cf):
        bs,fv=find_best_sol()
        all_best_solutions.append(bs)
        fit_val.append(fv)
    df=decode_results(all_best_solutions,fit_val)
    print(df)
    st.subheader(df)
    # Get the maximum value of the 'Heart Stroke' column
    max_heart_stroke = float(df['Heart Stroke'].max())
    #print(max_heart_stroke)
    #print(f"type = {type(max_heart_stroke)}")
    # Format and print the maximum probability as a percentage
    max_heart_stroke_percentage = max_heart_stroke * 100
    # Get the maximum value of the 'Diabetes' column
    max_diabetes = float(df['Diabetes'].max())
    # Format and print the maximum probability as a percentage
    max_diabetes_percentage = max_diabetes * 100
    st.subheader("Can adopt any one of the above counterfactuals.")
    st.subheader(f'Maximum probability of getting "Heart Stroke" is {max_heart_stroke_percentage:.4f}%')
    st.subheader(f'Maximum probability of getting "Diabetes" is {max_diabetes_percentage:.4f}%')


#st.subheader("for more details contact - prasantmohanty.r@gmail.com")
st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
           AI App created by @ <a href="" target="_blank">Prasant Kumar Mohanty</a> | Made with Google Gemini Pro 
        </div>
        """,
        unsafe_allow_html=True
    )
