import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('project.pkl', 'rb'))

st.title('Will the person make a purchase or not')

PageValues = st.slider("PageValues",0.00,361.76)
ExitRates = st.slider("ExitRates",0.000,0.200)
ProductRelated_Duration = st.slider("ProductRelated_Duration",0.00,63973.52)
ProductRelated = st.slider("ProductRelated",0,705)


def predict():
    float_features = [float(x) for x in [PageValues, ExitRates, ProductRelated_Duration, ProductRelated]]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    label = prediction[0]
    
    print(type(label))
    print(label)

    st.success('The costomer will make a purchase ' + str(label) + ' :thumbsup:')
    
trigger = st.button('Predict', on_click=predict)

