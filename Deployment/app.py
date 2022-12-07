import streamlit as st
import numpy as np

from tensorflow.keras.models import load_model
from pickle import load

model = load_model('../Stock_Prediction/BIST100/cnn_lstm_twitter.h5')

in_scaler = load(open('../Stock_Prediction/BIST100/in_scaler.pkl', 'rb'))
out_scaler = load(open('../Stock_Prediction/BIST100/out_scaler.pkl', 'rb'))

st.title("Stock Price Prediction")

with st.form(key='input_form'):
    col1, col2, col3 = st.columns([2, 2, 2])
    col4, col5, col6 = st.columns([2, 2, 2])
    col7, col8, col9 = st.columns([2, 2, 2])
    col10, col11, col12 = st.columns([2, 2, 2])
    col13, col14, col15 = st.columns([2, 2, 2])

    with col1:
        i0 = st.number_input("Closing day t-1", value=3847.62)
    with col2:
        i1 = st.number_input("Opening day t-1", value=3641.40)
    with col3:
        i2 = st.number_input("Average sentiment t-1", value=0.529757)
    with col4:
        i3 = st.number_input("Closing day t-2", value=3626.96)
    with col5:
        i4 = st.number_input("Opening day t-2", value=3584.94)
    with col6:
        i5 = st.number_input("Average sentiment t-2", value=0.479313)
    with col7:
        i6 = st.number_input("Closing day t-3", value=3553.43)
    with col8:
        i7 = st.number_input("Opening day t-3", value=3529.78)
    with col9:
        i8 = st.number_input("Average sentiment t-3", value=0.497640)
    with col10:
        i9 = st.number_input("Closing day t-4", value=3517.75)
    with col11:
        i10 = st.number_input("Opening day t-4", value=3576.84)
    with col12:
        i11 = st.number_input("Average sentiment t-4", value=0.488859)
    with col13:
        i12 = st.number_input("Closing day t-5", value=3571.55)
    with col14:
        i13 = st.number_input("Opening day t-5", value=3590.26)
    with col15:
        i14 = st.number_input("Average sentiment t-5", value=0.451031)

    submit = st.form_submit_button(label='Calculate')

if submit:
    test = np.array([[i0, i1, i2],
                     [i3, i4, i5],
                     [i6, i7, i8],
                     [i9, i10, i11],
                     [i12, i13, i14]])
    test = in_scaler.transform(test).reshape(1, test.shape[0], test.shape[1])
    test_res = model.predict(test)
    result = out_scaler.inverse_transform(test_res.reshape(1, 1))[0, 0]
    st.text(f'Result: {result}')
