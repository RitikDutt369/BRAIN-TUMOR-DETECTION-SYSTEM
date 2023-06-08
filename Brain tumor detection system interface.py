'''st.markdown(
    """
    <style>
    body {
    background-image: url("https://www.google.com/imgres?imgurl=https%3A%2F%2Ft3.ftcdn.net%2Fjpg%2F01%2F91%2F17%2F56%2F360_F_191175671_9mnsD02RpKNoYGccVrglSxzEzyZxuwyt.jpg&tbnid=36PCPfo6OzrdnM&vet=12ahUKEwiat6aok-X-AhVo33MBHZ3XDh8QMygSegUIARD8AQ..i&imgrefurl=https%3A%2F%2Fstock.adobe.com%2Fsearch%3Fk%3Dbrain%2Bcancer%2Bribbon&docid=1rZPLIkx2qrLAM&w=587&h=360&q=webpage%20background%20images%20%2F%20brain%20tumor&ved=2ahUKEwiat6aok-X-AhVo33MBHZ3XDh8QMygSegUIARD8AQ");
    background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True,
)'''


import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import streamlit.components.v1 as components

st.set_page_config(page_title = "Brain Tumor Detection System",
                  page_icon=":hospital:",
                  layout = "centered")

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title(":brain: :blue[Brain Tumor Detection System]")
st.caption(':blue[_Devoloped By_ : RITIK DUTT SHARMA, VIBHA JAISWAL, DEVANSHU PANWAR, K.M MANU, HARSHIT] :blue[Under Guidance Of _Prof._ ASHWINI KUMAR UPADHYAY]')
st.markdown("---")
st.subheader(":blue[Please Upload Brain MRI image for Tumor Detection]")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('braintumor10epochs.h5')
    return model

with st.spinner('Loading Model into Memory....'):
    model = load_model()

classes=[':red[Urgent Diagnosis is recommended, Higher Expectency of Brain Tumor]',':green[No Worries, Your Brain is Healthy]']
st.markdown("---")

#-HIDE STREAMLIT STYLE
hide_st_style = """
                <style>
                #MainMenu{visibility:hidden;}
                footer{visibility:hidden;}
                header{visibility:hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)


def scale(image):
    image = tf.cast(image, tf.float32)
    image /= 255.0

    return tf.image.resize(image,[244,244])

def decode_img(image):
    img = tf.image.decode_jpeg(image, channels=3)
    img = scale(img)
    return np.expand_dims(img, axis=0)

file = st.file_uploader(":blue[Please choose a file]", type=['jpg'])
st.markdown("---")
if file is None:
    st.caption(':blue[Please upload an MRI image file in jpg/jpeg format]')
    #st.text(":blue[Please upload an MRI image file in jpg/jpeg format]")

else:
    content = file.getvalue()
    st.write("Pridicted Class :")
    with st.spinner('classifying....'):
        label = np.argmax(model.predict(decode_img(content)),axis=1)
        st.write(classes[label[0]])
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classifying MRI Image', use_column_width=True, output_format="auto")
