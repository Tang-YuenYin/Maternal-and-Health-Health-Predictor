import streamlit as st
from PIL import Image

#display texts 
st.write("Hello, let's learn how to build a streamlit app together")

st.title("this is the app title :baby:")       #add title
st.markdown("this is the markdown")     #add markdown
st.header("this is the header")         #add header
st.subheader("this is the subheader")   #add sub-header
st.caption("this is the caption")       #add caption
st.code("x=2021")                       #set a code
st.latex(r'a + a r^1 + a r^2 + a r^3')  #display mathematical expressions as LaTeX

st.subheader("Images: ")
#Load local image from file
image=Image.open(r"Image_audio_video\images.jpeg")
photo_img=Image.open(r"Image_audio_video\photo.jpeg")
#Display the image
st.image(image)

st.image(
    photo_img,
    caption="Image of photograph",
    width=300,
    channels="RGB")


st.subheader("Audio: ")
st.audio(r"Image_audio_video\audio.mp3")      #add audio

st.subheader("Video: ")
st.video(r"Image_audio_video\video.mp4")      #add video


st.checkbox('yes')
st.button('Click')
st.radio('Pick your gender',['Male','Female'])
st.selectbox('Pick your gender',['Male','Female'])
st.multiselect('choose a planet',['Jupiter', 'Mars', 'neptune'])
st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
st.slider('Pick a number', 0,50)


st.number_input('Pick a number', 0,10)
st.text_input('Email address')
st.date_input('Travelling date')
st.time_input('School time')
st.text_area('Description')
st.file_uploader('Upload a photo')
st.color_picker('Choose your favorite color')