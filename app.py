from PIL import Image
import streamlit as st

from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from keras.models import load_model





st.subheader("Image Classifier ")

image_file = st.file_uploader("Airplane Automobile Bird Cat Deerdog Frog Horseship Truck", type=["png","jpg","jpeg"])

if image_file is not None:

	st.image(Image.open(image_file),width=250)



labels = '''Airplane Automobile Bird Cat Deerdog Frog Horseship Truck'''.split()



def load_image(filename):
	
	img = image.load_img(image_file, target_size=(32, 32))
	
	img = img_to_array(img)
	
	img = img.reshape(1, 32, 32, 3)
	
	img = img.astype('float32')
	img = img / 255.0
	return img


	
  


if st.button('Predict'):
    img = load_image(image_file)

    model = load_model('image_model2.h5')

    result = labels[model.predict(img).argmax()]
    # st.write(result)
    st.subheader(result, anchor=None)