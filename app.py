from unittest import result
from PIL import Image
import streamlit as st

from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from keras.models import load_model





st.subheader("Upload Image ......")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:

	st.image(Image.open(image_file),width=250)



labels = '''Airplane Automobile Bird Cat Deerdog Frog Horseship Truck'''.split()


# load and prepare the image
def load_image(filename):
	# load the image
	img = image.load_img(image_file, target_size=(32, 32))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 32, 32, 3)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img


	
  


if st.button('Predict'):
    img = load_image(image_file)
    # load model
    model = load_model('image_model2.h5')
    # predict the class
    result = labels[model.predict(img).argmax()]
    # st.write(result)
    st.subheader(result, anchor=None)