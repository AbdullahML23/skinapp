import numpy as np
import cv2 
import streamlit as st
from keras.utils import load_img, img_to_array
from keras.models import load_model

model=load_model('SkinModel.h5')
st.title("Skin Disease Diagnostic App")
class_labels=[
    "Acne and Rosacea",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema",
    "Exanthems and Drug Eruptions",
    "Herpes HPV and other STDs",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis Photos",
    "Warts Molluscum and other Viral Infections"
]
uploader = st.file_uploader('Select image', type=('jpg', 'png'))
if uploader is not None:
    image = load_img(uploader,target_size=(224,224),color_mode='rgb')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image.resize((224, 224)))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    st.subheader("Prediction:")
    d_predicted=class_labels[class_index]
    st.write(f'Your Skin Disease is {d_predicted}')
    st.write(prediction)
    

    

    
    
    
    
    
    












    
  
