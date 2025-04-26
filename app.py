import streamlit as st
import tensorflow as tf
import pickle
import numpy as np

# Loading the saved model (ensure the correct method is used for loading)
model_path = "sarcasm_detection_model"  # Path to your saved model
model = tf.saved_model.load(model_path)  # Load model directly (if TFSMLayer causes issues)

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Label map for predictions
label_map = {0: 'Genuine', 1: 'Sarcastic'}

# Streamlit app interface
st.title("🤖 Sarcasm Detection App")
st.write("Enter a sentence below and I'll predict if it's sarcastic or genuine!")

# Text input from user
user_input = st.text_area("Enter your text here:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        # Preprocess input using tokenizer
        encoding = tokenizer(
            [user_input],
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='tf'
        )

        # Prepare the input data for prediction
        inputs = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding.get('token_type_ids', None)  # Add token_type_ids if available
        }

        # Make prediction using the model
        predictions = model(**inputs)  # Use the correct function to call the model

        # Print predictions to inspect the structure
        st.write(predictions)

        # Access logits if predictions are in a tuple/list
        logits = predictions[0] if isinstance(predictions, (tuple, list)) else predictions['logits']
        
        # Get the predicted label (assuming a classification task with output size of 2)
        predicted_label = np.argmax(logits, axis=1)[0]  # Access logits from the model output
        confidence = np.max(logits)

        # Display the result
        st.subheader(f"Prediction: {label_map[predicted_label]}")
        st.caption(f"Confidence: {confidence:.2f}")
