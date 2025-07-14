import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect

# === CONFIG ===
# Replace with your Hugging Face Hub path or local folder
model_path = 'username/empaAI-bert-model'  # e.g., 'myuser/empaAI-bert-model'

# === Load model + tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# === Streamlit app ===
st.title("💬 EmpaAI - Depression Screening Chatbot")
st.markdown("This chatbot predicts emotional tone as **Sad** or **Happy**. It does **not** provide medical advice.")

user_input = st.text_input("You:")

if user_input:
    try:
        if detect(user_input) != 'en':
            st.warning("⚠️ Please enter text in English.")
        else:
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                label = torch.argmax(probs, dim=1).item()
                confidence = probs[0][label].item()

            label_map = {0: 'Happy', 1: 'Sad'}
            emoji_map = {0: '😊', 1: '💙'}
            st.success(f"**Prediction:** {label_map[label]} {emoji_map[label]} \n\n**Confidence:** {confidence:.2%}")

            if label == 1 and confidence > 0.85:
                st.warning(
                    "⚠️ You're not alone. Please consider talking to someone you trust.\n\n"
                    "[🧠 NHS Mental Health Services](https://www.nhs.uk/nhs-services/mental-health-services/)\n"
                    "📞 Samaritans UK: **116 123** (free, 24/7)"
                )
    except Exception as e:
        st.error(f"Error: {str(e)}")
