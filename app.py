import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BertTokenizer, BertForSequenceClassification
from streamlit_chat import message
import os
import random

# === Load BERT model ===
bert_model_path = "empaAI_bert_model"
bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)

# === Load GODEL chatbot ===
chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
chat_model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

# === Session state ===
if "past" not in st.session_state:
    st.session_state.past = []
if "generated" not in st.session_state:
    st.session_state.generated = []
if "labels" not in st.session_state:
    st.session_state.labels = []
if "confidences" not in st.session_state:
    st.session_state.confidences = []

# === Depression classifier ===
def detect_depression(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    bert_model.to("cpu")
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][label].item()
    return label, confidence

# === Chatbot reply ===
def chatbot_reply(user_input, label, context=""):
    instruction = "Respond empathetically and supportively." if label == 1 else "Respond cheerfully and positively."
    prompt = f"Instruction: {instruction} Context: {context} {user_input.strip()} Response:"

    input_ids = chat_tokenizer.encode(prompt, return_tensors="pt", truncation=True)
    output_ids = chat_model.generate(
        input_ids,
        max_length=100,
        do_sample=True,
        top_p=0.85,
        temperature=0.7,
        num_return_sequences=3,
        repetition_penalty=1.1
    )

    # Pick one randomly for diversity
    responses = [chat_tokenizer.decode(out, skip_special_tokens=True).strip() for out in output_ids]
    response = random.choice(responses)
    if "Response:" in response:
        response = response.split("Response:")[-1].strip()
    response = response.replace(">", "").strip()

    if not response:
        response = "Thanks for sharing. I'm here if you need me." if label == 1 else "That sounds great!"

    # Occasionally mix in human warmth
    if label == 1 and random.random() < 0.3:
        templates = [
            "Thank you for sharing this with me.",
            "I'm really sorry you're going through this.",
            "You are not alone — I'm here for you."
        ]
        response = random.choice(templates) + " " + response

    response += " 💙" if label == 1 else " 😊"
    return response

# === UI ===
st.title("💬 EmpaAI - Depression-Aware Chatbot")
st.markdown("This chatbot detects emotional tone and responds empathetically. It does **not** provide medical advice.")

user_input = st.text_input("You:", key="input")

if user_input:
    label, confidence = detect_depression(user_input)

    # 🔥 TEMPORARY FIX: force Happy if positive words found
    positive_keywords = ["happy", "amazing", "great", "love", "worth living", "excited", "grateful"]
    if any(word in user_input.lower() for word in positive_keywords):
        label = 0
        confidence = 1.0

    # Build short context from past 2 user inputs
    context = " ".join(st.session_state.past[:2][::-1]) if len(st.session_state.past) >= 2 else ""

    response = chatbot_reply(user_input, label, context)

    # Append chat history
    st.session_state.past.insert(0, user_input)
    st.session_state.generated.insert(0, response)
    st.session_state.labels.insert(0, label)
    st.session_state.confidences.insert(0, confidence)

    # Stronger NHS trigger: check length, confidence, and severe words
    severe_words = ["suicidal", "worthless", "can't go on", "no way out"]
    if label == 1 and confidence > 0.9 and len(user_input.split()) > 6 and any(word in user_input.lower() for word in severe_words):
        st.session_state.generated[0] += (
            "\n\n⚠️ **You're not alone.** Please consider talking to someone you trust "
            "or getting support from a professional.\n\n"
            "[🧠 Visit NHS Mental Health Services](https://www.nhs.uk/nhs-services/mental-health-services/)\n"
            "📞 Or call Samaritans UK at **116 123** (free, 24/7)"
        )

    # Save chat log
    try:
        os.makedirs("chat_logs", exist_ok=True)
        with open("chat_logs/chat_log.csv", "a", encoding="utf-8") as f:
            f.write(f'"{user_input}","{response}",{label},{confidence:.2f}\n')
    except Exception:
        pass

# === Chat history with confidence meter ===
if st.session_state.generated:
    for i in range(len(st.session_state.generated)):
        message(st.session_state.past[i], is_user=True, key=f"user_{i}")
        message(st.session_state.generated[i], key=f"bot_{i}")

        label = st.session_state.labels[i]
        confidence = st.session_state.confidences[i]
        label_str = "Depressed" if label == 1 else "Happy"
        st.markdown(f"**Prediction:** {label_str} | **Confidence:** {confidence:.2f}")
        st.progress(confidence)
        st.markdown("---")
