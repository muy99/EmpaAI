import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BertTokenizer, BertForSequenceClassification
from streamlit_chat import message
import os

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

# === Depression classifier ===
def detect_depression(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cpu")
    bert_model.to("cpu")
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][label].item()
    return label, confidence

# === Chatbot reply ===
def chatbot_reply(user_input, label):
    if label == 1:
        prompt = f"Instruction: Respond empathetically and supportively. Context: {user_input.strip()} Response:"
    else:
        prompt = f"Instruction: Respond cheerfully and positively. Context: {user_input.strip()} Response:"

    input_ids = chat_tokenizer.encode(prompt, return_tensors="pt", truncation=True)
    output_ids = chat_model.generate(
        input_ids,
        max_length=100,
        do_sample=True,
        top_p=0.85,
        temperature=0.7,
        num_return_sequences=1,
        repetition_penalty=1.1
    )

    response = chat_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if "Response:" in response:
        parts = response.split("Response:")
        response = parts[-1].strip() if len(parts) > 1 else response.strip()
    response = response.replace(">", "").strip()

    if not response:
        response = "Thanks for sharing. I'm here if you need me." if label == 1 else "That sounds great!"

    response += " 💙" if label == 1 else " 😊"
    return response

# === UI ===
st.title("💬 EmpaAI - Depression-Aware Chatbot")
st.markdown("This chatbot detects emotional tone and responds empathetically. It does **not** provide medical advice.")

user_input = st.text_input("You:", key="input")

if user_input:
    label, confidence = detect_depression(user_input)
    response = chatbot_reply(user_input, label)

   if label == 1 and confidence > 0.85:
    response += (
        "\n\n⚠️ **You're not alone.** Please consider talking to someone you trust "
        "or getting support from a professional.\n\n"
        "[🧠 Visit NHS Mental Health Services](https://www.nhs.uk/nhs-services/mental-health-services/)"
    )


    st.session_state.past.insert(0, user_input)
    st.session_state.generated.insert(0, response)

    try:
        os.makedirs("chat_logs", exist_ok=True)
        with open("chat_logs/chat_log.csv", "a", encoding="utf-8") as f:
            f.write(f'"{user_input}","{response}",{label},{confidence:.2f}\n')
    except Exception:
        pass

# === Chat history (most recent on top) ===
if st.session_state.generated:
    for i in range(len(st.session_state.generated)):
        message(st.session_state.past[i], is_user=True, key=f"user_{i}")
        message(st.session_state.generated[i], key=f"bot_{i}")
