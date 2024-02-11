import os
import time
import joblib
import streamlit as st
from banglaspeech2text import Speech2Text
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize speech to text
available_models = Speech2Text.list_models()
stt = Speech2Text("base", cache_path="cache", use_gpu=True)

new_chat_id = f'{time.time()}'
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'

# Create a data/ folder if it doesn't already exist
if not os.path.exists('data/'):
    os.mkdir('data/')

# Load past chats (if available)
try:
    past_chats = joblib.load('data/past_chats_list')
except FileNotFoundError:
    past_chats = {}

# Sidebar for past chats
with st.sidebar:
    st.write('# Past Chats')
    if 'chat_id' not in st.session_state:
        st.session_state['chat_id'] = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'New Chat'),
        )
    else:
        st.session_state['chat_id'] = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id, st.session_state['chat_id']] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state['chat_id'] else st.session_state['chat_title']),
        )
    st.session_state['chat_title'] = f'ChatSession-{st.session_state["chat_id"]}'

st.write('# Chat with Gemini')

# Initialize or load chat history
try:
    st.session_state['messages'] = joblib.load(f'data/{st.session_state["chat_id"]}-st_messages')
    st.session_state['gemini_history'] = joblib.load(f'data/{st.session_state["chat_id"]}-gemini_messages')
except FileNotFoundError:
    st.session_state['messages'] = []
    st.session_state['gemini_history'] = []

st.session_state['model'] = genai.GenerativeModel('gemini-pro')
st.session_state['chat'] = st.session_state['model'].start_chat(history=st.session_state['gemini_history'])

# Display previous messages
for message in st.session_state['messages']:
    with st.chat_message(name=message['role'], avatar=message.get('avatar')):
        st.markdown(message['content'])

# Audio input for transcription
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
if uploaded_file is not None:
    audio_path = "temp_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    text = stt.recognize(audio_path)
    prompt = f"the bengali string that is given here, was transcribed from audio to text, it has errors here and there and no punctuations. Read it and try to understand what it is trying to say then rewrite it properly in english (Don't change valuable numbers or medical details in the process): {text}"

    # User message display
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state['messages'].append({'role': 'user', 'content': prompt})

    # Send message to AI and display response
    response = st.session_state['chat'].send_message(prompt, stream=True)
    with st.chat_message(name=MODEL_ROLE, avatar=AI_AVATAR_ICON):
        message_placeholder = st.empty()
        full_response = ''
        for chunk in response:
            for ch in chunk.text.split(' '):
                full_response += ch + ' '
                time.sleep(0.05)
                message_placeholder.write(full_response + '▌')
        message_placeholder.write(full_response)

    # Update chat history
    st.session_state['messages'].append({'role': MODEL_ROLE, 'content': st.session_state['chat'].history[-1].parts[0].text, 'avatar': AI_AVATAR_ICON})
    st.session_state['gemini_history'] = st.session_state['chat'].history
    joblib.dump(st.session_state['messages'], f'data/{st.session_state["chat_id"]}-st_messages')
    joblib.dump(st.session_state['gemini_history'], f'data/{st.session_state["chat_id"]}-gemini_messages')

    # Update past chats
    if st.session_state['chat_id'] not in past_chats.keys():
        past_chats[st.session_state['chat_id']] = st.session_state['chat_title']
        joblib.dump(past_chats, 'data/past_chats_list')
