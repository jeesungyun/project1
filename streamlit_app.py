import os
import openai
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# API 키 설정 (Streamlit Secrets 사용)
openai.api_key = st.secrets["general"]["OPENAI_API_KEY"]

# 세션 상태 초기화
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = []
if 'current_user' not in st.session_state:
    st.session_state.current_user = 1
if 'start_button' not in st.session_state:
    st.session_state.start_button = False
if 'num_users' not in st.session_state:
    st.session_state.num_users = 1
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 챗봇 응답 생성 함수
def get_chatbot_response(prompt, max_retries=5):
    for retry in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message['content'].strip()
        except openai.error.RateLimitError as e:
            if retry < max_retries - 1:
                time.sleep(20)  # 재시도 전 대기 시간 설정
            else:
                st.error(f"Rate limit error: {e}")
                return None
        except openai.error.OpenAIError as e:
            st.error(f"OpenAI API error: {e}")
            return None

# 사용자 수 입력 받기
if not st.session_state.start_button:
    st.session_state.num_users = st.number_input("총 몇명이 고를건가요?", min_value=1, max_value=10, step=1)
    if st.button("시작"):
        st.session_state.start_button = True

if st.session_state.start_button:
    user_num = st.session_state.current_user
    st.header(f"User {user_num}")

    # 사용자 주관식 입력 받기
    st.write("선호하지 않는 음식을 입력하세요. 예시:")
    st.write("카테고리(한식/일식/중식/양식/동남아식 등), 기름진 음식, 고칼로리, 매운 음식 등")
    user_input = st.text_input("먹기 싫은 음식을 입력하세요:", key=f"user_input_{user_num}")
    if user_input:
        st.session_state.chat_history.append(f"User {user_num}: {user_input}")
        chatbot_response = get_chatbot_response(user_input)
        st.session_state.chat_history.append(f"Chatbot: {chatbot_response}")

        st.text_area("대화 기록", value="\n".join(st.session_state.chat_history), height=200, key=f"chat_history_{user_num}")

    if st.button(f"User {user_num} 선택 완료"):
        st.session_state.user_inputs.append(user_input)
        if st.session_state.current_user < st.session_state.num_users:
            st.session_state.current_user += 1
        else:
            st.session_state.start_button = False

if len(st.session_state.user_inputs) == st.session_state.num_users:
    st.write("모든 사용자가 선택을 완료했습니다.")

    # 사용자 입력을 임베딩 벡터로 변환하는 함수 (재시도 로직 추가)
    def get_embedding(text, max_retries=5):
        for retry in range(max_retries):
            try:
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=[text]
                )
                return response['data'][0]['embedding']
            except openai.error.RateLimitError as e:
                if retry < max_retries - 1:
                    time.sleep(20)  # 재시도 전 대기 시간 설정
                else:
                    st.error(f"Rate limit error: {e}")
                    return None
            except openai.error.OpenAIError as e:
                st.error(f"OpenAI API error: {e}")
                return None

    user_embeddings = [get_embedding(user_input) for user_input in st.session_state.user_inputs]
    user_embeddings = [embedding for embedding in user_embeddings if embedding is not None]  # None 값 제거

    # 메뉴 데이터 (예시로 몇 가지 메뉴만 포함)
    menu_db = {
        "돈까스": "일식, 기름진 음식, 비국물 요리, 고칼로리, 안매운거, 짠거, 안단거",
        "스시": "일식, 담백한 음식, 비국물 요리, 저칼로리, 안매운거, 안짠거, 단거"
    }

    # 메뉴 설명 임베딩 생성
    menu_embeddings = {menu: get_embedding(description) for menu, description in menu_db.items()}
    menu_embeddings = {menu: embedding for menu, embedding in menu_embeddings.items() if embedding is not None}  # None 값 제거

    # 코사인 유사도를 통해 가장 유사도가 낮은 메뉴 찾기
    def find_least_similar(user_embeddings, menu_embeddings):
        combined_similarities = np.zeros(len(menu_embeddings))
        for user_embedding in user_embeddings:
            similarities = np.array([cosine_similarity([user_embedding], [embedding])[0][0] for embedding in menu_embeddings.values()])
            combined_similarities += similarities
        least_similar_menu = list(menu_embeddings.keys())[np.argmin(combined_similarities)]
        return least_similar_menu

    # 유사도가 가장 낮은 메뉴 찾기
    least_similar_menu = find_least_similar(user_embeddings, menu_embeddings)

    st.write("임베딩을 사용한 유사도가 가장 낮은 메뉴 (추천 메뉴):", least_similar_menu)
