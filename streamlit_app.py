import os
import openai
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai.error import RateLimitError
import time

# API 키 설정
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 세션 상태 초기화
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = []
if 'current_user' not in st.session_state:
    st.session_state.current_user = 1
if 'start_button' not in st.session_state:
    st.session_state.start_button = False
if 'num_users' not in st.session_state:
    st.session_state.num_users = 1

# 사용자 수 입력 받기
if not st.session_state.start_button:
    st.session_state.num_users = st.number_input("총 몇명이 고를건가요?", min_value=1, max_value=10, step=1)
    if st.button("시작"):
        st.session_state.start_button = True

if st.session_state.start_button:
    user_num = st.session_state.current_user
    st.header(f"User {user_num}")

    # 음식 카테고리 선택 (복수 선택 가능)
    categories = st.multiselect("음식 카테고리", ["한식", "중식", "일식", "양식", "동남아"], key=f"categories_{user_num}")

    # 나머지 질문들
    style = st.radio("음식 스타일", ["기름진 음식", "상관 없음", "담백한 음식"], key=f"style_{user_num}")
    soup = st.radio("국물 여부", ["국물 요리", "상관 없음", "비국물 요리"], key=f"soup_{user_num}")
    calorie = st.radio("칼로리", ["저칼로리", "상관 없음", "고칼로리"], key=f"calorie_{user_num}")
    spicy = st.radio("매운 정도", ["안매운거", "상관 없음", "매운거"], key=f"spicy_{user_num}")
    sweet = st.radio("단 정도", ["안단거", "상관 없음", "단거"], key=f"sweet_{user_num}")
    salty = st.radio("짠 정도", ["안짠거", "상관 없음", "짠거"], key=f"salty_{user_num}")

    if st.button(f"User {user_num} 선택 완료"):
        user_input_text = f"카테고리: {', '.join(categories)}, 스타일: {style}, 국물 여부: {soup}, 칼로리: {calorie}, 매운 정도: {spicy}, 단 정도: {sweet}, 짠 정도: {salty}"
        st.session_state.user_inputs.append(user_input_text)
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
            except RateLimitError:
                if retry < max_retries - 1:
                    time.sleep(20)  # 재시도 전 대기 시간 설정
                else:
                    raise
    
    user_embeddings = [get_embedding(user_input) for user_input in st.session_state.user_inputs]
    
    # 메뉴 데이터
    menu_db = {
        "돈까스": "일식, 기름진 음식, 비국물 요리, 고칼로리, 안매운거, 짠거, 안단거",
        "스시": "일식, 담백한 음식, 비국물 요리, 저칼로리, 안매운거, 안짠거, 단거",
         "회": "일식, 담백한 음식, 비국물 요리, 저칼로리, 안매운거, 안짠거, 안단거",
    "라멘": "일식, 기름진 음식, 국물 요리, 고칼로리, 매운거, 짠거, 안단거",
    "소바": "일식, 담백한 음식, 국물 요리, 저칼로리, 안매운거, 안짠거, 안단거",
    "우동": "일식, 담백한 음식, 국물 요리, 저칼로리, 안매운거, 짠거, 단거",
    "덮밥": "일식, 기름진 음식, 비국물 요리, 중간칼로리, 안매운거, 짠거, 안단거",
    "커리": "일식, 기름진 음식, 국물 요리, 고칼로리, 매운거, 짠거, 안단거",
    "짜장면": "중식, 기름진 음식, 비국물 요리, 중간칼로리, 안매운거, 안짠거, 단거",
    "탕수육": "중식, 기름진 음식, 비국물 요리, 고칼로리, 안매운거, 안짠거, 단거",
    "짬뽕": "중식, 기름진 음식, 국물 요리, 고칼로리, 매운거, 짠거, 안단거",
    "볶음밥": "중식, 기름진 음식, 비국물 요리, 중간칼로리, 안매운거, 짠거, 안단거",
    "마라탕": "중식, 기름진 음식, 국물 요리, 고칼로리, 매운거, 짠거, 안단거",
    "마라샹궈": "중식, 기름진 음식, 비국물 요리, 고칼로리, 매운거, 짠거, 안단거",
    "양꼬치": "중식, 기름진 음식, 비국물 요리, 고칼로리, 매운거, 짠거, 안단거",
    "고기구이(삼겹살)": "한식, 기름진 음식, 비국물 요리, 고칼로리, 안매운거, 짠거, 안단거",
    "국밥": "한식, 기름진 음식, 국물 요리, 중간칼로리, 안매운거, 짠거, 안단거",
    "보쌈": "한식, 기름진 음식, 비국물 요리, 고칼로리, 안매운거, 짠거, 안단거",
    "불고기": "한식, 기름진 음식, 비국물 요리, 중간칼로리, 안매운거, 짠거, 단거",
    "비빔밥": "한식, 담백한 음식, 비국물 요리, 중간칼로리, 안매운거, 안짠거, 안단거",
    "김치찌개": "한식, 기름진 음식, 국물 요리, 중간칼로리, 매운거, 짠거, 안단거",
    "떡볶이": "한식, 기름진 음식, 비국물 요리, 고칼로리, 매운거, 안짠거, 단거",
    "국수": "한식, 담백한 음식, 국물 요리, 저칼로리, 안매운거, 안짠거, 안단거",
    "김밥": "한식, 담백한 음식, 비국물 요리, 중간칼로리, 안매운거, 안짠거, 안단거",
    "피자": "양식, 기름진 음식, 비국물 요리, 고칼로리, 안매운거, 짠거, 안단거",
    "파스타": "양식, 기름진 음식, 비국물 요리, 고칼로리, 안매운거, 짠거, 단거",
    "스테이크": "양식, 기름진 음식, 비국물 요리, 고칼로리, 안매운거, 짠거, 안단거",
    "뇨끼": "양식, 담백한 음식, 비국물 요리, 저칼로리, 안매운거, 안짠거, 안단거",
    "샐러드": "양식, 담백한 음식, 비국물 요리, 저칼로리, 안매운거, 안짠거, 안단거",
    "햄버거": "양식, 기름진 음식, 비국물 요리, 고칼로리, 안매운거, 짠거, 안단거",
    "샌드위치": "양식, 담백한 음식, 비국물 요리, 고칼로리, 안매운거, 안짠거, 안단거",
    "쌀국수": "동남아, 담백한 음식, 국물 요리, 중간칼로리, 안매운거, 안짠거, 안단거",
    "분짜": "동남아, 담백한 음식, 비국물 요리, 중간칼로리, 안매운거, 안짠거, 안단거",
    "팟타이": "동남아, 담백한 음식, 비국물 요리, 중간칼로리, 안매운거, 안짠거, 안단거"
    }
    
    # 메뉴 설명 임베딩 생성
    menu_embeddings = {menu: get_embedding(description) for menu, description in menu_db.items()}
    
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
