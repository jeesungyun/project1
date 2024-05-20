import openai
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# API 키 설정
openai.api_key = 'sk-proj-3jl7j9Eiayeingc1pMghT3BlbkFJkxlBtbKywVBI11NOQiW3'

# 사용자 입력 받기
st.title("음식 추천 시스템")

# 프롬프트 1: 음식 카테고리 선택
category = st.selectbox("음식 카테고리", ["한식", "중식", "일식", "양식", "동남아"])

# 프롬프트 2: 음식 스타일 선택
style = st.radio("음식 스타일", ["기름진 음식", "상관 없음", "담백한 음식"])

# 프롬프트 3: 국물 여부 선택
soup = st.radio("국물 여부", ["국물 요리", "상관 없음", "비국물 요리"])

# 프롬프트 4: 칼로리 선택
calorie = st.radio("칼로리", ["저칼로리", "상관 없음", "고칼로리"])

# 프롬프트 5: 매운 정도 선택
spicy = st.radio("매운 정도", ["안매운거", "상관 없음", "매운거"])

# 프롬프트 6: 단 정도 선택
sweet = st.radio("단 정도", ["안단거", "상관 없음", "단거"])

# 프롬프트 7: 짠 정도 선택
salty = st.radio("짠 정도", ["안짠거", "상관 없음", "짠거"])

# 사용자 입력을 텍스트로 변환하는 함수
def create_user_input_text(category, style, soup, calorie, spicy, sweet, salty):
    style_text = "기름진 음식" if style == "기름진 음식" else ("담백한 음식" if style == "담백한 음식" else "상관 없음")
    soup_text = "국물 요리" if soup == "국물 요리" else ("비국물 요리" if soup == "비국물 요리" else "상관 없음")
    calorie_text = "저칼로리" if calorie == "저칼로리" else ("고칼로리" if calorie == "고칼로리" else "상관 없음")
    spicy_text = "안매운거" if spicy == "안매운거" else ("매운거" if spicy == "매운거" else "상관 없음")
    sweet_text = "안단거" if sweet == "안단거" else ("단거" if sweet == "단거" else "상관 없음")
    salty_text = "안짠거" if salty == "안짠거" else ("짠거" if salty == "짠거" else "상관 없음")
    
    return f"카테고리: {category}, 스타일: {style_text}, 국물 여부: {soup_text}, 칼로리: {calorie_text}, 매운 정도: {spicy_text}, 단 정도: {sweet_text}, 짠 정도: {salty_text}"

# 사용자 입력을 텍스트로 생성
user_input_text = create_user_input_text(category, style, soup, calorie, spicy, sweet, salty)

# 사용자 입력을 임베딩 벡터로 변환하는 함수
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

user_embedding = get_embedding(user_input_text)

# 메뉴 데이터
menu_db = {
    "돈까스": "일식, 기름진 음식, 비국물 요리, 고칼로리, 안매운거, 짠거, 안단거",
    "스시": "일식, 담백한 음식, 비국물 요리, 저칼로리, 안매운거, 안짠거, 단거",
    # 생략된 메뉴 데이터들
}

# 메뉴 설명 임베딩 생성
menu_embeddings = {menu: get_embedding(description) for menu, description in menu_db.items()}

# 코사인 유사도를 통해 가장 유사도가 낮은 메뉴 찾기
def find_least_similar(query_embedding, menu_embeddings):
    min_similarity = 1
    least_similar = None
    for menu, embedding in menu_embeddings.items():
        similarity = cosine_similarity([query_embedding], [embedding])[0][0]
        if similarity < min_similarity:
            min_similarity = similarity
            least_similar = menu
    return least_similar

# 유사도가 가장 낮은 메뉴 찾기
least_similar_menu = find_least_similar(user_embedding, menu_embeddings)

st.write("임베딩을 사용한 유사도가 가장 낮은 메뉴 (추천 메뉴):", least_similar_menu)
