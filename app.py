import streamlit as st
import pandas as pd
import google.generativeai as genai
from rapidfuzz import fuzz, process
import unidecode
import json
import os
import re
from datetime import datetime

# --- CẤU HÌNH ---
st.set_page_config(page_title="PharmaMatch: Logic Phân Tầng", layout="wide")

# --- HÀM CHUẨN HÓA ---
def normalize_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = unidecode.unidecode(text)
    return text.strip()

# --- HÀM TÁCH SỐ TỪ HÀM LƯỢNG (Để so sánh chính xác) ---
def extract_numbers(text):
    """Lấy các con số từ chuỗi hàm lượng. VD: '160mg/4.5mcg' -> {'160', '4.5'}"""
    if pd.isna(text): return set()
    # Tìm các số (bao gồm cả số thập phân)
    nums = re.findall(r"\d+\.?\d*", str(text))
    return set(nums)

# --- LOAD DATA ---
@st.cache_data
def load_vtma_data():
    try:
        df = pd.read_csv("data/vtma_standard.csv")
        # Chuẩn hóa trước để tìm kiếm nhanh
        df['norm_name'] = df['ten_thuoc'].apply(normalize_text)
        df['norm_strength'] = df['ham_luong'].apply(normalize_text)
        df['norm_ingre'] = df['hoat_chat'].apply(normalize_text)
        df['norm_manu'] = df['ten_cong_ty'].apply(normalize_text)
        return df
    except:
        return pd.DataFrame()

# --- AI PHÂN TÁCH THÔNG TIN (Quan trọng nhất) ---
def ai_parse_product(product_raw_name, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prompt bắt buộc AI tách riêng Hàm Lượng
        prompt = f"""
        Nhiệm vụ: Trích xuất thông tin dược phẩm từ chuỗi: "{product_raw_name}".
        Yêu cầu trả về JSON chính xác:
        - "brand_name": Tên biệt dược (VD: Panadol, Symbicort)
        - "strength": Hàm lượng số (VD: 500mg, 160/4.5, 10mg). Nếu không có ghi null.
        - "active_ingredient": Hoạt chất.
        - "manufacturer": Tên hãng/nước.
        """
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
    except:
        # Fallback nếu AI lỗi: Trả về chính cái tên đó
        return {"brand_name": product_raw_name, "strength": "", "active_ingredient": "", "manufacturer": ""}

# --- LOGIC MAP PHÂN TẦNG (HIERARCHICAL) ---
def hierarchical_match(input_data, vtma_df):
    """
    Input: Dữ liệu đã được AI làm sạch (Tên, Hàm lượng...)
    Logic:
    1. Lọc theo Tên (Brand Name) -> Lấy Top 30 ứng viên.
    2. So hàm lượng (Strength) -> Re-rank lại Top 30 này.
    3. So các tiêu chí phụ.
    """
    
    input_brand = normalize_text(input_data.get('brand_name', ''))
    input_strength = normalize_text(input_data.get('strength', ''))
    input_ingre = normalize_text(input_data.get('active_ingredient', ''))
    input_manu = normalize_text(input_data.get('manufacturer', ''))
    
    # BƯỚC 1: LỌC THEO TÊN (Ưu tiên số 1)
    # Dùng rapidfuzz lấy nhanh 30 mã có tên giống nhất trong toàn bộ DB
    # threshold=60: Tên phải giống ít nhất 60% mới được xét tiếp
    candidates = process.extract(
        input_brand, 
        vtma_df['norm_name'], 
        limit=50, 
        scorer=fuzz.token_set_ratio
    )
    
    # Lấy ra index của các ứng viên này
    candidate_indices = [x[2] for x in candidates if x[1] >= 50]
    
    if not candidate_indices:
        return None, 0, "Không tìm thấy tên tương tự"

    subset_df = vtma_df.iloc[candidate_indices].copy()
    
    # BƯỚC 2: TÍNH ĐIỂM CHI TIẾT CHO TỪNG ỨNG VIÊN
    results = []
    
    input_nums = extract_numbers(input_strength) # VD: {160, 4.5}
    
    for idx, row in subset_df.iterrows():
        # ĐIỂM TÊN (Base Score): Max 40đ
        name_score = fuzz.token_set_ratio(input_brand, row['norm_name']) * 0.4
        
        # ĐIỂM HÀM LƯỢNG (Critical): Max 40đ
        # Logic cứng: Nếu Input có số mà VTMA không có số đó -> PHẠT NẶNG
        str_score = 0
        row_nums = extract_numbers(row['norm_strength'])
        
        if not input_nums: 
            # Nếu Input không ghi hàm lượng, so sánh chuỗi mờ
            str_score = fuzz.ratio(input_strength, row['norm_strength']) * 0.4
        else:
            # Nếu Input có số (VD: 500), check xem VTMA có số 500 ko
            # Nếu tập số khớp nhau (VD: input {160, 4.5} vs row {160, 4.5}) -> Điểm tuyệt đối
            if input_nums.issubset(row_nums) or row_nums.issubset(input_nums):
                str_score = 40 # Max điểm
            else:
                str_score = 0 # Phạt về 0 nếu lệch số (VD: 10 vs 15)
        
        # ĐIỂM PHỤ (Hoạt chất + Hãng): Max 20đ
        ing_score = fuzz.token_sort_ratio(input_ingre, row['norm_ingre']) * 0.1
        manu_score = fuzz.partial
    
