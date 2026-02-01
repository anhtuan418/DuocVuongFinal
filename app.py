import streamlit as st
import pandas as pd
import google.generativeai as genai
from rapidfuzz import fuzz, process
import unidecode
import json
import os
import re
import time
from datetime import datetime

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="PharmaMatch: Batch Processing", layout="wide")

# --- HÀM XỬ LÝ TEXT ---
def normalize_text(text):
    if pd.isna(text): return ""
    return unidecode.unidecode(str(text).lower()).strip()

def extract_numbers(text):
    """Lấy tập hợp số từ chuỗi. VD: '160/4.5' -> {'160', '4.5'}"""
    if pd.isna(text): return set()
    # Tìm số nguyên và số thập phân
    nums = re.findall(r"\d+\.?\d*", str(text))
    return set(nums)

# --- LOAD DATA ---
@st.cache_data
def load_vtma_data():
    try:
        df = pd.read_csv("data/vtma_standard.csv")
        # Tạo các cột chuẩn hóa sẵn để chạy cho nhanh
        df['norm_name'] = df['ten_thuoc'].apply(normalize_text)
        df['norm_strength'] = df['ham_luong'].apply(normalize_text)
        df['norm_ingre'] = df['hoat_chat'].apply(normalize_text)
        df['norm_manu'] = df['ten_cong_ty'].apply(normalize_text)
        return df
    except Exception as e:
        return pd.DataFrame()

# --- GỌI AI THEO LÔ (BATCH) ---
def ai_process_batch(product_list, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Tạo prompt danh sách
        items_str = "\n".join([f"- ID_{i}: {p}" for i, p in enumerate(product_list)])
        
        prompt = f"""
        Danh sách thuốc cần trích xuất thông tin:
        {items_str}
        
        Yêu cầu trả về JSON dạng List of Objects (Tuyệt đối không Markdown), mỗi object gồm:
        - "id": "ID_..." (giữ nguyên ID tương ứng)
        - "brand_name": Tên biệt dược
        - "strength": Hàm lượng số (VD: 500mg, 160/4.5). Null nếu không có.
        - "active_ingredient": Hoạt chất.
        - "manufacturer": Tên hãng.
        """
        
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(text)
        
        # Chuyển về Dictionary
        result_dict = {item['id']: item for item in data}
        return result_dict
        
    except Exception:
        return {}

# --- LOGIC MAP PHÂN TẦNG ---
def hierarchical_match(input_data, vtma_df):
    if not input_data: return None, 0, "Lỗi AI"
    
    input_brand = normalize_text(input_data.get('brand_name', ''))
    input_strength = normalize_text(input_data.get('strength', ''))
    input_ingre = normalize_text(input_data.get('active_ingredient', ''))
    
    # BƯỚC 1: LỌC THEO TÊN (Tìm 30 mã giống tên nhất)
    candidates = process.extract(
        input_brand, 
        vtma_df['norm_name'], 
        limit=30, 
        scorer=fuzz.token_set_ratio
    )
    
    # Chỉ lấy những mã có độ giống tên >= 50%
    candidate_indices = [x[2] for x in candidates if x[1] >= 50]
    
    if not candidate_
