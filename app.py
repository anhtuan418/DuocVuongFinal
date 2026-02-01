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

# --- CẤU HÌNH ---
st.set_page_config(page_title="PharmaMatch: Batch Speed", layout="wide")

def normalize_text(text):
    if pd.isna(text): return ""
    return unidecode.unidecode(str(text).lower()).strip()

def extract_numbers(text):
    """Lấy tập hợp số để so sánh chính xác."""
    if pd.isna(text): return set()
    nums = re.findall(r"\d+\.?\d*", str(text))
    return set(nums)

# --- LOAD DATA ---
@st.cache_data
def load_vtma_data():
    try:
        df = pd.read_csv("data/vtma_standard.csv")
        df['norm_name'] = df['ten_thuoc'].apply(normalize_text)
        df['norm_strength'] = df['ham_luong'].apply(normalize_text)
        df['norm_ingre'] = df['hoat_chat'].apply(normalize_text)
        df['norm_manu'] = df['ten_cong_ty'].apply(normalize_text)
        return df
    except:
        return pd.DataFrame()

# --- AI BATCH PROCESSING (GỘP NHIỀU DÒNG) ---
def ai_process_batch(product_list, api_key):
    """Gửi 1 danh sách sản phẩm lên AI cùng lúc"""
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
        - "strength": Hàm lượng số (VD: 500mg, 10mg). Null nếu không có.
        - "active_ingredient": Hoạt chất.
        - "manufacturer": Tên hãng.
        """
        
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        
        # Parse JSON
        data = json.loads(text)
        
        # Chuyển về dict để dễ map lại: {'ID_0': {...}, 'ID_1': {...}}
        result_dict = {item['id']: item for item in data}
        return result_dict
        
    except Exception as e:
        # Nếu lỗi cả batch, trả về rỗng để xử lý sau (hoặc in lỗi ra console)
        print(f"Batch Error: {e}")
        return {}

# --- LOGIC MATCHING (GIỮ NGUYÊN ĐỂ ĐẢM BẢO CHÍNH XÁC) ---
def hierarchical_match(input_data, vtma_df):
    if not input_data: return None, 0, "AI Lỗi"
    
    input_brand = normalize_text(input_data.get('brand_name', ''))
    input_strength = normalize_text(input_data.get('strength', ''))
    input_ingre = normalize_text(input_data.get('active_ingredient', ''))
    
    # 1. Lọc theo Tên (Brand Name)
    candidates = process.extract(
        input_brand, 
        vtma_df['norm_name'], 
        limit=30, 
        scorer=fuzz.token_set_ratio
    )
    
    candidate_indices = [x[2] for x in candidates if x[1] >= 50]
    if not candidate_indices: return None, 0, "Không tìm thấy tên"

    subset_df = vtma_df.iloc[candidate_indices].copy()
    
    # 2. Re-rank
    results = []
    input_nums = extract_numbers(input_strength)
    
    for idx, row in subset_df.iterrows():
        name_score = fuzz.token_set_ratio(input_brand, row['norm_name']) * 0.4
        
        # Logic Hàm Lượng Nghiêm Ngặt
        str_score = 0
        row_nums = extract_numbers(row['norm_strength'])
        
        if not input_nums: 
            str_score = fuzz.ratio(input_strength, row['norm_strength']) * 0.4
        else:
            # Nếu Input có số, bắt buộc VTMA phải chứa tập số đó
            if input_nums.issubset(row_nums) or row_nums.issubset(input_nums):
                str_score = 40 
            else:
                str_score = 0 # Phạt nặng
        
        ing_score = fuzz.token_sort_ratio(input_ingre, row['norm_ingre']) * 0.2
        
        final_score = name_score + str_score + ing_score
        results.append({'row': row, 'score': final_score})
    
    results.sort(key=lambda x
