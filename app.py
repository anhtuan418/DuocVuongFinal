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
    
    results.sort(key=lambda x: x['score'], reverse=True)
    if results:
        best = results[0]
        return best['row'], best['score'], "OK"
    return None, 0, "Low Score"

# --- GIAO DIỆN ---
st.title("🚀 PharmaMatch: Tốc Độ Cao (Batch Processing)")
st.info("Chế độ Gộp Đơn: Xử lý 10 sản phẩm cùng lúc giúp tăng tốc độ gấp 5 lần.")

with st.sidebar:
    api_key = st.text_input("Gemini API Key", type="password")
    if not api_key and "GENAI_API_KEY" in st.secrets:
        api_key = st.secrets["GENAI_API_KEY"]
    
    batch_size = st.slider("Kích thước gói (Batch Size)", 5, 20, 10, help="Số lượng SP gửi đi 1 lần. Mạng khoẻ thì để cao.")

vtma_df = load_vtma_data()
if vtma_df.empty: st.stop()

uploaded = st.file_uploader("Upload File Dược Vương", type=['xlsx', 'csv'])

if uploaded and st.button("🚀 CHẠY BATCH MAPPING"):
    if not api_key: st.stop()
    
    if uploaded.name.endswith('.csv'): df_in = pd.read_csv(uploaded)
    else: df_in = pd.read_excel(uploaded)
    
    col_name = df_in.columns[0]
    results = []
    
    # Chia dữ liệu thành các batch (gói nhỏ)
    input_data = df_in[col_name].astype(str).tolist()
    total_items = len(input_data)
    
    progress_bar = st.progress(0, text="Đang khởi động...")
    
    # Vòng lặp xử lý từng gói
    for i in range(0, total_items, batch_size):
        batch_items = input_data[i : i + batch_size] # Lấy danh sách 10 sp
        
        # 1. Gọi AI cho cả gói
        try:
            ai_results_dict = ai_process_batch(batch_items, api_key)
        except:
            ai_results_dict = {} # Nếu lỗi thì bỏ qua batch này (hoặc retry nếu muốn phức tạp hơn)
        
        # 2. Xử lý map cho từng sp trong gói
        for idx, item_name in enumerate(batch_items):
            item_id = f"ID_{idx}"
            ai_info = ai_results_dict.get(item_id, {})
            
            # Map với VTMA
            match_row, score, note = hierarchical_match(ai_info, vtma_df)
            
            # Ghi kết quả
            res = {
                'DV_Input': item_name,
                'AI_Data': f"{ai_info.get('brand_name')} {ai_info.get('strength')}",
                'VTMA_Code': '', 'VTMA_Name': '', 'VTMA_HamLuong': '',
                'Score': score, 'Danh_Gia': 'Thấp'
            }
            
            if match_row is not None:
                res.update({
                    'VTMA
