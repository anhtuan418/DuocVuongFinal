import streamlit as st
import pandas as pd
import google.generativeai as genai
from rapidfuzz import fuzz
import unidecode
import json
import time
import os
from datetime import datetime

# --- CẤU HÌNH ---
st.set_page_config(page_title="Dược Vương Mapping", layout="wide")

def normalize_text(text):
    if pd.isna(text): return ""
    return unidecode.unidecode(str(text).lower()).strip()

# --- GỌI AI ---
def get_ai_info(product_name, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Phân tích thuốc: "{product_name}". 
        Trả về JSON keys: "active_ingredient", "brand_name", "strength", "manufacturer".
        Nếu không biết thì để null.
        """
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
    except:
        return {}

# --- TÍNH ĐIỂM KHỚP ---
def calculate_score(input_item, db_row):
    score = 0
    # Hoạt chất (40%)
    if input_item.get('active_ingredient'):
        score += fuzz.token_sort_ratio(normalize_text(input_item['active_ingredient']), normalize_text(db_row['hoat_chat'])) * 0.4
    # Hàm lượng (30%)
    if input_item.get('strength'):
        score += fuzz.ratio(normalize_text(input_item['strength']), normalize_text(db_row['ham_luong'])) * 0.3
    # Tên (20%)
    score += fuzz.token_set_ratio(normalize_text(input_item.get('brand_name','')), normalize_text(db_row['ten_thuoc'])) * 0.2
    # Hãng (10%)
    if input_item.get('manufacturer'):
        score += fuzz.partial_ratio(normalize_text(input_item['manufacturer']), normalize_text(db_row['ten_cong_ty'])) * 0.1
    return round(score, 1)

# --- GIAO DIỆN ---
st.title("💊 Dược Vương: Tool Map Dữ Liệu Tự Động")

with st.sidebar:
    st.header("Cài đặt")
    # Lấy API Key từ Secrets (Cloud) hoặc nhập tay
    user_api_key = st.text_input("Gemini API Key", type="password")
    if not user_api_key and "GENAI_API_KEY" in st.secrets:
        user_api_key = st.secrets["GENAI_API_KEY"]
        
    threshold = st.slider("Độ chính xác (%)", 0, 100, 50)
    top_n = st.number_input("Số mã gợi ý", 1, 10, 3)

# Load Data
try:
    vtma_df = pd.read_csv("data/vtma_standard.csv")
    st.success(f"✅ Đã tải {len(vtma_df)} mã VTMA chuẩn.")
except:
    st.error("❌ Lỗi: Không tìm thấy file data/vtma_standard.csv")
    st.stop()

# Upload File
uploaded = st.file_uploader("Chọn file Dược Vương (Excel/CSV)", type=['xlsx', 'csv'])

if uploaded and st.button("🚀 CHẠY MAPPING"):
    if not user_api_key:
        st.warning("⚠️ Vui lòng nhập API Key!")
        st.stop()
        
    if uploaded.name.endswith('.csv'): df_in = pd.read_csv(uploaded)
    else: df_in = pd.read_excel(uploaded)
    
    col_name = df_in.columns[0] # Lấy cột đầu tiên
    st.info(f"Đang xử lý cột tên: {col_name}")
    
    results = []
    bar = st.progress(0)
    
    for i, row in df_in.iterrows():
        raw = str(row[col_name])
        ai_data = get_ai_info(raw, user_api_key)
        
        matches = []
        for _, v_row in vtma_df.iterrows():
            s = calculate_score(ai_data, v_row)
            if s >= threshold:
                matches.append({**v_row.to_dict(), 'score': s})
        
        matches = sorted(matches, key=lambda x: x['score'], reverse=True)[:top_n]
        
        if not matches:
            results.append({'DV_Input': raw, 'Status': 'Không tìm thấy'})
        else:
            for m in matches:
                results.append({
                    'DV_Input': raw,
                    'AI_Info': f"{ai_data.get('active_ingredient')} {ai_data.get('strength')}",
                    'VTMA_Code': m['ma_thuoc'],
                    'VTMA_Name': m['ten_thuoc'],
                    'Match_%': m['score'],
                    'Check': 'OK' if m['score']>80 else 'Check lại'
                })
        bar.progress((i+1)/len(df_in))
        
    res_df = pd.DataFrame(results)
    st.dataframe(res_df)
    
    # Download
    os.makedirs('output', exist_ok=True)
    fname = f"output/ket_qua_{datetime.now().strftime('%H%M')}.xlsx"
    res_df.to_excel(fname, index=False)
    with open(fname, "rb") as f:
        st.download_button("📥 Tải kết quả Excel", f, file_name="ket_qua_map.xlsx")