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
st.set_page_config(page_title="Dược Vương Mapping Tool", layout="wide")

def normalize_text(text):
    if pd.isna(text): return ""
    return unidecode.unidecode(str(text).lower()).strip()

# --- GỌI AI GEMINI ---
def get_ai_info(product_name, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        # Prompt được tối ưu để trả về đúng cấu trúc so sánh
        prompt = f"""
        Phân tích thuốc: "{product_name}". 
        Trả về JSON keys: 
        "active_ingredient" (hoạt chất chính, tiếng anh càng tốt), 
        "brand_name" (tên biệt dược ngắn gọn), 
        "strength" (hàm lượng số+đơn vị), 
        "manufacturer" (tên hãng sản xuất),
        "dosage_form" (dạng bào chế: viên, gói, ống...).
        Nếu không rõ thì để null.
        """
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
    except:
        return {}

# --- TÍNH ĐIỂM KHỚP (LOGIC MỚI CHO FILE VTMA CHUẨN) ---
def calculate_score(input_item, db_row):
    score = 0
    
    # 1. SO KHỚP HOẠT CHẤT (Quan trọng nhất - 40%)
    # So cột 'hoat_chat' (Cột D trong file VTMA)
    if input_item.get('active_ingredient'):
        score += fuzz.token_sort_ratio(normalize_text(input_item['active_ingredient']), normalize_text(db_row['hoat_chat'])) * 0.4
    
    # 2. SO KHỚP HÀM LƯỢNG (30%)
    # So cột 'ham_luong' (Cột G trong file VTMA)
    if input_item.get('strength'):
        s_score = fuzz.ratio(normalize_text(input_item['strength']), normalize_text(db_row['ham_luong']))
        score += s_score * 0.3
        
    # 3. SO KHỚP TÊN THƯƠNG MẠI (20%)
    # So cột 'ten_thuoc' (Cột C - tên ngắn gọn như A.T DOMPERIDON) thay vì tên đầy đủ
    brand_score = fuzz.token_set_ratio(normalize_text(input_item.get('brand_name','')), normalize_text(db_row['ten_thuoc']))
    score += brand_score * 0.2
    
    # 4. NHÀ SẢN XUẤT (10%)
    # So cột 'ten_cong_ty' (Cột F - AN THIEN_A.T PHARM)
    if input_item.get('manufacturer'):
        manu_score = fuzz.partial_ratio(normalize_text(input_item['manufacturer']), normalize_text(db_row['ten_cong_ty']))
        score += manu_score * 0.1
        
    return round(score, 1)

# --- GIAO DIỆN ---
st.title("💊 Dược Vương Mapping Tool (Phiên bản VTMA Chuẩn)")

with st.sidebar:
    st.header("Cài đặt")
    user_api_key = st.text_input("Gemini API Key", type="password")
    if not user_api_key and "GENAI_API_KEY" in st.secrets:
        user_api_key = st.secrets["GENAI_API_KEY"]
        
    threshold = st.slider("Độ chính xác (%)", 0, 100, 50)
    top_n = st.number_input("Số mã gợi ý", 1, 10, 3)

# Load Data VTMA
try:
    vtma_df = pd.read_csv("data/vtma_standard.csv")
    st.success(f"✅ Đã tải {len(vtma_df)} mã VTMA. Hệ thống sẵn sàng!")
except FileNotFoundError:
    st.error("❌ Lỗi: Không tìm thấy file data/vtma_standard.csv. Hãy chắc chắn anh/chị đã lưu file vào đúng thư mục data.")
    st.stop()
except Exception as e:
    st.error(f"❌ Lỗi đọc file CSV: {e}. Hãy đảm bảo file CSV được lưu với Encoding UTF-8.")
    st.stop()

# Upload File Dược Vương
uploaded = st.file_uploader("Chọn file Danh mục Dược Vương (Excel/CSV)", type=['xlsx', 'csv'])

if uploaded and st.button("🚀 CHẠY MAPPING"):
    if not user_api_key:
        st.warning("⚠️ Chưa nhập API Key!")
        st.stop()
        
    if uploaded.name.endswith('.csv'): df_in = pd.read_csv(uploaded)
    else: df_in = pd.read_excel(uploaded)
    
    col_name = df_in.columns[0]
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
                matches.append({
                    'ma_thuoc': v_row['ma_thuoc'],
                    'ten_thuoc': v_row['ten_thuoc'],
                    'hoat_chat': v_row['hoat_chat'],
                    'ham_luong': v_row['ham_luong'],
                    'ten_cong_ty': v_row['ten_cong_ty'], # Lấy chính xác cột F
                    'dang_bao_che': v_row['dang_bao_che'],
                    'score': s
                })
        
        matches = sorted(matches, key=lambda x: x['score'], reverse=True)[:top_n]
        
        if not matches:
            results.append({'DV_Input': raw, 'Status': 'Không tìm thấy'})
        else:
            for m in matches:
                # Logic đánh giá
                danh_gia = 'Cao' if m['score'] > 85 else ('Trung bình' if m['score'] > 60 else 'Thấp')
                
                results.append({
                    'DV_Input': raw,
                    'AI_Hieu_La': f"{ai_data.get('brand_name')} / {ai_data.get('active_ingredient')} / {ai_data.get('strength')}",
                    'VTMA_Code': m['ma_thuoc'],
                    'VTMA_Name': m['ten_thuoc'],
                    'VTMA_HoatChat': m['hoat_chat'],
                    'VTMA_HamLuong': m['ham_luong'],
                    'VTMA_NSX': m['ten_cong_ty'],
                    'Match_Score': m['score'],
                    'Do_Tin_Cay': danh_gia
                })
        bar.progress((i+1)/len(df_in))
        
    res_df = pd.DataFrame(results)
    st.dataframe(res_df)
    
    # Download logic
    os.makedirs('output', exist_ok=True)
    fname = f"output/ket_qua_{datetime.now().strftime('%H%M')}.xlsx"
    res_df.to_excel(fname, index=False)
    with open(fname, "rb") as f:
        st.download_button("📥 Tải kết quả Mapping", f, file_name="ket_qua_map.xlsx")
        
