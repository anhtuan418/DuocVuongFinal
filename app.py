import streamlit as st
import pandas as pd
import google.generativeai as genai
from rapidfuzz import fuzz, process
import unidecode
import json
import time
import os
from datetime import datetime

# --- CẤU HÌNH ---
st.set_page_config(page_title="Dược Vương Speed Map", layout="wide")

# Cache dữ liệu VTMA để không phải load lại mỗi lần click
@st.cache_data
def load_vtma_data():
    try:
        df = pd.read_csv("data/vtma_standard.csv")
        # Tạo cột text tổng hợp để so sánh nhanh
        df['search_text'] = df.apply(lambda x: normalize_text(f"{x['ten_thuoc']} {x['hoat_chat']} {x['ten_cong_ty']}"), axis=1)
        return df
    except:
        return pd.DataFrame()

def normalize_text(text):
    if pd.isna(text): return ""
    return unidecode.unidecode(str(text).lower()).strip()

# --- GỌI AI (Chỉ dùng khi cần thiết) ---
def get_ai_info(product_name, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Phân tích thuốc: "{product_name}". 
        Trả về JSON keys: "active_ingredient", "strength", "brand_name", "manufacturer".
        Nếu không rõ để null.
        """
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
    except:
        return {}

# --- HÀM TÍNH ĐIỂM (Đã tối ưu) ---
def calculate_score_detailed(input_info, db_row, is_ai_data=False):
    score = 0
    
    # Nếu dữ liệu từ AI
    if is_ai_data:
        if input_info.get('active_ingredient'):
            score += fuzz.token_sort_ratio(normalize_text(input_info['active_ingredient']), normalize_text(db_row['hoat_chat'])) * 0.4
        if input_info.get('strength'):
            score += fuzz.ratio(normalize_text(input_info['strength']), normalize_text(db_row['ham_luong'])) * 0.3
        score += fuzz.token_set_ratio(normalize_text(input_info.get('brand_name','')), normalize_text(db_row['ten_thuoc'])) * 0.2
        if input_info.get('manufacturer'):
            score += fuzz.partial_ratio(normalize_text(input_info['manufacturer']), normalize_text(db_row['ten_cong_ty'])) * 0.1
            
    # Nếu dữ liệu thô (So sánh chuỗi trực tiếp)
    else:
        # So khớp tên thuốc Dược Vương với (Tên + Hoạt chất VTMA)
        score = fuzz.token_set_ratio(input_info, db_row['search_text'])
        
    return round(score, 1)

# --- GIAO DIỆN ---
st.title("⚡ PharmaMatch Speed: Map Dữ Liệu Tốc Độ Cao")

with st.sidebar:
    st.header("Cài đặt")
    user_api_key = st.text_input("Gemini API Key", type="password")
    if not user_api_key and "GENAI_API_KEY" in st.secrets:
        user_api_key = st.secrets["GENAI_API_KEY"]
        
    threshold = st.slider("Ngưỡng gọi AI (%)", 50, 90, 70, help="Nếu so khớp thô dưới mức này mới gọi AI")
    top_n = st.number_input("Số mã gợi ý", 1, 5, 1)

vtma_df = load_vtma_data()
if vtma_df.empty:
    st.error("❌ Chưa có file data/vtma_standard.csv")
    st.stop()
else:
    st.success(f"✅ Database: {len(vtma_df)} mã")

uploaded = st.file_uploader("Upload File Dược Vương", type=['xlsx', 'csv'])

if uploaded and st.button("🚀 CHẠY MAPPING SIÊU TỐC"):
    if not user_api_key:
        st.warning("⚠️ Cần API Key để xử lý các ca khó!")
        st.stop()
        
    if uploaded.name.endswith('.csv'): df_in = pd.read_csv(uploaded)
    else: df_in = pd.read_excel(uploaded)
    
    col_name = df_in.columns[0]
    results = []
    
    # Thanh progress bar
    progress_text = "Đang xử lý..."
    my_bar = st.progress(0, text=progress_text)
    
    # Chuyển cột VTMA search text sang list để tìm kiếm vector nhanh hơn
    vtma_search_list = vtma_df['search_text'].tolist()
    vtma_indices = vtma_df.index.tolist()
    
    total_rows = len(df_in)
    ai_call_count = 0
    
    for i, row in df_in.iterrows():
        raw_name = str(row[col_name])
        normalized_name = normalize_text(raw_name)
        
        # BƯỚC 1: QUÉT NHANH (FAST SCAN)
        # Tìm 5 ứng viên sáng giá nhất dựa trên text thuần túy
        # process.extract dùng thuật toán C++ nên cực nhanh, không cần loop thủ công
        candidates_raw = process.extract(normalized_name, vtma_search_list, limit=10, scorer=fuzz.token_set_ratio)
        
        best_match = None
        best_score = 0
        
        # Kiểm tra ứng viên tốt nhất
        if candidates_raw:
            top_candidate_text, top_score, top_index = candidates_raw[0]
            if top_score >= threshold:
                # Nếu điểm cao -> CHỐT LUÔN (Không gọi AI)
                best_match = vtma_df.iloc[top_index]
                best_score = top_score
                method = "Text Match (Nhanh)"
            else:
                # Nếu điểm thấp -> GỌI AI (Chậm nhưng chắc)
                ai_data = get_ai_info(raw_name, user_api_key)
                ai_call_count += 1
                method = "AI Analysis (Sâu)"
                
                # Tính lại điểm với thông tin AI
                re_ranked = []
                # Chỉ so sánh lại với 10 ứng viên tiềm năng lúc nãy (đỡ phải quét cả 10.000 dòng)
                for _, _, idx in candidates_raw:
                    v_row = vtma_df.iloc[idx]
                    s = calculate_score_detailed(ai_data, v_row, is_ai_data=True)
                    re_ranked.append((v_row, s))
                
                # Sort lại
                re_ranked.sort(key=lambda x: x[1], reverse=True)
                if re_ranked:
                    best_match, best_score = re_ranked[0]

        # Ghi kết quả
        if best_match is not None:
             results.append({
                'DV_Input': raw_name,
                'VTMA_Code': best_match['ma_thuoc'],
                'VTMA_Name': best_match['ten_thuoc'],
                'VTMA_HoatChat': best_match['hoat_chat'],
                'Score': best_score,
                'Method': method
            })
        else:
            results.append({'DV_Input': raw_name, 'Status': 'Không tìm thấy'})
            
        # Update progress
        my_bar.progress((i + 1) / total_rows, text=f"Đã xử lý {i+1}/{total_rows} (Gọi AI: {ai_call_count} lần)")

    # Hiển thị kết quả
    st.success(f"Hoàn tất! Chỉ phải gọi AI {ai_call_count}/{total_rows} dòng.")
    res_df = pd.DataFrame(results)
    st.dataframe(res_df)
    
    # Download
    os.makedirs('output', exist_ok=True)
    fname = f"output/map_sieutoc_{datetime.now().strftime('%H%M')}.xlsx"
    res_df.to_excel(fname, index=False)
    with open(fname, "rb") as f:
        st.download_button("📥 Tải kết quả", f, file_name="ket_qua_sieu_toc.xlsx")
