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

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="PharmaMatch: Final Batch", layout="wide")
st.title("🚀 PharmaMatch: Công cụ Map Dược Phẩm")

# --- 2. CÁC HÀM XỬ LÝ ---
def normalize_text(text):
    if pd.isna(text): return ""
    return unidecode.unidecode(str(text).lower()).strip()

def extract_numbers(text):
    if pd.isna(text): return set()
    nums = re.findall(r"\d+\.?\d*", str(text))
    return set(nums)

# --- 3. LOAD DATA (Có xử lý lỗi nhưng KHÔNG DỪNG app ngay) ---
@st.cache_data
def load_vtma_data():
    try:
        # Kiểm tra cả chữ thường và hoa cho chắc ăn
        file_path = "data/vtma_standard.csv"
        if not os.path.exists(file_path):
             # Thử tìm file viết hoa nếu user lỡ đặt tên khác
            if os.path.exists("Data/vtma_standard.csv"): file_path = "Data/vtma_standard.csv"
            else: return None
            
        df = pd.read_csv(file_path)
        df['norm_name'] = df['ten_thuoc'].apply(normalize_text)
        df['norm_strength'] = df['ham_luong'].apply(normalize_text)
        df['norm_ingre'] = df['hoat_chat'].apply(normalize_text)
        df['norm_manu'] = df['ten_cong_ty'].apply(normalize_text)
        return df
    except Exception as e:
        return None

# --- 4. GỌI AI ---
def ai_process_batch(product_list, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        items_str = "\n".join([f"- ID_{i}: {p}" for i, p in enumerate(product_list)])
        prompt = f"""
        Trích xuất thông tin dược phẩm:
        {items_str}
        Trả về JSON List Objects:
        - "id": "ID_..."
        - "brand_name": Tên biệt dược
        - "strength": Hàm lượng số (VD: 500mg, 160/4.5).
        - "active_ingredient": Hoạt chất.
        - "manufacturer": Tên hãng.
        """
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(text)
        return {item['id']: item for item in data}
    except:
        return {}

# --- 5. LOGIC MAP ---
def hierarchical_match(input_data, vtma_df):
    if not input_data: return None, 0, "Lỗi AI"
    
    input_brand = normalize_text(input_data.get('brand_name', ''))
    input_strength = normalize_text(input_data.get('strength', ''))
    input_ingre = normalize_text(input_data.get('active_ingredient', ''))
    
    candidates = process.extract(input_brand, vtma_df['norm_name'], limit=30, scorer=fuzz.token_set_ratio)
    candidate_indices = [x[2] for x in candidates if x[1] >= 50]
    
    if not candidate_indices: return None, 0, "Không tìm thấy tên"

    subset_df = vtma_df.iloc[candidate_indices].copy()
    results = []
    input_nums = extract_numbers(input_strength)
    
    for idx, row in subset_df.iterrows():
        name_score = fuzz.token_set_ratio(input_brand, row['norm_name']) * 0.4
        str_score = 0
        row_nums = extract_numbers(row['norm_strength'])
        
        if not input_nums: str_score = fuzz.ratio(input_strength, row['norm_strength']) * 0.4
        else:
            if input_nums.issubset(row_nums) or row_nums.issubset(input_nums): str_score = 40
            else: str_score = 0
        
        ing_score = fuzz.token_sort_ratio(input_ingre, row['norm_ingre']) * 0.2
        final_score = name_score + str_score + ing_score
        results.append({'row': row, 'score': final_score})
    
    results.sort(key=lambda x: x['score'], reverse=True)
    if results: return results[0]['row'], results[0]['score'], "OK"
    else: return None, 0, "Không KQ"

# --- 6. GIAO DIỆN (ĐÃ CHỈNH SỬA VỊ TRÍ) ---

# Cài đặt Sidebar
with st.sidebar:
    st.header("Cấu hình")
    api_key = st.text_input("Gemini API Key", type="password")
    if not api_key and "GENAI_API_KEY" in st.secrets:
        api_key = st.secrets["GENAI_API_KEY"]
    batch_size = st.slider("Batch Size", 5, 20, 10)

# Load data ngầm
vtma_df = load_vtma_data()

# --- PHẦN UPLOAD FILE (Đưa lên đầu trang để luôn nhìn thấy) ---
st.subheader("1. Tải danh mục Dược Vương")
uploaded = st.file_uploader("Kéo thả file vào đây (Excel/CSV)", type=['xlsx', 'csv'])

# --- NÚT CHẠY VÀ HIỂN THỊ LỖI ---
if uploaded:
    # Đọc file ngay để user thấy dữ liệu
    if uploaded.name.endswith('.csv'): df_in = pd.read_csv(uploaded)
    else: df_in = pd.read_excel(uploaded)
    
    st.write(f"Đã nhận file: {len(df_in)} dòng. Cột sẽ map: **{df_in.columns[0]}**")
    
    # Nút chạy
    if st.button("🚀 CHẠY MAPPING NGAY"):
        # Kiểm tra điều kiện chạy lúc ấn nút
        if vtma_df is None or vtma_df.empty:
            st.error("❌ LỖI: Chưa tìm thấy file dữ liệu chuẩn VTMA trên hệ thống!")
            st.info("Cách sửa: Hãy kiểm tra trên GitHub của bạn đã có folder 'data' và file 'vtma_standard.csv' bên trong chưa.")
            st.stop()
            
        if not api_key:
            st.error("❌ Thiếu API Key!")
            st.stop()

        # BẮT ĐẦU CHẠY
        col_name = df_in.columns[0]
        final_results = []
        input_list = df_in[col_name].astype(str).tolist()
        total = len(input_list)
        bar = st.progress(0, text="Đang xử lý...")
        
        for i in range(0, total, batch_size):
            batch_items = input_list[i : i + batch_size]
            try: ai_data_dict = ai_process_batch(batch_items, api_key)
            except: ai_data_dict = {}
            
            for idx_in_batch, item_name in enumerate(batch_items):
                item_id = f"ID_{idx_in_batch}"
                ai_info = ai_data_dict.get(item_id, {})
                match_row, score, note = hierarchical_match(ai_info, vtma_df)
                
                res_row = {
                    'DV_Input': item_name,
                    'AI_Info': f"{ai_info.get('brand_name')} {ai_info.get('strength')}",
                    'VTMA_Code': '', 'VTMA_Name': '', 'VTMA_HamLuong': '',
                    'Score': score, 'Danh_Gia': 'Thấp'
                }
                if match_row is not None:
                    res_row.update({
                        'VTMA_Code': match_row['ma_thuoc'],
                        'VTMA_Name': match_row['ten_thuoc'],
                        'VTMA_HamLuong': match_row['ham_luong'],
                        'Danh_Gia': 'Cao' if score > 75 else 'Kiểm tra'
                    })
                final_results.append(res_row)
            
            bar.progress(min((i + batch_size) / total, 1.0))
            time.sleep(1)

        st.success("Xong!")
        res_df = pd.DataFrame(final_results)
        st.dataframe(res_df)
        
        os.makedirs('output', exist_ok=True)
        fname = f"output/map_final_{datetime.now().strftime('%H%M')}.xlsx"
        res_df.to_excel(fname, index=False)
        with open(fname, "rb") as f:
            st.download_button("📥 Tải kết quả", f, file_name="ket_qua.xlsx")

elif vtma_df is None:
    # Nếu chưa upload file input, thì hiện cảnh báo nhẹ về file VTMA nếu thiếu
    st.warning("⚠️ Cảnh báo: Hệ thống chưa tìm thấy file 'data/vtma_standard.csv'. Bạn vẫn có thể thấy nút upload bên trên, nhưng khi chạy sẽ báo lỗi.")
    
