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
        manu_score = fuzz.partial_ratio(input_manu, row['norm_manu']) * 0.1
        
        final_score = name_score + str_score + ing_score + manu_score
        
        results.append({
            'row': row,
            'score': final_score,
            'reason': f"Tên:{int(name_score)} + HL:{int(str_score)}"
        })
    
    # Sắp xếp lấy cao nhất
    results.sort(key=lambda x: x['score'], reverse=True)
    
    if results:
        best = results[0]
        return best['row'], best['score'], best['reason']
    else:
        return None, 0, "Không khớp logic"

# --- GIAO DIỆN ---
st.title("🛡️ PharmaMatch: Chế Độ Map Chính Xác (Strict Mode)")
st.info("Logic mới: Tên thuốc (Ưu tiên 1) -> Hàm lượng (Bắt buộc khớp số) -> Các thông tin khác.")

with st.sidebar:
    st.header("Cấu hình")
    user_api_key = st.text_input("Gemini API Key", type="password")
    if not user_api_key and "GENAI_API_KEY" in st.secrets:
        user_api_key = st.secrets["GENAI_API_KEY"]
    
    st.warning("⚠️ Chế độ này sẽ gọi AI cho TẤT CẢ các dòng để đảm bảo chính xác nhất. Tốc độ sẽ chậm hơn (khoảng 3-4s/dòng).")

vtma_df = load_vtma_data()
if vtma_df.empty:
    st.error("Chưa có file data!")
    st.stop()

uploaded = st.file_uploader("Upload File Dược Vương", type=['xlsx', 'csv'])

if uploaded and st.button("🚀 CHẠY MAP CHÍNH XÁC"):
    if not user_api_key:
        st.error("Cần API Key để phân tích hàm lượng!")
        st.stop()
        
    if uploaded.name.endswith('.csv'): df_in = pd.read_csv(uploaded)
    else: df_in = pd.read_excel(uploaded)
    
    col_name = df_in.columns[0]
    results = []
    
    bar = st.progress(0, text="Đang khởi động AI...")
    
    for i, row in df_in.iterrows():
        raw = str(row[col_name])
        
        # 1. AI Phân tích (Bắt buộc)
        ai_data = ai_parse_product(raw, user_api_key)
        
        # 2. Logic Phân Tầng
        match_row, score, reason = hierarchical_match(ai_data, vtma_df)
        
        # 3. Ghi log
        res = {
            'DV_Input': raw,
            'AI_Hieu_La': f"{ai_data.get('brand_name')} | HL: {ai_data.get('strength')}",
            'VTMA_Code': '',
            'VTMA_Name': '',
            'VTMA_HamLuong': '',
            'Match_Score': score,
            'Chi_Tiet_Diem': reason,
            'Danh_Gia': 'Thấp'
        }
        
        if match_row is not None:
            res.update({
                'VTMA_Code': match_row['ma_thuoc'],
                'VTMA_Name': match_row['ten_thuoc'],
                'VTMA_HamLuong': match_row['ham_luong'],
                'Danh_Gia': 'Cao' if score > 70 else 'Kiểm tra lại'
            })
            
        results.append(res)
        bar.progress((i+1)/len(df_in), text=f"Đang xử lý: {raw}")
        
    final_df = pd.DataFrame(results)
    st.success("Hoàn thành mapping!")
    st.dataframe(final_df)
    
    # Download
    os.makedirs('output', exist_ok=True)
    fname = f"output/map_chinhxac_{datetime.now().strftime('%H%M')}.xlsx"
    final_df.to_excel(fname, index=False)
    with open(fname, "rb") as f:
        st.download_button("📥 Tải kết quả", f, file_name="ket_qua_chinh_xac.xlsx")
