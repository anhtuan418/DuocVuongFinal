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

# --- 2. CÁC HÀM XỬ LÝ TEXT ---
def normalize_text(text):
    if pd.isna(text): return ""
    return unidecode.unidecode(str(text).lower()).strip()

def extract_numbers(text):
    """
    Lấy tập hợp số từ chuỗi để so sánh chính xác.
    Ví dụ: '160/4.5' -> {'160', '4.5'}
    """
    if pd.isna(text): return set()
    nums = re.findall(r"\d+\.?\d*", str(text))
    return set(nums)

# --- 3. LOAD DATA VTMA ---
@st.cache_data
def load_vtma_data():
    try:
        # Đảm bảo đường dẫn file đúng
        if not os.path.exists("data/vtma_standard.csv"):
            return pd.DataFrame()
            
        df = pd.read_csv("data/vtma_standard.csv")
        # Tạo cột chuẩn hóa sẵn
        df['norm_name'] = df['ten_thuoc'].apply(normalize_text)
        df['norm_strength'] = df['ham_luong'].apply(normalize_text)
        df['norm_ingre'] = df['hoat_chat'].apply(normalize_text)
        df['norm_manu'] = df['ten_cong_ty'].apply(normalize_text)
        return df
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return pd.DataFrame()

# --- 4. GỌI AI THEO BATCH (GỘP NHIỀU DÒNG) ---
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
        
        # Chuyển về Dictionary: {'ID_0': {...}, 'ID_1': {...}}
        result_dict = {item['id']: item for item in data}
        return result_dict
        
    except Exception as e:
        return {}

# --- 5. LOGIC MAP PHÂN TẦNG (HIERARCHICAL) ---
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
    
    # Lấy index của các dòng có tên giống >= 50%
    candidate_indices = [x[2] for x in candidates if x[1] >= 50]
    
    # --- ĐÂY LÀ CHỖ BẠN HAY BỊ LỖI, TÔI ĐÃ KIỂM TRA KỸ ---
    if not candidate_indices:
        return None, 0, "Không tìm thấy tên"

    subset_df = vtma_df.iloc[candidate_indices].copy()
    
    # BƯỚC 2: TÍNH ĐIỂM CHI TIẾT
    results = []
    input_nums = extract_numbers(input_strength)
    
    for idx, row in subset_df.iterrows():
        # Điểm Tên (40đ)
        name_score = fuzz.token_set_ratio(input_brand, row['norm_name']) * 0.4
        
        # Điểm Hàm Lượng (40đ) - Logic ngặt nghèo
        str_score = 0
        row_nums = extract_numbers(row['norm_strength'])
        
        if not input_nums:
            # Nếu Input không có số, so sánh text tương đối
            str_score = fuzz.ratio(input_strength, row['norm_strength']) * 0.4
        else:
            # Nếu Input có số, BẮT BUỘC VTMA phải chứa đủ các số đó
            if input_nums.issubset(row_nums) or row_nums.issubset(input_nums):
                str_score = 40
            else:
                str_score = 0 # Phạt về 0 nếu lệch số (VD: 10 vs 15)
        
        # Điểm Hoạt chất (20đ)
        ing_score = fuzz.token_sort_ratio(input_ingre, row['norm_ingre']) * 0.2
        
        final_score = name_score + str_score + ing_score
        
        # Lưu lại kết quả
        results.append({'row': row, 'score': final_score})
    
    # Sắp xếp từ cao xuống thấp
    results.sort(key=lambda x: x['score'], reverse=True)
    
    if results:
        best = results[0]
        return best['row'], best['score'], "OK"
    else:
        return None, 0, "Không có kết quả"

# --- 6. GIAO DIỆN CHÍNH ---
st.title("🚀 PharmaMatch: Final Batch Version")
st.info("Phiên bản ổn định: Chạy Batch 10 SP + Logic Hàm lượng chặt chẽ.")

with st.sidebar:
    st.header("Cấu hình")
    api_key = st.text_input("Gemini API Key", type="password")
    # Lấy key từ secrets nếu có
    if not api_key and "GENAI_API_KEY" in st.secrets:
        api_key = st.secrets["GENAI_API_KEY"]
    
    batch_size = st.slider("Kích thước lô (Batch Size)", 5, 20, 10)

# Load Data
vtma_df = load_vtma_data()
if vtma_df.empty:
    st.error("⚠️ Không tìm thấy file 'data/vtma_standard.csv'. Vui lòng kiểm tra lại folder data.")
    st.stop()

uploaded = st.file_uploader("Upload File Dược Vương (Excel/CSV)", type=['xlsx', 'csv'])

if uploaded and st.button("🚀 CHẠY MAPPING"):
    if not api_key:
        st.error("Vui lòng nhập API Key!")
        st.stop()
        
    # Đọc file
    if uploaded.name.endswith('.csv'): 
        df_in = pd.read_csv(uploaded)
    else: 
        df_in = pd.read_excel(uploaded)
    
    col_name = df_in.columns[0]
    final_results = []
    
    input_list = df_in[col_name].astype(str).tolist()
    total = len(input_list)
    
    bar = st.progress(0, text="Đang xử lý...")
    
    # Vòng lặp xử lý từng lô (Batch Loop)
    for i in range(0, total, batch_size):
        # Cắt lô
        batch_items = input_list[i : i + batch_size]
        
        # Gọi AI (Xử lý lỗi nếu AI chết giữa chừng)
        try:
            ai_data_dict = ai_process_batch(batch_items, api_key)
        except:
            ai_data_dict = {}
        
        # Xử lý từng phần tử trong lô
        for idx_in_batch, item_name in enumerate(batch_items):
            item_id = f"ID_{idx_in_batch}"
            ai_info = ai_data_dict.get(item_id, {})
            
            # Map dữ liệu
            match_row, score, note = hierarchical_match(ai_info, vtma_df)
            
            # Tạo dòng kết quả
            res_row = {
                'DV_Input': item_name,
                'AI_Info': f"{ai_info.get('brand_name')} {ai_info.get('strength')}",
                'VTMA_Code': '',
                'VTMA_Name': '',
                'VTMA_HamLuong': '',
                'VTMA_HoatChat': '',
                'Score': score,
                'Danh_Gia': 'Thấp'
            }
            
            if match_row is not None:
                res_row.update({
                    'VTMA_Code': match_row['ma_thuoc'],
                    'VTMA_Name': match_row['ten_thuoc'],
                    'VTMA_HamLuong': match_row['ham_luong'],
                    'VTMA_HoatChat': match_row['hoat_chat'],
                    'Danh_Gia': 'Cao' if score > 75 else 'Kiểm tra'
                })
            
            final_results.append(res_row)
        
        # Update tiến độ
        prog = min((i + batch_size) / total, 1.0)
        bar.progress(prog, text=f"Đang chạy {min(i + batch_size, total)}/{total}...")
        
        time.sleep(1) # Nghỉ 1s tránh spam

    # Hiển thị bảng kết quả
    st.success("Hoàn thành!")
    res_df = pd.DataFrame(final_results)
    st.dataframe(res_df)
    
    # Nút tải xuống
    os.makedirs('output', exist_ok=True)
    fname = f"output/final_map_{datetime.now().strftime('%H%M')}.xlsx"
    res_df.to_excel(fname, index=False)
    
    with open(fname, "rb") as f:
        st.download_button("📥 Tải kết quả Excel", f, file_name="ket_qua_map.xlsx")
