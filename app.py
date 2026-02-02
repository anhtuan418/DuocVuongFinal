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
st.set_page_config(page_title="PharmaMatch: Chi Tiết 15 Cột", layout="wide")
st.title("💊 PharmaMatch: Báo Cáo Chi Tiết Độ Khớp (15 Cột)")

# --- 2. CÁC HÀM XỬ LÝ ---
def normalize_text(text):
    if pd.isna(text): return ""
    return unidecode.unidecode(str(text).lower()).strip()

def extract_numbers(text):
    """Lấy tập hợp số từ chuỗi."""
    if pd.isna(text): return set()
    nums = re.findall(r"\d+\.?\d*", str(text))
    return set(nums)

def get_match_quality(score):
    """Chuyển điểm số thành chữ đánh giá."""
    if score >= 95: return "Rất cao"
    if score >= 80: return "Cao"
    if score >= 60: return "Trung bình"
    if score > 0: return "Thấp"
    return "Không khớp"

# --- 3. LOAD DATA VTMA ---
@st.cache_data
def load_vtma_data():
    try:
        path = "data/vtma_standard.csv"
        if not os.path.exists(path):
            if os.path.exists("Data/vtma_standard.csv"): path = "Data/vtma_standard.csv"
            else: return None
            
        df = pd.read_csv(path)
        # Chuẩn hóa dữ liệu
        df['norm_name'] = df['ten_thuoc'].apply(normalize_text)
        df['norm_ingre'] = df['hoat_chat'].apply(normalize_text)
        df['norm_strength'] = df['ham_luong'].apply(normalize_text)
        df['norm_manu'] = df['ten_cong_ty'].apply(normalize_text)
        df['norm_form'] = df['dang_bao_che'].apply(normalize_text)
        return df
    except:
        return None

# --- 4. GỌI AI (BATCH PROCESSING) ---
def ai_process_batch(product_list, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        items_str = "\n".join([f"- ID_{i}: {p}" for i, p in enumerate(product_list)])
        
        prompt = f"""
        Phân tích danh sách thuốc sau:
        {items_str}
        
        Trả về JSON List Objects, mỗi object gồm:
        - "id": "ID_..."
        - "brand_name": Tên thương mại (Biệt dược).
        - "active_ingredient": Hoạt chất chính.
        - "strength": Hàm lượng/Nồng độ (VD: 500mg, 10%).
        - "manufacturer": Tên hãng/Thương hiệu.
        - "dosage_form": Dạng bào chế.
        """
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(text)
        return {item['id']: item for item in data}
    except:
        return {}

# --- 5. LOGIC TÍNH ĐIỂM (CÓ TÁCH CHI TIẾT) ---
def compare_detailed(ai_data, row):
    """Hàm này vừa tính tổng điểm, vừa trả về chi tiết từng thành phần."""
    
    # 1. TÊN THƯƠNG MẠI (40%)
    ai_name = normalize_text(ai_data.get('brand_name', ''))
    score_name_raw = fuzz.token_set_ratio(ai_name, row['norm_name'])
    score_total = score_name_raw * 0.4
    
    # 2. HOẠT CHẤT (20%)
    ai_ingre = normalize_text(ai_data.get('active_ingredient', ''))
    score_ingre_raw = fuzz.token_sort_ratio(ai_ingre, row['norm_ingre'])
    score_total += score_ingre_raw * 0.2
    
    # 3. HÀM LƯỢNG (20%) - Logic Số học
    ai_strength = normalize_text(ai_data.get('strength', ''))
    ai_nums = extract_numbers(ai_strength)
    row_nums = extract_numbers(row['norm_strength'])
    
    score_str_raw = 0
    if ai_nums and row_nums:
        if ai_nums.issubset(row_nums) or row_nums.issubset(ai_nums):
            score_str_raw = 100
        else:
            score_str_raw = 0 # Sai số
    else:
        score_str_raw = fuzz.ratio(ai_strength, row['norm_strength'])
    score_total += score_str_raw * 0.2
    
    # 4. NHÀ SẢN XUẤT (10%)
    ai_manu = normalize_text(ai_data.get('manufacturer', ''))
    score_manu_raw = fuzz.partial_ratio(ai_manu, row['norm_manu'])
    score_total += score_manu_raw * 0.1
    
    # 5. DẠNG BÀO CHẾ (10%)
    ai_form = normalize_text(ai_data.get('dosage_form', ''))
    score_form_raw = fuzz.partial_ratio(ai_form, row['norm_form'])
    score_total += score_form_raw * 0.1
    
    return {
        'total_score': round(score_total, 1),
        'details': {
            'name': score_name_raw,
            'ingre': score_ingre_raw,
            'strength': score_str_raw,
            'manu': score_manu_raw,
            'form': score_form_raw
        }
    }

def find_top_matches(ai_data, vtma_df, min_score, top_n):
    # Lọc nhanh bằng tên
    ai_name = normalize_text(ai_data.get('brand_name', ''))
    candidates = process.extract(ai_name, vtma_df['norm_name'], limit=50, scorer=fuzz.token_set_ratio)
    indices = [x[2] for x in candidates if x[1] >= 40]
    
    if not indices: return []

    subset = vtma_df.iloc[indices].copy()
    results = []
    
    for idx, row in subset.iterrows():
        # Tính toán chi tiết
        calc = compare_detailed(ai_data, row)
        if calc['total_score'] >= min_score:
            results.append({
                'row': row,
                'score': calc['total_score'],
                'details': calc['details']
            })
            
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_n]

# --- 6. GIAO DIỆN ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    api_key = st.text_input("Gemini API Key", type="password")
    if not api_key and "GENAI_API_KEY" in st.secrets:
        api_key = st.secrets["GENAI_API_KEY"]
    
    st.divider()
    threshold = st.slider("Tỉ lệ khớp tối thiểu (%)", 0, 100, 60)
    top_n = st.number_input("Số mã VTMA tối đa (Top N)", 1, 10, 1)
    batch_size = st.slider("Batch Size", 5, 20, 10)

vtma_df = load_vtma_data()

st.subheader("📂 Upload Dược Vương")
uploaded = st.file_uploader("File Input (Excel/CSV)", type=['xlsx', 'csv'])

if uploaded:
    if uploaded.name.endswith('.csv'): df_in = pd.read_csv(uploaded)
    else: df_in = pd.read_excel(uploaded)
    
    if st.button("🚀 CHẠY MAPPING CHI TIẾT"):
        if vtma_df is None or not api_key:
            st.error("Thiếu Data hoặc API Key")
            st.stop()

        col_name = df_in.columns[0]
        final_results = []
        input_list = df_in[col_name].astype(str).tolist()
        total = len(input_list)
        bar = st.progress(0, text="Đang xử lý...")
        
        for i in range(0, total, batch_size):
            batch_items = input_list[i : i + batch_size]
            try: ai_dict = ai_process_batch(batch_items, api_key)
            except: ai_dict = {}
            
            for idx, item_name in enumerate(batch_items):
                item_id = f"ID_{idx}"
                ai_info = ai_dict.get(item_id, {})
                
                matches = find_top_matches(ai_info, vtma_df, threshold, top_n)
                
                # Hàm helper để tạo dòng kết quả rỗng (khi không tìm thấy)
                base_res = {
                    'DV_Input': item_name,
                    'VTMA_Code': '', 'Tong_Diem': 0, 'Xep_Hang': '-',
                    # 1. Tên
                    'AI_Ten': ai_info.get('brand_name'), 'VTMA_Ten': '', 'Khop_Ten': '',
                    # 2. Hoạt chất
                    'AI_HoatChat': ai_info.get('active_ingredient'), 'VTMA_HoatChat': '', 'Khop_HoatChat': '',
                    # 3. Hàm lượng
                    'AI_HamLuong': ai_info.get('strength'), 'VTMA_HamLuong': '', 'Khop_HamLuong': '',
                    # 4. NSX
                    'AI_NSX': ai_info.get('manufacturer'), 'VTMA_NSX': '', 'Khop_NSX': '',
                    # 5. Dạng bào chế
                    'AI_DangBaoChe': ai_info.get('dosage_form'), 'VTMA_DangBaoChe': '', 'Khop_DangBaoChe': ''
                }
                
                if not matches:
                    res_row = base_res.copy()
                    res_row['VTMA_Name'] = 'Không tìm thấy'
                    final_results.append(res_row)
                else:
                    for rank, m in enumerate(matches, 1):
                        row = m['row']
                        det = m['details']
                        res_row = base_res.copy()
                        
                        # Điền thông tin VTMA và đánh giá
                        res_row.update({
                            'VTMA_Code': row['ma_thuoc'],
                            'Tong_Diem': m['score'],
                            'Xep_Hang': f"Top {rank}",
                            
                            'VTMA_Ten': row['ten_thuoc'], 
                            'Khop_Ten': get_match_quality(det['name']),
                            
                            'VTMA_HoatChat': row['hoat_chat'], 
                            'Khop_HoatChat': get_match_quality(det['ingre']),
                            
                            'VTMA_HamLuong': row['ham_luong'], 
                            'Khop_HamLuong': get_match_quality(det['strength']),
                            
                            'VTMA_NSX': row['ten_cong_ty'], 
                            'Khop_NSX': get_match_quality(det['manu']),
                            
                            'VTMA_DangBaoChe': row['dang_bao_che'], 
                            'Khop_DangBaoChe': get_match_quality(det['form'])
                        })
                        final_results.append(res_row)
            
            bar.progress(min((i + batch_size) / total, 1.0))
            time.sleep(1)

        res_df = pd.DataFrame(final_results)
        # Sắp xếp để dễ nhìn
        res_df.sort_values(by=['DV_Input', 'Tong_Diem'], ascending=[True, False], inplace=True)
        
        st.success("Hoàn tất!")
        st.dataframe(res_df)
        
        os.makedirs('output', exist_ok=True)
        fname = f"output/map_chitiet_{datetime.now().strftime('%H%M')}.xlsx"
        res_df.to_excel(fname, index=False)
        with open(fname, "rb") as f:
            st.download_button("📥 Tải Báo Cáo Chi Tiết (Excel)", f, file_name="ket_qua_chi_tiet.xlsx")

elif vtma_df is None:
    st.warning("⚠️ Chưa tìm thấy Data.")
