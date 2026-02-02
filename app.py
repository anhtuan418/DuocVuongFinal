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
st.set_page_config(page_title="PharmaMatch: Dữ Liệu Nạp Sẵn", layout="wide")
st.title("💊 PharmaMatch: Hệ Thống Mapping Dược Vương")

# --- 2. CÁC HÀM XỬ LÝ TEXT & SỐ ---
def normalize_text(text):
    if pd.isna(text): return ""
    return unidecode.unidecode(str(text).lower()).strip()

def extract_numbers(text):
    if pd.isna(text): return set()
    nums = re.findall(r"\d+\.?\d*", str(text))
    return set(nums)

def get_match_quality(score):
    if score >= 95: return "Rất cao"
    if score >= 80: return "Cao"
    if score >= 60: return "Trung bình"
    if score > 0: return "Thấp"
    return "Không khớp"

# --- 3. GỌI AI (BATCH 5 SẢN PHẨM) ---
def ai_process_batch(product_list, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        items_str = "\n".join([f"- ID_{i}: {p}" for i, p in enumerate(product_list)])
        prompt = f"""
        Phân tích danh sách thuốc sau:
        {items_str}
        Trả về JSON List Objects (không Markdown), mỗi object gồm:
        - "id": "ID_..."
        - "brand_name": Tên thương mại.
        - "active_ingredient": Hoạt chất.
        - "strength": Hàm lượng.
        - "manufacturer": Tên hãng.
        - "dosage_form": Dạng bào chế.
        """
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(text)
        return {item['id']: item for item in data}
    except:
        return {}

# --- 4. LOGIC TÍNH ĐIỂM CHI TIẾT (5 TIÊU CHÍ) ---
def compare_detailed(ai_data, row):
    # 1. TÊN (40%)
    ai_name = normalize_text(ai_data.get('brand_name', ''))
    score_name = fuzz.token_set_ratio(ai_name, row['norm_name']) * 0.4
    
    # 2. HOẠT CHẤT (20%)
    ai_ingre = normalize_text(ai_data.get('active_ingredient', ''))
    score_ingre = fuzz.token_sort_ratio(ai_ingre, row['norm_ingre']) * 0.2
    
    # 3. HÀM LƯỢNG (20%) - Logic Số học
    ai_str = normalize_text(ai_data.get('strength', ''))
    ai_nums = extract_numbers(ai_str)
    row_nums = extract_numbers(row['norm_strength'])
    score_str = 0
    if ai_nums and row_nums:
        if ai_nums.issubset(row_nums) or row_nums.issubset(ai_nums): score_str = 100
    else: score_str = fuzz.ratio(ai_str, row['norm_strength'])
    weighted_str = score_str * 0.2
    
    # 4. NSX (10%)
    ai_manu = normalize_text(ai_data.get('manufacturer', ''))
    score_manu = fuzz.partial_ratio(ai_manu, row['norm_manu']) * 0.1
    
    # 5. DẠNG (10%)
    ai_form = normalize_text(ai_data.get('dosage_form', ''))
    score_form = fuzz.partial_ratio(ai_form, row['norm_form']) * 0.1
    
    total = score_name + score_ingre + weighted_str + score_manu + score_form
    return {'total': round(total, 1), 'raw_scores': {'name': score_name/0.4, 'ingre': score_ingre/0.2, 'str': score_str, 'manu': score_manu/0.1, 'form': score_form/0.1}}

def find_matches(ai_data, vtma_df, min_score, top_n):
    ai_name = normalize_text(ai_data.get('brand_name', ''))
    candidates = process.extract(ai_name, vtma_df['norm_name'], limit=50, scorer=fuzz.token_set_ratio)
    indices = [x[2] for x in candidates if x[1] >= 40]
    if not indices: return []
    subset = vtma_df.iloc[indices].copy()
    results = []
    for idx, row in subset.iterrows():
        res = compare_detailed(ai_data, row)
        if res['total'] >= min_score:
            results.append({'row': row, 'res': res})
    results.sort(key=lambda x: x['res']['total'], reverse=True)
    return results[:top_n]

# --- 5. TỰ ĐỘNG NẠP DỮ LIỆU VTMA ---
@st.cache_data
def load_fixed_vtma():
    # Đường dẫn file nạp sẵn trong thư mục data trên GitHub
    file_path = "data/vtma_standard.csv"
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            # Chuẩn hóa ngay khi load
            df['norm_name'] = df['ten_thuoc'].apply(normalize_text)
            df['norm_ingre'] = df['hoat_chat'].apply(normalize_text)
            df['norm_strength'] = df['ham_luong'].apply(normalize_text)
            df['norm_manu'] = df['ten_cong_ty'].apply(normalize_text)
            df['norm_form'] = df['dang_bao_che'].apply(normalize_text)
            return df
        except: return None
    return None

# --- 6. GIAO DIỆN ---
with st.sidebar:
    st.header("Cài đặt")
    api_key = st.text_input("Gemini API Key", type="password")
    if not api_key and "GENAI_API_KEY" in st.secrets:
        api_key = st.secrets["GENAI_API_KEY"]
    
    st.divider()
    threshold = st.slider("Độ chính xác tối thiểu (%)", 0, 100, 60)
    top_n = st.number_input("Số mã VTMA tối đa", 1, 10, 1)

# Thực hiện nạp dữ liệu ngầm
vtma_df = load_fixed_vtma()

if vtma_df is not None:
    st.sidebar.success(f"✅ Đã nạp sẵn {len(vtma_df)} mã VTMA")
else:
    st.sidebar.error("❌ Không tìm thấy file dữ liệu nạp sẵn tại 'data/vtma_standard.csv'")
    st.stop()

# Hiển thị khu vực Upload Dược Vương
st.subheader("📂 Tải danh mục Dược Vương")
dv_file = st.file_uploader("Kéo thả file cần map", type=['xlsx', 'csv'])

if dv_file and st.button("🚀 CHẠY MAPPING"):
    if not api_key:
        st.error("Thiếu API Key!")
        st.stop()
        
    if dv_file.name.endswith('.csv'): df_in = pd.read_csv(dv_file)
    else: df_in = pd.read_excel(dv_file)
    
    col_name = df_in.columns[0]
    final_results = []
    input_list = df_in[col_name].astype(str).tolist()
    total = len(input_list)
    bar = st.progress(0, text="Đang xử lý...")
    
    batch_size = 5
    for i in range(0, total, batch_size):
        batch_items = input_list[i : i + batch_size]
        ai_dict = ai_process_batch(batch_items, api_key)
        
        for idx, item in enumerate(batch_items):
            item_id = f"ID_{idx}"
            ai_info = ai_dict.get(item_id, {})
            matches = find_matches(ai_info, vtma_df, threshold, top_n)
            
            row_base = {
                'DV_Input': item, 'VTMA_Code': '', 'Tong_Diem': 0, 'Rank': '-',
                'AI_Ten': ai_info.get('brand_name'), 'VTMA_Ten': '', 'Khop_Ten': '',
                'AI_HoatChat': ai_info.get('active_ingredient'), 'VTMA_HoatChat': '', 'Khop_HoatChat': '',
                'AI_HamLuong': ai_info.get('strength'), 'VTMA_HamLuong': '', 'Khop_HamLuong': '',
                'AI_NSX': ai_info.get('manufacturer'), 'VTMA_NSX': '', 'Khop_NSX': '',
                'AI_DangBaoChe': ai_info.get('dosage_form'), 'VTMA_DangBaoChe': '', 'Khop_DangBaoChe': ''
            }
            
            if not matches:
                r = row_base.copy(); r['VTMA_Code'] = "Không thấy"; final_results.append(r)
            else:
                for rank, m in enumerate(matches, 1):
                    r = row_base.copy(); v_row = m['row']; scores = m['res']['raw_scores']
                    r.update({
                        'VTMA_Code': v_row['ma_thuoc'], 'Tong_Diem': m['res']['total'], 'Rank': f"Top {rank}",
                        'VTMA_Ten': v_row['ten_thuoc'], 'Khop_Ten': get_match_quality(scores['name']),
                        'VTMA_HoatChat': v_row['hoat_chat'], 'Khop_HoatChat': get_match_quality(scores['ingre']),
                        'VTMA_HamLuong': v_row['ham_luong'], 'Khop_HamLuong': get_match_quality(scores['str']),
                        'VTMA_NSX': v_row['ten_cong_ty'], 'Khop_NSX': get_match_quality(scores['manu']),
                        'VTMA_DangBaoChe': v_row['dang_bao_che'], 'Khop_DangBaoChe': get_match_quality(scores['form']),
                    })
                    final_results.append(r)
        
        bar.progress(min((i + batch_size) / total, 1.0))
        time.sleep(1)
        
    res_df = pd.DataFrame(final_results)
    st.success("Xong!")
    st.dataframe(res_df)
    
    os.makedirs('output', exist_ok=True)
    fname = f"output/map_final_{datetime.now().strftime('%H%M')}.xlsx"
    res_df.to_excel(fname, index=False)
    with open(fname, "rb") as f:
        st.download_button("📥 Tải báo cáo chi tiết", f, file_name="ket_qua.xlsx")
