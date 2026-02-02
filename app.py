import streamlit as st
import pandas as pd
import google.generativeai as genai
from rapidfuzz import fuzz, process
import unidecode
import json
import re
import time
from datetime import datetime
import os

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="PharmaMatch: Debug Mode", layout="wide")
st.title("💊 PharmaMatch: Hệ Thống Mapping (Có Debug)")

# --- 2. CÁC HÀM XỬ LÝ ---
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

# --- 3. GỌI AI ---
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
    except Exception as e:
        # Debug: In lỗi ra nếu AI hỏng
        st.error(f"Lỗi gọi AI: {e}")
        return {}

# --- 4. TÍNH ĐIỂM ---
def compare_detailed(ai_data, row):
    # 1. TÊN (40%)
    ai_name = normalize_text(ai_data.get('brand_name', ''))
    score_name = fuzz.token_set_ratio(ai_name, row['norm_name']) * 0.4
    
    # 2. HOẠT CHẤT (20%)
    ai_ingre = normalize_text(ai_data.get('active_ingredient', ''))
    score_ingre = fuzz.token_sort_ratio(ai_ingre, row['norm_ingre']) * 0.2
    
    # 3. HÀM LƯỢNG (20%)
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
    
    return {
        'total': round(total, 1),
        'raw_scores': {'name': score_name/0.4, 'ingre': score_ingre/0.2, 'str': score_str, 'manu': score_manu/0.1, 'form': score_form/0.1}
    }

def find_matches(ai_data, vtma_df, min_score, top_n):
    ai_name = normalize_text(ai_data.get('brand_name', ''))
    # Nếu AI không tìm ra tên -> Dùng luôn tên gốc từ Dược Vương để search
    if not ai_name: 
        return []

    candidates = process.extract(ai_name, vtma_df['norm_name'], limit=50, scorer=fuzz.token_set_ratio)
    indices = [x[2] for x in candidates if x[1] >= 30] # Hạ ngưỡng tìm sơ bộ xuống 30
    
    if not indices: return []
    subset = vtma_df.iloc[indices].copy()
    results = []
    
    for idx, row in subset.iterrows():
        res = compare_detailed(ai_data, row)
        if res['total'] >= min_score:
            results.append({'row': row, 'res': res})
            
    results.sort(key=lambda x: x['res']['total'], reverse=True)
    return results[:top_n]

# --- 5. GIAO DIỆN ---
with st.sidebar:
    st.header("Cấu hình")
    api_key = st.text_input("Gemini API Key", type="password")
    if not api_key and "GENAI_API_KEY" in st.secrets:
        api_key = st.secrets["GENAI_API_KEY"]
    
    st.divider()
    st.info("File VTMA")
    vtma_file = st.file_uploader("Upload VTMA", type=['csv'])
    
    st.divider()
    # MẶC ĐỊNH HẠ THẤP NGƯỠNG ĐỂ TEST
    threshold = st.slider("Độ chính xác tối thiểu (%)", 0, 100, 30, help="Hãy để thấp (30-40) để xem có ra kết quả không")
    top_n = st.number_input("Top N", 1, 10, 3)
    
    st.divider()
    # NÚT DEBUG QUAN TRỌNG
    debug_mode = st.checkbox("🔍 Bật chế độ Soi Lỗi (Debug)", value=True)

# LOAD VTMA
vtma_df = None
if vtma_file:
    try: vtma_df = pd.read_csv(vtma_file)
    except: pass
elif os.path.exists("data/vtma_standard.csv"):
    vtma_df = pd.read_csv("data/vtma_standard.csv")
    
if vtma_df is not None:
    # Chuẩn hóa
    for col, src in [('norm_name','ten_thuoc'), ('norm_ingre','hoat_chat'), ('norm_strength','ham_luong'), ('norm_manu','ten_cong_ty'), ('norm_form','dang_bao_che')]:
        if src in vtma_df.columns:
            vtma_df[col] = vtma_df[src].apply(normalize_text)
        else:
            vtma_df[col] = "" # Tạo cột rỗng nếu thiếu

st.subheader("Upload Dược Vương")
dv_file = st.file_uploader("File Input", type=['xlsx', 'csv'])

if dv_file and st.button("🚀 CHẠY NGAY"):
    if vtma_df is None or not api_key:
        st.error("Thiếu Data VTMA hoặc API Key")
        st.stop()
        
    if dv_file.name.endswith('.csv'): df_in = pd.read_csv(dv_file)
    else: df_in = pd.read_excel(dv_file)
    
    col_name = df_in.columns[0]
    
    # DEBUG: Kiểm tra file input
    if debug_mode:
        st.warning(f"🔍 DEBUG: Đang đọc cột '{col_name}'. Dòng đầu tiên là: {df_in.iloc[0][col_name]}")
    
    final_results = []
    input_list = df_in[col_name].astype(str).tolist()
    # Test chạy thử 5 dòng đầu nếu đang debug
    run_list = input_list[:5] if debug_mode else input_list 
    
    bar = st.progress(0)
    batch_size = 5
    
    for i in range(0, len(run_list), batch_size):
        batch_items = run_list[i : i + batch_size]
        try: ai_dict = ai_process_batch(batch_items, api_key)
        except: ai_dict = {}
        
        # DEBUG: Kiểm tra AI trả về gì
        if debug_mode and i == 0:
            st.code(f"🔍 DEBUG AI Trả lời: {json.dumps(ai_dict, ensure_ascii=False, indent=2)}")

        for idx, item in enumerate(batch_items):
            item_id = f"ID_{idx}"
            ai_info = ai_dict.get(item_id, {})
            
            matches = find_matches(ai_info, vtma_df, threshold, top_n)
            
            # DEBUG: Nếu không tìm thấy, in ra tại sao
            if debug_mode and not matches:
                st.write(f"❌ '{item}' -> Điểm thấp hơn {threshold} hoặc AI không tách được tên.")

            row_base = {
                'DV_Input': item, 'VTMA_Code': '', 'Tong_Diem': 0,
                'AI_Ten': ai_info.get('brand_name'), 'VTMA_Ten': '', 
                'AI_HamLuong': ai_info.get('strength'), 'VTMA_HamLuong': ''
            }
            
            if not matches:
                r = row_base.copy()
                r['VTMA_Code'] = "Không tìm thấy"
                final_results.append(r)
            else:
                for rank, m in enumerate(matches, 1):
                    r = row_base.copy()
                    v_row = m['row']
                    res = m['res']
                    scores = res['raw_scores']
                    r.update({
                        'VTMA_Code': v_row['ma_thuoc'],
                        'Tong_Diem': res['total'],
                        'VTMA_Ten': v_row['ten_thuoc'],
                        'VTMA_HamLuong': v_row['ham_luong'],
                        'ChiTiet_Diem': f"Tên:{int(scores['name'])} HL:{int(scores['str'])}"
                    })
                    final_results.append(r)
                    
        bar.progress((i+batch_size)/len(run_list))
        time.sleep(1)

    res_df = pd.DataFrame(final_results)
    st.dataframe(res_df)
    
    if debug_mode:
        st.info("ℹ️ Đang chạy chế độ Debug (chỉ 5 dòng đầu). Hãy tắt Debug để chạy full.")
        
