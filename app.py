import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process
import unidecode
import re
import os
from datetime import datetime

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="PharmaMatch: Local Offline", layout="wide")
st.title("💻 PharmaMatch: Phiên bản Offline (Tốc độ cao)")

# --- 2. CÁC HÀM XỬ LÝ TEXT & SỐ HỌC ---
def normalize_text(text):
    if pd.isna(text): return ""
    # Chuyển về tiếng việt không dấu, chữ thường
    return unidecode.unidecode(str(text).lower()).strip()

def extract_numbers(text):
    """
    Hàm này thay thế AI để đọc hàm lượng.
    Nó tìm tất cả các con số trong chuỗi. 
    VD: "Panadol Extra 500mg vỉ 10" -> Tìm thấy {500, 10}
    """
    if pd.isna(text): return set()
    # Regex tìm số nguyên và số thập phân (VD: 4.5, 0.5)
    nums = re.findall(r"\d+\.?\d*", str(text))
    # Lọc bỏ các số 0 vô nghĩa ở đầu (nếu cần) và chuyển về set để so sánh
    return set(nums)

# --- 3. LOAD DATA VTMA ---
@st.cache_data
def load_vtma_data():
    try:
        # Đường dẫn file
        path = "data/vtma_standard.csv"
        if not os.path.exists(path): return None
        
        df = pd.read_csv(path)
        
        # Tạo cột SEARCH_TEXT gộp tất cả thông tin lại để tìm kiếm tổng quát
        # (Vì input Dược Vương là 1 chuỗi dài, nên ta gộp VTMA lại để so sánh tương đồng)
        df['norm_search'] = df.apply(lambda x: normalize_text(f"{x['ten_thuoc']} {x['hoat_chat']} {x['ham_luong']} {x['ten_cong_ty']}"), axis=1)
        
        # Tạo các cột chuẩn hóa riêng lẻ để tính điểm chi tiết
        df['norm_name'] = df['ten_thuoc'].apply(normalize_text)
        df['norm_strength'] = df['ham_luong'].apply(normalize_text)
        
        return df
    except:
        return None

# --- 4. LOGIC MAPPING (THAY THẾ AI BẰNG THUẬT TOÁN) ---
def local_match(input_raw, vtma_df):
    """
    Logic không dùng AI:
    1. Chuẩn hóa Input.
    2. Quét nhanh tìm 30 ứng viên trong VTMA có chuỗi text giống Input nhất.
    3. Soi kỹ từng ứng viên: Đặc biệt là SO KHỚP SỐ (Hàm lượng).
    """
    norm_input = normalize_text(input_raw)
    input_nums = extract_numbers(input_raw) # Lấy số từ Input
    
    # BƯỚC 1: SÀNG LỌC (Tìm 30 ứng viên sáng giá)
    # So sánh Input với cột 'norm_search' (gộp tên+hoạt chất+hàm lượng) của VTMA
    candidates = process.extract(
        norm_input, 
        vtma_df['norm_search'], 
        limit=30, 
        scorer=fuzz.token_set_ratio
    )
    
    # Lấy index của các ứng viên
    candidate_indices = [x[2] for x in candidates if x[1] >= 40] # Giảm ngưỡng xuống 40 để ko bỏ sót
    
    if not candidate_indices: return None, 0, "Không tìm thấy"

    subset_df = vtma_df.iloc[candidate_indices].copy()
    results = []
    
    # BƯỚC 2: CHẤM ĐIỂM CHI TIẾT (STRICT MODE)
    for idx, row in subset_df.iterrows():
        # Điểm tương đồng văn bản (Max 60đ)
        # So sánh Input với Tên Thuốc hoặc Hoạt Chất
        text_score = fuzz.token_set_ratio(norm_input, row['norm_search']) * 0.6
        
        # Điểm Số Học / Hàm Lượng (Max 40đ) - QUAN TRỌNG NHẤT
        num_score = 0
        row_nums = extract_numbers(row['ham_luong']) # Lấy số từ cột hàm lượng chuẩn
        
        if not input_nums:
            # Nếu Input không có số (VD: "Panadol"), thì bỏ qua check số, dựa vào text
            num_score = 20 
        else:
            # Nếu Input có số (VD: "Zinc 10"), VTMA cũng phải có số 10
            # Logic: Tập số của VTMA phải nằm trong Input hoặc ngược lại
            # VD: VTMA {10} nằm trong Input {10, 100} -> OK
            common_nums = input_nums.intersection(row_nums)
            
            if common_nums: # Nếu có ít nhất 1 số trùng nhau (VD số 10)
                num_score = 40
            else:
                # Nếu Input có số mà map vào dòng không có số nào trùng -> PHẠT
                # VD: Input "Zinc 15", Row "Zinc 10" -> Chung số 0, Phạt!
                num_score = -50 # Trừ điểm cực nặng để loại bỏ
        
        final_score = text_score + num_score
        results.append({'row': row, 'score': final_score})
    
    # Sắp xếp lấy điểm cao nhất
    results.sort(key=lambda x: x['score'], reverse=True)
    
    if results:
        best = results[0]
        # Nếu điểm bị âm (do phạt số) thì coi như không khớp
        if best['score'] < 30: return None, best['score'], "Sai hàm lượng"
        return best['row'], best['score'], "OK"
    else:
        return None, 0, "Không khớp"

# --- 5. GIAO DIỆN ---
# Load data ngay khi vào
vtma_df = load_vtma_data()

st.sidebar.header("Cấu hình Local")
st.sidebar.info("Chế độ này chạy 100% trên máy tính của bạn, không cần Internet để gọi AI.")

# Upload
st.subheader("📂 1. Tải file Dược Vương cần map")
uploaded = st.file_uploader("Chọn file (Excel/CSV)", type=['xlsx', 'csv'])

if uploaded:
    if uploaded.name.endswith('.csv'): df_in = pd.read_csv(uploaded)
    else: df_in = pd.read_excel(uploaded)
    
    st.write(f"Dữ liệu Input: {len(df_in)} dòng.")
    col_name = df_in.columns[0] # Lấy cột đầu tiên
    
    if st.button("🚀 CHẠY MAPPING (OFFLINE)"):
        if vtma_df is None:
            st.error("❌ Chưa có file data/vtma_standard.csv")
            st.stop()
            
        results = []
        bar = st.progress(0, text="Đang xử lý...")
        
        # Chạy vòng lặp (Rất nhanh nên không cần Batch)
        for i, row in df_in.iterrows():
            raw_input = str(row[col_name])
            
            # Gọi hàm map local
            match_row, score, note = local_match(raw_input, vtma_df)
            
            # Ghi kết quả
            res = {
                'DV_Input': raw_input,
                'VTMA_Code': '', 'VTMA_Name': '', 'VTMA_HamLuong': '', 'VTMA_HoatChat': '', 'VTMA_NSX': '',
                'Score': score,
                'Danh_Gia': 'Thấp'
            }
            
            if match_row is not None:
                res.update({
                    'VTMA_Code': match_row['ma_thuoc'],
                    'VTMA_Name': match_row['ten_thuoc'],
                    'VTMA_HamLuong': match_row['ham_luong'],
                    'VTMA_HoatChat': match_row['hoat_chat'],
                    'VTMA_NSX': match_row['ten_cong_ty'],
                    'Danh_Gia': 'Cao' if score > 80 else 'Kiểm tra'
                })
            
            results.append(res)
            # Update progress
            bar.progress((i+1)/len(df_in), text=f"Đang chạy: {raw_input}")
            
        st.success("✅ Hoàn tất!")
        res_df = pd.DataFrame(results)
        st.dataframe(res_df)
        
        # Download
        os.makedirs('output', exist_ok=True)
        fname = f"output/local_map_{datetime.now().strftime('%H%M')}.xlsx"
        res_df.to_excel(fname, index=False)
        with open(fname, "rb") as f:
            st.download_button("📥 Tải kết quả", f, file_name="ket_qua_local.xlsx")

elif vtma_df is None:
    st.warning("⚠️ Chưa tìm thấy file 'data/vtma_standard.csv'. Vui lòng kiểm tra lại folder data.")
