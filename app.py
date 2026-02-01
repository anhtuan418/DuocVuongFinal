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
        df['norm_search'] = df.apply(lambda x: normalize_text(f"{x['ten_thu
