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
st.set_page_config(page_title="PharmaMatch: Final Pro", layout="wide")
st.title("💊 PharmaMatch: Hệ Thống Mapping Dược Phẩm (Chi Tiết)")

# --- 2. CÁC HÀM XỬ LÝ TEXT & SỐ ---
def normalize_text(text):
    if pd.isna(text): return ""
    return unidecode.unidecode(str(text).lower()).strip()

def extract_numbers(text):
    """Lấy tập hợp số từ chuỗi để so sánh hàm lượng (VD: 500mg -> {500})."""
    if pd.isna(text): return set()
    nums = re.findall(r"\d+\.?\d*", str(text))
    return set(nums)

def get_match_quality(score):
    """Đánh giá độ khớp bằng chữ."""
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
        
        # Tạo danh sách sản phẩm để gửi 1 lần
        items_str = "\n".join([f"- ID_{i}: {p}" for i, p in enumerate(product_list)])
        
        prompt = f"""
        Phân tích danh sách thuốc sau đây:
        {items_str}
        
        Yêu cầu trả về JSON dạng List of Objects (Tuyệt đối không dùng Markdown ```json), mỗi object gồm:
        - "id": "ID_..." (Giữ nguyên ID tương ứng)
        - "brand_name": Tên thương mại/Biệt dược.
        - "active_ingredient": Hoạt chất chính.
        - "strength": Hàm lượng/Nồng độ (VD: 500mg, 10%).
        - "manufacturer": Tên hãng/Thương hiệu.
        - "dosage_form": Dạng bào chế.
        """
        
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(text)
        # Chuyển list thành dict để map lại dễ dàng
        return {item['id']: item for item in data}
    except Exception as e:
        return {} # Trả về rỗng nếu lỗi

# --- 4. LOGIC TÍNH ĐIỂM CHI TIẾT (5 TIÊU CHÍ) ---
def compare_detailed(ai_data, row):
    """So sánh dữ liệu AI tìm được với 1 dòng VTMA."""
    
    # 1. TÊN THƯƠNG MẠ
