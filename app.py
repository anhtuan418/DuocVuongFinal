import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process
import unidecode
import re
import os
import pickle
from collections import Counter

# --- 1. CẤU HÌNH CLASS MACHINE LEARNING (PHARMA BRAIN) ---
class PharmaBrain:
    def __init__(self):
        self.brand_memory = {}  # Bộ nhớ: Từ khóa -> Tên Hãng Chuẩn
        self.learned_status = False

    def _tokenize(self, text):
        """Tách chuỗi thành từ khóa (tokens)"""
        if pd.isna(text): return []
        text = unidecode.unidecode(str(text).lower())
        return re.findall(r"\w+", text)

    def learn(self, history_df, input_col, brand_col):
        """Học từ file lịch sử cũ"""
        brand_counter = {}
        count_learned = 0
        
        for _, row in history_df.iterrows():
            raw_text = row[input_col]
            true_brand = row[brand_col]
            if pd.isna(true_brand) or pd.isna(raw_text): continue
            
            tokens = self._tokenize(raw_text)
            for token in tokens:
                if len(token) < 2 or token.isdigit(): continue # Bỏ qua từ quá ngắn hoặc số
                
                if token not in brand_counter: brand_counter[token] = Counter()
                brand_counter[token][true_brand] += 1

        # Chỉ nhớ quy luật có độ tin cậy > 70%
        self.brand_memory = {}
        for token, counts in brand_counter.items():
            most_common_brand, count = counts.most_common(1)[0]
            total = sum(counts.values())
            confidence = count / total
            
            if total >= 2 and confidence > 0.7: # Quy tắc lọc nhiễu
                self.brand_memory[token] = most_common_brand
                count_learned += 1
                
        self.learned_status = True
        return count_learned

    def predict_brand(self, raw_text):
        """Dự đoán hãng từ tên thuốc mới nhập"""
        if not self.brand_memory: return None
        tokens = self._tokenize(raw_text)
        detected_brands = []
        for token in tokens:
            if token in self.brand_memory:
                detected_brands.append(self.brand_memory[token])
        
        if detected_brands:
            return Counter(detected_brands).most_common(1)[0][0]
        return None

    def save_model(self):
        with open("pharma_brain.pkl", "wb") as f: pickle.dump(self.brand_memory, f)

    def load_model(self):
        if os.path.exists("pharma_brain.pkl"):
            with open("pharma_brain.pkl", "rb") as f: self.brand_memory = pickle.load(f)
            self.learned_status = True
            return True
        return False

# --- 2. CÁC HÀM XỬ LÝ TEXT & TÍNH ĐIỂM ---
def normalize_text(text):
    if pd.isna(text): return ""
    return unidecode.unidecode(str(text).lower()).strip()

def extract_numbers(text):
    if pd.isna(text): return set()
    return set(re.findall(r"\d+\.?\d*", str(text)))

def calculate_weighted_score(input_str, row_data, ml_predicted_brand=None):
    """
    Tính điểm tổng hợp: Text + Số + ML Bonus
    """
    norm_input = normalize_text(input_str)
    
    # 1. Điểm Tên (40%)
    score_name = fuzz.token_set_ratio(norm_input, row_data['norm_name'])
    
    # 2. Điểm Hãng (20%)
    score_brand = fuzz.partial_ratio(row_data['norm_brand'], norm_input)
    
    # 3. Điểm Hoạt chất (20%)
    score_active = fuzz.token_set_ratio(row_data['norm_active'], norm_input)
    
    # 4. Điểm Hàm lượng (10%) - Logic so khớp số
    input_nums = extract_numbers(input_str)
    row_nums = extract_numbers(row_data['ham_luong'])
    if not row_nums: score_strength = 50
    elif input_nums.intersection(row_nums): score_strength = 100
    else: score_strength = 0
    
    # 5. Điểm Dạng bào chế (10%)
    score_form = fuzz.partial_ratio(row_data['norm_form'], norm_input)
    
    # --- TÍNH ĐIỂM CƠ BẢN ---
    base_score = (score_name*0.4) + (score_brand*0.2) + (score_active*0.2) + (score_strength*0.1) + (score_form*0.1)
    
    # --- 6. ML BONUS (ĐIỂM THƯỞNG AI) ---
    ml_bonus = 0
    match_ml = "No"
    
    # Nếu AI dự đoán được hãng, và hãng đó trùng với dòng dữ liệu này
    if ml_predicted_brand:
        # So sánh hãng dự đoán với hãng trong data (fuzzy nhẹ để tránh lỗi chính tả)
        similarity = fuzz.token_set_ratio(normalize_text(ml_predicted_brand), row_data['norm_brand'])
        if similarity > 85: # Nếu khớp hãng > 85%
            ml_bonus = 15 # CỘNG 15 ĐIỂM THƯỞNG
            match_ml = "Yes"
            
    final_score = base_score + ml_bonus
    
    return {
        'total': min(final_score, 100), # Max là 100
        'detail': f"Tên:{int(score_name)} | Hãng:{int(score_brand)} | Số:{int(score_strength)} | ML_Bonus:+{ml_bonus}",
        'ml_match': match_ml
    }

# --- 3. HÀM TÌM KIẾM CHÍNH ---
def search_product(input_text, db_df, brain_model, min_score=50, top_n=1):
    # Bước 1: Hỏi ý kiến AI trước
    predicted_brand = brain_model.predict_brand(input_text)
    
    norm_input = normalize_text(input_text)
    
    # Bước 2: Lọc sơ bộ 50 ứng viên bằng Tên thuốc
    candidates = process.extract(norm_input, db_df['norm_name'], limit=50, scorer=fuzz.token_set_ratio)
    candidate_indices = [x[2] for x in candidates if x[1] > 30] # Lấy nếu giống > 30%
    
    if not candidate_indices: return []

    subset = db_df.iloc[candidate_indices].copy()
    results = []
    
    # Bước 3: Chấm điểm chi tiết từng ứng viên
    for idx, row in subset.iterrows():
        # Truyền dự đoán của AI vào hàm chấm điểm
        scoring = calculate_weighted_score(input_text, row, ml_predicted_brand=predicted_brand)
        
        if scoring['total'] >= min_score:
            results.append({
                'Mã VTMA': row['ma_vtma'],
                'Tên VTMA': row['ten_thuoc'],
                'NSX': row['ten_cong_ty'],
                'AI Dự Đoán NSX': predicted_brand if predicted_brand else "-",
                'Điểm Tổng': round(scoring['total'], 1),
                'Chi Tiết Điểm': scoring['detail']
            })
            
    # Sắp xếp và cắt Top N
    results.sort(key=lambda x: x['Điểm Tổng'], reverse=True)
    return results[:top_n]

# --- 4. GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="PharmaMaster AI", layout="wide", page_icon="💊")

# Khởi tạo App State
if 'brain' not in st.session_state:
    st.session_state.brain = PharmaBrain()
    st.session_state.brain.load_model() # Tự động load nếu có file cũ

if 'db_vtma' not in st.session_state:
    # --- MOCK DATA (DỮ LIỆU GIẢ LẬP ĐỂ CHẠY NGAY) ---
    data = {
        'ma_vtma': ['V01', 'V02', 'V03', 'V04'],
        'ten_thuoc': ['Hapacol 650', 'Panadol Extra', 'Efferalgan', 'Augmentin 1g'],
        'ten_cong_ty': ['DHG Pharma', 'GSK', 'UPSA', 'GSK'],
        'hoat_chat': ['Paracetamol', 'Para, Cafein', 'Paracetamol', 'Amoxicillin'],
        'ham_luong': ['650mg', '500mg', '500mg', '1g'],
        'dang_bao_che': ['Viên nén', 'Viên nén', 'Sủi', 'Bột']
    }
    df = pd.DataFrame(data)
    # Chuẩn hóa Data 1 lần
    df['norm_name'] = df['ten_thuoc'].apply(normalize_text)
    df['norm_brand'] = df['ten_cong_ty'].apply(normalize_text)
    df['norm_active'] = df['hoat_chat'].apply(normalize_text)
    df['norm_form'] = df['dang_bao_che'].apply(normalize_text)
    df['norm_strength'] = df['ham_luong'].apply(normalize_text)
    st.session_state.db_vt
