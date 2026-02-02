import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process
import unidecode
import re
import os
import pickle
from collections import Counter

# =============================================================================
# 1. CẤU HÌNH TRANG
# =============================================================================
st.set_page_config(page_title="PharmaMaster: Final Fix", layout="wide", page_icon="💊")

# =============================================================================
# 2. CLASS MACHINE LEARNING (GIỮ NGUYÊN)
# =============================================================================
class PharmaBrain:
    def __init__(self):
        self.brand_memory = {} 
        self.learned_status = False

    def _tokenize(self, text):
        if pd.isna(text): return []
        text = unidecode.unidecode(str(text).lower())
        return re.findall(r"\w+", text)

    def learn(self, history_df, input_col, brand_col):
        brand_counter = {}
        count_learned = 0
        for _, row in history_df.iterrows():
            raw_text = row[input_col]
            true_brand = row[brand_col]
            if pd.isna(true_brand) or pd.isna(raw_text): continue
            tokens = self._tokenize(raw_text)
            for token in tokens:
                if len(token) < 2 or token.isdigit(): continue
                if token not in brand_counter: brand_counter[token] = Counter()
                brand_counter[token][true_brand] += 1

        self.brand_memory = {}
        for token, counts in brand_counter.items():
            most_common_brand, count = counts.most_common(1)[0]
            total = sum(counts.values())
            if total >= 2 and (count / total) > 0.7: 
                self.brand_memory[token] = most_common_brand
                count_learned += 1
        self.learned_status = True
        return count_learned

    def predict_brand(self, raw_text):
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

# =============================================================================
# 3. XỬ LÝ DỮ LIỆU & LOAD FILE
# =============================================================================

def normalize_text(text):
    if pd.isna(text): return ""
    return unidecode.unidecode(str(text).lower()).strip()

def extract_numbers(text):
    """
    Trích xuất số thông minh. 
    Femoston 1/10 -> {1.0, 10.0}
    """
    if pd.isna(text): return set()
    # Thay các ký tự phân cách bằng khoảng trắng để tách số dính nhau (1mg/5mg)
    clean_text = str(text).replace('/', ' ').replace('-', ' ').replace('+', ' ')
    # Regex lấy số thực
    nums = re.findall(r"\d+\.?\d*", clean_text)
    # Chuyển về float để so sánh (1.0 == 1)
    return {float(n) for n in nums}

@st.cache_data
def load_master_data():
    file_path = "data/vtma_standard.csv"
    if not os.path.exists(file_path):
        st.error(f"❌ Không tìm thấy file data tại: {file_path}")
        return None

    try:
        # Ưu tiên đọc utf-8-sig (để xử lý BOM)
        df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
    except:
        try:
            df = pd.read_csv(file_path, sep=None, engine='python', encoding='latin1')
        except Exception as e:
            st.error(f"❌ Lỗi đọc file: {e}")
            return None

    # Chuẩn hóa tên cột: Bỏ BOM, chữ thường, bỏ khoảng trắng
    df.columns = df.columns.str.strip().str.lower().str.replace('\ufeff', '').str.replace('ï»¿', '')
    
    # Mapping Cột (Theo ảnh image_f5fcd8.png của bạn)
    mapping_dict = {
        'ma_vtma': ['ma_thuoc', 'vtma code'],
        'ten_thuoc': ['ten_thuoc', 'product'],
        'hoat_chat': ['hoat_chat', 'molecule'],
        'ten_cong_ty': ['ten_cong_ty', 'manufacturer', 'ten_tap_doan'],
        'ham_luong': ['ham_luong', 'galenic'],
        'dang_bao_che': ['dang_bao_che', 'unit_measure', 'dang_dung'],
        'sku_full': ['ten_day_du', 'sku', 'product_name'] 
    }

    final_rename = {}
    current_cols = df.columns.tolist()
    for std, aliases in mapping_dict.items():
        found = False
        for alias in aliases:
            if alias in current_cols:
                final_rename[alias] = std
                found = True
                break
    
    if final_rename: df.rename(columns=final_rename, inplace=True)
    
    # Tạo các cột chuẩn hóa
    required = ['ma_vtma', 'ten_thuoc', 'ten_cong_ty', 'hoat_chat', 'ham_luong', 'dang_bao_che']
    for col in required:
        if col not in df.columns: df[col] = "" # Tạo cột rỗng nếu thiếu
        df[col] = df[col].astype(str).replace('nan', '')

    df['norm_name'] = df['ten_thuoc'].apply(normalize_text)
    df['norm_brand'] = df['ten_cong_ty'].apply(normalize_text)
    df['norm_active'] = df['hoat_chat'].apply(normalize_text)
    df['norm_strength'] = df['ham_luong'].apply(normalize_text)
    df['norm_form'] = df['dang_bao_che'].apply(normalize_text)
    
    # Search Index để lọc sơ bộ
    df['search_index'] = df.apply(lambda x: f"{x['norm_name']} {x['norm_active']} {x['norm_strength']}", axis=1)

    if 'sku_full' in df.columns and len(df['sku_full']) > 0:
        df['display_name'] = df['sku_full']
    else:
        df['display_name'] = df['ten_thuoc'] + " " + df['ham_luong']

    return df

# =============================================================================
# 4. THUẬT TOÁN TÍNH ĐIỂM (CORE ENGINE - ĐÃ FIX FEMOSTON)
# =============================================================================

def calculate_detailed_score(input_str, row_data, ml_predicted_brand=None):
    norm_input = normalize_text(input_str)
    
    # 1. Tên thuốc (40%)
    score_name = fuzz.token_set_ratio(norm_input, row_data['norm_name'])
    
    # 2. Hãng (20%)
    score_brand = fuzz.partial_ratio(row_data['norm_brand'], norm_input)
    
    # 3. Hoạt chất (20%)
    score_active = 0
    if row_data['norm_active']:
        score_active = fuzz.token_set_ratio(row_data['norm_active'], norm_input)
    else:
        score_active = 50 # Không có dữ liệu hoạt chất thì cho điểm trung bình

    # 4. Hàm lượng (10%) - LOGIC MỚI CHO FEMOSTON 1/10
    input_nums = extract_numbers(input_str)
    row_nums = extract_numbers(row_data['ham_luong'])
    
    score_strength = 0
    if not row_nums:
        score_strength = 50
    elif not input_nums:
        score_strength = 50
    else:
        # Giao thoa số: Input {1, 10}, Row {1, 5} -> Giao {1} -> Sai
        # Input {1, 10}, Row {1, 10} -> Giao {1, 10} -> Đúng
        intersection = input_nums.intersection(row_nums)
        
        if len(intersection) == len(input_nums) and len(intersection) == len(row_nums):
            score_strength = 100 # Khớp hoàn toàn bộ số
        elif len(intersection) > 0:
            # Có khớp 1 phần (Ví dụ khớp số 1 nhưng lệch số 10)
            # PHẠT NẶNG: Nếu số lượng số khác nhau -> Trừ điểm
            score_strength = 40 
        else:
            score_strength = 0 # Không khớp số nào

    # 5. Dạng bào chế (10%)
    score_form = fuzz.partial_ratio(row_data['norm_form'], norm_input)
    
    # TỔNG HỢP
    base_score = (score_name*0.4) + (score_brand*0.2) + (score_active*0.2) + (score_strength*0.1) + (score_form*0.1)
    
    # AI BONUS
    ml_bonus = 0
    if ml_predicted_brand and row_data['norm_brand']:
        if fuzz.token_set_ratio(normalize_text(ml_predicted_brand), row_data['norm_brand']) > 85:
            ml_bonus = 15

    final_score = min(base_score + ml_bonus, 100)

    # TRẢ VỀ DICTIONARY ĐỂ TÁCH CỘT
    return {
        'total': final_score,
        's_name': score_name,
        's_brand': score_brand,
        's_active': score_active,
        's_strength': score_strength,
        's_form': score_form,
        'ml_bonus': ml_bonus
    }

def search_product_v3(input_text, db_df, brain_model, min_score=50, top_n=3):
    predicted_brand = brain_model.predict_brand(input_text)
    norm_input = normalize_text(input_text)
    
    # B1: Lọc sơ bộ (Search Index) - Quan trọng để bắt Femoston
    candidates = process.extract(
        norm_input, 
        db_df['search_index'], 
        limit=100, 
        scorer=fuzz.token_set_ratio
    )
    
    # Lấy index ứng viên (ngưỡng thấp 30% để không bỏ sót)
    candidate_indices = [x[2] for x in candidates if x[1] > 30]
    
    if not candidate_indices: return []

    subset = db_df.iloc[candidate_indices].copy()
    results = []
    
    # B2: Chấm điểm chi tiết
    for idx, row in subset.iterrows():
        scores = calculate_detailed_score(input_text, row, ml_predicted_brand=predicted_brand)
        
        if scores['total'] >= min_score:
            results.append({
                'Mã VTMA': row['ma_vtma'],
                'Tên Thuốc (SKU)': row['display_name'],
                'NSX': row['ten_cong_ty'],
                'Hàm Lượng': row['ham_luong'], # Hiện thêm cột này để check
                'Điểm Tổng': round(scores['total'], 1),
                # Các cột điểm chi tiết
                'Điểm Tên (40%)': int(scores['s_name']),
                'Điểm Hãng (20%)': int(scores['s_brand']),
                'Điểm HoạtChất (20%)': int(scores['s_active']),
                'Điểm HàmLượng (10%)': int(scores['s_strength']),
                'Điểm Dạng (10%)': int(scores['s_form']),
                'AI Bonus': scores['ml_bonus']
            })
            
    results.sort(key=lambda x: x['Điểm Tổng'], reverse=True)
    return results[:top_n]

# =============================================================================
# 5. GIAO DIỆN CHÍNH
# =============================================================================

if 'brain' not in st.session_state:
    st.session_state.brain = PharmaBrain()
    st.session_state.brain.load_model()

if 'db_vtma' not in st.session_state:
    with st.spinner("⏳ Đang tải dữ liệu chuẩn (VTMA)..."):
        df_loaded = load_master_data()
        if df_loaded is not None: st.session_state.db_vtma = df_loaded
        else: st.stop()

with st.sidebar:
    st.header("⚙️ Cấu hình")
    # Điều chỉnh mặc định về 60 để lọc bớt rác
    min_score = st.slider("Min Score (%)", 0, 100, 60) 
    top_n = st.number_input("Top N (Số kết quả)", 1, 10, 3)
    st.info(f"Database: {len(st.session_state.db_vtma)} SKU")

st.title("💊 PharmaMaster: Final Edition (Font Fix + Multi-Columns)")

tab1, tab2 = st.tabs(["🚀 Mapping & Báo Cáo", "🧠 Training AI"])

with tab1:
    st.subheader("Mapping File Excel")
    uploaded = st.file_uploader("Upload file Excel cần map", type=['xlsx', 'csv'])
    
    if uploaded:
        if uploaded.name.endswith('.csv'): df_in = pd.read_csv(uploaded)
        else: df_in = pd.read_excel(uploaded)
        
        st.write(f"Đã nhận {len(df_in)} dòng.")
        col_target = st.selectbox("Chọn cột Tên thuốc:", df_in.columns)
        
        if st.button("🚀 CHẠY MAPPING"):
            all_results = []
            bar = st.progress(0)
            
            for i, row in df_in.iterrows():
                inp = str(row[col_target])
                matches = search_product_v3(inp, st.session_state.db_vtma, st.session_state.brain, min_score, top_n)
                
                if matches:
                    for rank, m in enumerate(matches, 1):
                        all_results.append({
                            'Input_Goc': inp,
                            'Rank': rank,
                            'Ma_VTMA': m['Mã VTMA'],
                            'Ten_VTMA': m['Tên Thuốc (SKU)'],
                            'NSX_Chuan': m['NSX'],
                            'Ham_Luong_Chuan': m['Hàm Lượng'],
                            'Diem_Tong': m['Điểm Tổng'],
                            # Tách thành 5 cột như yêu cầu
                            'Diem_Ten': m['Điểm Tên (40%)'],
                            'Diem_Hang': m['Điểm Hãng (20%)'],
                            'Diem_HoatChat': m['Điểm HoạtChất (20%)'],
                            'Diem_HamLuong': m['Điểm HàmLượng (10%)'],
                            'Diem_Dang': m['Điểm Dạng (10%)'],
                            'AI_Bonus': m['AI Bonus']
                        })
                else:
                    # Dòng trống nếu không tìm thấy
                    empty_row = {
                        'Input_Goc': inp, 'Rank': '-', 'Ma_VTMA': 'Không tìm thấy',
                        'Ten_VTMA': '', 'NSX_Chuan': '', 'Ham_Luong_Chuan': '',
                        'Diem_Tong': 0, 'Diem_Ten':0, 'Diem_Hang':0, 'Diem_HoatChat':0,
                        'Diem_HamLuong':0, 'Diem_Dang':0, 'AI_Bonus':0
                    }
                    all_results.append(empty_row)
                
                bar.progress((i+1)/len(df_in))
                
            df_out = pd.DataFrame(all_results)
            st.success("✅ Hoàn tất!")
            
            # Hiển thị
            st.dataframe(df_out, use_container_width=True)
            
            # Xuất Excel
            excel_name = "ket_qua_map_final.xlsx"
            df_out.to_excel(excel_name, index=False)
            with open(excel_name, "rb") as f:
                st.download_button("📥 Tải Excel (Chuẩn font)", f, excel_name)
                
            # Xuất CSV (FIX LỖI FONT Ở ĐÂY)
            csv = df_out.to_csv(index=False, encoding='utf-8-sig') # Quan trọng: utf-8-sig
            st.download_button("📥 Tải CSV (Chuẩn font)", csv, "ket_qua_map_final.csv", "text/csv")

with tab2:
    st.write("Phần Training AI (Giữ nguyên)...")
