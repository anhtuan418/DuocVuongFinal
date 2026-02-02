import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process
import unidecode
import re
import os
import pickle
from collections import Counter
import google.generativeai as genai
import time

# =============================================================================
# 1. CẤU HÌNH TRANG
# =============================================================================
st.set_page_config(page_title="PharmaMaster AI", layout="wide", page_icon="🧬")

# =============================================================================
# 2. CLASS GEMINI AI (FIXED & DEBUG MODE)
# =============================================================================
class GeminiAgent:
    def __init__(self, api_key, model_pref='gemini-1.5-flash'):
        self.is_ready = False
        self.current_model = "None"
        self.available_models = []
        
        if api_key:
            try:
                genai.configure(api_key=api_key)
                
                # Lấy danh sách model thực tế từ Google
                self.available_models = [m.name.replace('models/', '') for m in genai.list_models()]
                
                # Ưu tiên model người dùng chọn
                if model_pref in self.available_models:
                    self.model = genai.GenerativeModel(model_pref)
                    self.current_model = model_pref
                # Nếu không có, thử ép dùng Flash (thường luôn có)
                elif 'gemini-1.5-flash' in self.available_models:
                    self.model = genai.GenerativeModel('gemini-1.5-flash')
                    self.current_model = 'gemini-1.5-flash'
                # Nếu vẫn không có, thử Pro mới nhất
                elif 'gemini-1.5-pro' in self.available_models:
                    self.model = genai.GenerativeModel('gemini-1.5-pro')
                    self.current_model = 'gemini-1.5-pro'
                # Cuối cùng mới thử 1.0 pro (nhưng tên là gemini-1.0-pro chứ không phải gemini-pro)
                elif 'gemini-1.0-pro' in self.available_models:
                    self.model = genai.GenerativeModel('gemini-1.0-pro')
                    self.current_model = 'gemini-1.0-pro'
                else:
                    # Cố gắng khởi tạo dù không tìm thấy trong list (đôi khi list bị ẩn)
                    self.model = genai.GenerativeModel(model_pref)
                    self.current_model = model_pref + " (Forced)"

                self.is_ready = True
            except Exception as e:
                self.error_msg = str(e)
                self.is_ready = False
        else:
            self.is_ready = False

    def smart_match(self, input_drug, candidates_df):
        if not self.is_ready: return "⚠️ Lỗi: Chưa nhập API Key hoặc Key sai"

        candidates_str = ""
        for idx, row in candidates_df.iterrows():
            candidates_str += f"- ID: {row['ma_vtma']} | Tên: {row['ten_thuoc']} | HL: {row['ham_luong']} | NSX: {row['ten_cong_ty']}\n"

        prompt = f"""
        Bạn là Dược sĩ. Tìm mã thuốc chuẩn (ID) cho sản phẩm đầu vào.
        INPUT: "{input_drug}"
        DATABASE:
        {candidates_str}
        YÊU CẦU: Chọn 1 ID khớp nhất.
        TRẢ LỜI 1 DÒNG: ID_CHON | ĐỘ_TIN_CẬY | LÝ DO
        Ví dụ: VTMA_001 | Cao | Khớp tên và hãng
        Nếu không khớp >70%, trả về: "NONE | - | Không tìm thấy"
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"AI Error: {str(e)}"

# =============================================================================
# 3. CLASS MACHINE LEARNING (PHARMA BRAIN)
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
# 4. XỬ LÝ DỮ LIỆU & LOAD FILE
# =============================================================================
def normalize_text(text):
    if pd.isna(text): return ""
    return unidecode.unidecode(str(text).lower()).strip()

def extract_numbers(text):
    if pd.isna(text): return set()
    clean_text = str(text).replace('/', ' ').replace('-', ' ').replace('+', ' ')
    nums = re.findall(r"\d+\.?\d*", clean_text)
    return {float(n) for n in nums}

@st.cache_data
def load_master_data():
    file_path = "data/vtma_standard.csv"
    if not os.path.exists(file_path):
        st.error(f"❌ Không tìm thấy file data tại: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
    except:
        try:
            df = pd.read_csv(file_path, sep=None, engine='python', encoding='latin1')
        except Exception as e:
            st.error(f"❌ Lỗi đọc file: {e}")
            return None

    df.columns = df.columns.str.strip().str.lower().str.replace('\ufeff', '').str.replace('ï»¿', '')
    
    mapping_dict = {
        'ma_vtma': ['ma_thuoc', 'vtma code', 'code'],
        'ten_thuoc': ['ten_thuoc', 'product', 'name'],
        'hoat_chat': ['hoat_chat', 'molecule'],
        'ten_cong_ty': ['ten_cong_ty', 'manufacturer', 'ten_tap_doan'],
        'ham_luong': ['ham_luong', 'galenic', 'nong do'],
        'dang_bao_che': ['dang_bao_che', 'unit_measure', 'dang_dung'],
        'sku_full': ['ten_day_du', 'sku', 'product_name'] 
    }

    final_rename = {}
    current_cols = df.columns.tolist()
    for std, aliases in mapping_dict.items():
        for alias in aliases:
            if alias in current_cols:
                final_rename[alias] = std
                break
    
    if final_rename: df.rename(columns=final_rename, inplace=True)
    
    required = ['ma_vtma', 'ten_thuoc', 'ten_cong_ty', 'hoat_chat', 'ham_luong', 'dang_bao_che']
    for col in required:
        if col not in df.columns: df[col] = "" 
        df[col] = df[col].astype(str).replace('nan', '')

    df['norm_name'] = df['ten_thuoc'].apply(normalize_text)
    df['norm_brand'] = df['ten_cong_ty'].apply(normalize_text)
    df['norm_active'] = df['hoat_chat'].apply(normalize_text)
    df['norm_strength'] = df['ham_luong'].apply(normalize_text)
    df['norm_form'] = df['dang_bao_che'].apply(normalize_text)
    
    df['search_index'] = df.apply(lambda x: f"{x['norm_name']} {x['norm_active']} {x['norm_strength']}", axis=1)

    if 'sku_full' in df.columns and len(df['sku_full']) > 0:
        df['display_name'] = df['sku_full']
    else:
        df['display_name'] = df['ten_thuoc'] + " " + df['ham_luong']

    return df

# =============================================================================
# 5. CORE ENGINE (FUZZY LOGIC)
# =============================================================================
def calculate_detailed_score(input_str, row_data, ml_predicted_brand=None):
    norm_input = normalize_text(input_str)
    score_name = fuzz.token_set_ratio(norm_input, row_data['norm_name'])
    score_brand = fuzz.partial_ratio(row_data['norm_brand'], norm_input)
    
    score_active = 0
    if row_data['norm_active']: score_active = fuzz.token_set_ratio(row_data['norm_active'], norm_input)
    else: score_active = 50 

    input_nums = extract_numbers(input_str)
    row_nums = extract_numbers(row_data['ham_luong'])
    score_strength = 0
    if not row_nums or not input_nums: score_strength = 50
    else:
        intersection = input_nums.intersection(row_nums)
        if len(intersection) == len(input_nums) and len(intersection) == len(row_nums): score_strength = 100 
        elif len(intersection) > 0: score_strength = 40 
        else: score_strength = 0

    score_form = fuzz.partial_ratio(row_data['norm_form'], norm_input)
    
    base_score = (score_name*0.4) + (score_brand*0.2) + (score_active*0.2) + (score_strength*0.1) + (score_form*0.1)
    
    ml_bonus = 0
    if ml_predicted_brand and row_data['norm_brand']:
        if fuzz.token_set_ratio(normalize_text(ml_predicted_brand), row_data['norm_brand']) > 85:
            ml_bonus = 15

    return {
        'total': min(base_score + ml_bonus, 100),
        's_name': score_name, 's_brand': score_brand, 's_active': score_active,
        's_strength': score_strength, 's_form': score_form, 'ml_bonus': ml_bonus
    }

def get_candidates(input_text, db_df, limit=20):
    if 'search_index' not in db_df.columns:
        db_df['search_index'] = db_df.apply(lambda x: f"{x['norm_name']} {x['norm_active']} {x['norm_strength']}", axis=1)
    
    norm_input = normalize_text(input_text)
    candidates = process.extract(norm_input, db_df['search_index'], limit=limit, scorer=fuzz.token_set_ratio)
    indices = [x[2] for x in candidates]
    return db_df.iloc[indices].copy()

def search_product_v3(input_text, db_df, brain_model, min_score=50, top_n=3):
    predicted_brand = brain_model.predict_brand(input_text)
    subset = get_candidates(input_text, db_df, limit=50)
    
    results = []
    for idx, row in subset.iterrows():
        scores = calculate_detailed_score(input_text, row, ml_predicted_brand=predicted_brand)
        if scores['total'] >= min_score:
            results.append({
                'Mã VTMA': row['ma_vtma'],
                'Tên Thuốc (SKU)': row['display_name'],
                'NSX': row['ten_cong_ty'],
                'Hàm Lượng': row['ham_luong'], 
                'Điểm Tổng': round(scores['total'], 1),
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
# 6. GIAO DIỆN STREAMLIT
# =============================================================================

if 'brain' not in st.session_state:
    st.session_state.brain = PharmaBrain()
    st.session_state.brain.load_model()

if 'db_vtma' not in st.session_state:
    with st.spinner("⏳ Đang tải dữ liệu chuẩn..."):
        df_loaded = load_master_data()
        if df_loaded is not None: st.session_state.db_vtma = df_loaded
        else: st.stop()

# --- SIDEBAR: CẤU HÌNH API ---
with st.sidebar:
    st.header("🤖 Cấu hình Gemini AI")
    api_key = st.text_input("Nhập Google API Key", type="password")
    
    model_option = st.radio(
        "Chọn Phiên bản AI:",
        ("Gemini 1.5 Flash (Khuyên dùng)", "Gemini 1.5 Pro"),
        index=0
    )
    
    selected_model_name = 'gemini-1.5-flash' if "Flash" in model_option else 'gemini-1.5-pro'
    
    if api_key:
        st.session_state.gemini = GeminiAgent(api_key, selected_model_name)
        
        # --- HIỂN THỊ TRẠNG THÁI KẾT NỐI ---
        if st.session_state.gemini.is_ready:
            st.success(f"✅ Đã kết nối: {st.session_state.gemini.current_model}")
            
            # NÚT DEBUG ĐỂ BẠN KIỂM TRA MODEL
            with st.expander("🛠️ Debug: Xem Model khả dụng"):
                if st.button("🆘 Kiểm tra danh sách Model"):
                    st.write(st.session_state.gemini.available_models)
        else:
            st.error("❌ Kết nối thất bại.")
    else:
        st.warning("⚠️ Cần API Key để dùng AI")
        st.session_state.gemini = GeminiAgent(None)

    st.divider()
    st.header("⚙️ Cấu hình Map")
    min_score = st.slider("Min Score (%)", 0, 100, 60) 
    top_n = st.number_input("Top N", 1, 10, 3)
    
    st.subheader("⚙️ Cấu hình AI Rà Soát")
    threshold_ai = st.number_input("Ngưỡng kích hoạt AI (xx)", 0, 100, 70)

st.title("🧬 PharmaMaster: AI-Powered Mapping")

tab1, tab2 = st.tabs(["🚀 Mapping & Gemini", "🧠 Training Model"])

with tab1:
    st.subheader("Mapping Dữ Liệu")
    uploaded = st.file_uploader("Upload file Excel", type=['xlsx', 'csv'])
    
    if uploaded:
        if uploaded.name.endswith('.csv'): df_in = pd.read_csv(uploaded)
        else: df_in = pd.read_excel(uploaded)
        
        st.write(f"Đã nhận {len(df_in)} dòng.")
        col_target = st.selectbox("Chọn cột Tên thuốc:", df_in.columns)
        
        # --- NÚT 1: CHẠY MAP CƠ BẢN ---
        if st.button("🚀 BƯỚC 1: CHẠY MAPPING CƠ BẢN"):
            all_results = []
            bar = st.progress(0)
            
            for i, row in df_in.iterrows():
                inp = str(row[col_target])
                matches = search_product_v3(inp, st.session_state.db_vtma, st.session_state.brain, min_score, top_n)
                
                if matches:
                    for rank, m in enumerate(matches, 1):
                        current_score = m['Điểm Tổng']
                        status = "Khớp 100%" if current_score == 100 else ("Khớp cao" if rank == 1 else ("Trung bình" if current_score >= 70 else "Thấp"))
                        
                        all_results.append({
                            'Input_Goc': inp, 'Rank': rank, 'Trang_Thai': status,
                            'Ma_VTMA': m['Mã VTMA'], 'Ten_VTMA': m['Tên Thuốc (SKU)'],
                            'NSX_Chuan': m['NSX'], 'Ham_Luong_Chuan': m['Hàm Lượng'],
                            'Diem_Tong': m['Điểm Tổng'], 'AI_Suggestion': '' 
                        })
                else:
                    all_results.append({
                        'Input_Goc': inp, 'Rank': 1, 'Trang_Thai': 'Không tìm thấy',
                        'Ma_VTMA': '', 'Ten_VTMA': '', 'NSX_Chuan': '', 'Ham_Luong_Chuan': '',
                        'Diem_Tong': 0, 'AI_Suggestion': ''
                    })
                bar.progress((i+1)/len(df_in))
            
            st.session_state.result_df = pd.DataFrame(all_results)
            st.success("✅ Đã chạy xong Fuzzy Match cơ bản!")

        if 'result_df' in st.session_state:
            st.divider()
            st.subheader("🛠️ CÔNG CỤ AI NÂNG CAO")
            
            col_ai_1, col_ai_2 = st.columns(2)
            
            with col_ai_1:
                st.write("**Option 1: AI Rà Soát**")
                if st.button("🕵️ Kích hoạt AI Rà Soát"):
                    if not st.session_state.gemini.is_ready:
                        st.error("Thiếu API Key!")
                    else:
                        df_res = st.session_state.result_df
                        target_rows = df_res[df_res['Rank'] == 1]
                        my_bar = st.progress(0)
                        count = 0
                        for idx, row in target_rows.iterrows():
                            if row['Diem_Tong'] < 90:
                                candidates = get_candidates(row['Input_Goc'], st.session_state.db_vtma, limit=15)
                                ai_response = st.session_state.gemini.smart_match(row['Input_Goc'], candidates)
                                st.session_state.result_df.at[idx, 'AI_Suggestion'] = f"🤖 {ai_response}"
                            count += 1
                            my_bar.progress(count / len(target_rows))
                        st.success("Đã rà soát xong!")
                        st.session_state.result_df = st.session_state.result_df 

            with col_ai_2:
                st.write("**Option 2: Deep Search**")
                if st.button("🔍 Kích hoạt Deep Search"):
                    if not st.session_state.gemini.is_ready:
                        st.error("Thiếu API Key!")
                    else:
                        df_res = st.session_state.result_df
                        mask = (df_res['Diem_Tong'] < threshold_ai) | (df_res['Trang_Thai'] == 'Không tìm thấy')
                        hard_cases = df_res[mask]
                        if hard_cases.empty:
                            st.info("Không có ca nào.")
                        else:
                            my_bar = st.progress(0)
                            count = 0
                            for idx, row in hard_cases.iterrows():
                                candidates = get_candidates(row['Input_Goc'], st.session_state.db_vtma, limit=30)
                                ai_response = st.session_state.gemini.smart_match(row['Input_Goc'], candidates)
                                st.session_state.result_df.at[idx, 'AI_Suggestion'] = f"🔍 {ai_response}"
                                count += 1
                                my_bar.progress(count / len(hard_cases))
                            st.success(f"Đã xử lý {len(hard_cases)} ca khó!")
                            st.session_state.result_df = st.session_state.result_df 

            st.dataframe(st.session_state.result_df)
            df_final = st.session_state.result_df
            excel_name = "ket_qua_map_AI.xlsx"
            df_final.to_excel(excel_name, index=False)
            with open(excel_name, "rb") as f:
                st.download_button("📥 Tải Excel", f, excel_name)

with tab2:
    st.write("Phần Training AI (Giữ nguyên)...")


with tab2:
    st.write("Phần Training AI (Giữ nguyên)...")

with tab2:
    st.write("Phần Training AI (Giữ nguyên)...")
