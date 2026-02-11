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
import random
import io

# =============================================================================
# 1. CẤU HÌNH TRANG & CSS
# =============================================================================
st.set_page_config(page_title="PharmaMaster Ultimate", layout="wide", page_icon="🧬")

# =============================================================================
# 2. CLASS GEMINI AI (VỚI CƠ CHẾ RETRY MẠNH MẼ TỪ FILE 02)
# =============================================================================
class GeminiAgent:
    def __init__(self, api_key, model_name):
        self.is_ready = False
        self.current_model = "None"
        
        if api_key and model_name:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model_name)
                self.current_model = model_name
                self.is_ready = True
            except Exception as e:
                self.is_ready = False
                self.error = str(e)
        else:
            self.is_ready = False

    def smart_match(self, input_drug, candidates_df):
        """
        Gửi yêu cầu với cơ chế Retry (Thử lại) khi gặp lỗi 429
        """
        if not self.is_ready: return "⚠️ Lỗi: Chưa chọn Model hoặc API Key sai"

        candidates_str = ""
        for idx, row in candidates_df.iterrows():
            candidates_str += f"- ID: {row['ma_vtma']} | Tên: {row['ten_thuoc']} | HL: {row['ham_luong']} | NSX: {row['nsx_full']}\n"

        prompt = f"""
        Bạn là Dược sĩ. Tìm mã thuốc chuẩn (ID) cho sản phẩm đầu vào.
        INPUT: "{input_drug}"
        DATABASE:
        {candidates_str}
        YÊU CẦU: Chọn 1 ID khớp nhất.
        TRẢ LỜI 1 DÒNG DUY NHẤT: ID_CHON | ĐỘ_TIN_CẬY (Thấp/Trung bình/Cao) | LÝ DO NGẮN GỌN
        Ví dụ: VTMA_001 | Cao | Khớp tên và hãng
        Nếu không khớp >70%, trả về: "NONE | - | Không tìm thấy"
        """
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    wait_time = (attempt + 1) * 3 + random.uniform(0, 2)
                    time.sleep(wait_time)
                    continue
                else:
                    return f"AI Error: {error_str}"
        
        return "⚠️ AI Busy (Hết hạn mức, vui lòng chờ 1 phút)"

# =============================================================================
# 3. CLASS MACHINE LEARNING (PHARMA BRAIN - GIỮ NGUYÊN)
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
# 4. XỬ LÝ DỮ LIỆU TỐI ƯU (MERGE LOGIC)
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
        try: df = pd.read_csv(file_path, sep=None, engine='python', encoding='latin1')
        except Exception as e:
            st.error(f"❌ Lỗi đọc file: {e}")
            return None

    df.columns = df.columns.str.strip().str.lower().str.replace('\ufeff', '').str.replace('ï»¿', '')
    
    # 1. Mapping cột thông minh (Từ File 01)
    mapping_dict = {
        'ma_vtma': ['ma_thuoc', 'vtma code', 'code'],
        'ten_thuoc': ['ten_thuoc', 'product', 'name'],
        'hoat_chat': ['hoat_chat', 'molecule'],
        'ten_cong_ty': ['ten_cong_ty', 'manufacturer', 'ten_tap_doan'],
        'corporation': ['corporation', 'tap_doan'],
        'ham_luong': ['ham_luong', 'galenic', 'nong do'],
        'dang_bao_che': ['dang_bao_che', 'unit_measure'],
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

    # 2. Xử lý gộp cột NSX (Từ File 01 - Quan trọng cho lọc)
    if 'corporation' in df.columns:
        df['nsx_full'] = df['ten_cong_ty'] + " (" + df['corporation'].fillna('') + ")"
    else:
        df['nsx_full'] = df['ten_cong_ty']
    df['nsx_full'] = df['nsx_full'].str.replace(r'\(\s*\)', '', regex=True).str.strip()

    # 3. Tính toán trước các cột chuẩn hóa (Từ File 02 - Tối ưu tốc độ)
    df['norm_name'] = df['ten_thuoc'].apply(normalize_text)
    df['norm_brand'] = df['ten_cong_ty'].apply(normalize_text) # Vẫn giữ norm_brand gốc để so sánh tên
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
# 5. CORE ENGINE (KẾT HỢP LOGIC)
# =============================================================================
def calculate_detailed_score(input_str, row_data, ml_predicted_brand=None):
    norm_input = normalize_text(input_str)
    
    # Dùng cột đã chuẩn hóa sẵn (tối ưu từ File 02)
    score_name = fuzz.token_set_ratio(norm_input, row_data['norm_name'])
    score_brand = fuzz.partial_ratio(row_data['norm_brand'], norm_input)
    
    score_active = 0
    if row_data['norm_active']: score_active = fuzz.token_set_ratio(row_data['norm_active'], norm_input)
    else: score_active = 50 

    # Logic số học (Từ cả 2 file)
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

def get_candidates(input_text, db_df, limit=20, filtered_nsx=None):
    # Logic lọc NSX (Từ File 01)
    working_df = db_df
    if filtered_nsx:
        # Lọc dataframe trước khi fuzzy search -> Tăng tốc & Chính xác cực cao
        working_df = db_df[db_df['nsx_full'].isin(filtered_nsx)]
    
    if working_df.empty: return pd.DataFrame()

    norm_input = normalize_text(input_text)
    # Search trên tập đã lọc
    candidates = process.extract(norm_input, working_df['search_index'], limit=limit, scorer=fuzz.token_set_ratio)
    indices = [x[2] for x in candidates]
    return working_df.iloc[indices].copy()

def search_product_v3(input_text, db_df, brain_model, min_score=50, top_n=3, filtered_nsx=None):
    predicted_brand = brain_model.predict_brand(input_text)
    subset = get_candidates(input_text, db_df, limit=50, filtered_nsx=filtered_nsx)
    
    if subset.empty: return []

    results = []
    for idx, row in subset.iterrows():
        scores = calculate_detailed_score(input_text, row, ml_predicted_brand=predicted_brand)
        if scores['total'] >= min_score:
            results.append({
                'Mã VTMA': row['ma_vtma'],
                'Tên Thuốc (SKU)': row['display_name'],
                'NSX': row['nsx_full'],
                'Hàm Lượng': row['ham_luong'], 
                'Điểm Tổng': round(scores['total'], 1),
                'Điểm Tên (40%)': int(scores['s_name']),
                'Điểm Hãng (20%)': int(scores['s_brand']),
                'Điểm HoạtChất (20%)': int(scores['s_active']),
                'Điểm HàmLượng (10%)': int(scores['s_strength']),
                'AI Bonus': scores['ml_bonus']
            })
            
    results.sort(key=lambda x: x['Điểm Tổng'], reverse=True)
    return results[:top_n]

# =============================================================================
# 6. GIAO DIỆN STREAMLIT (MERGE WORKFLOW)
# =============================================================================

if 'brain' not in st.session_state:
    st.session_state.brain = PharmaBrain()
    st.session_state.brain.load_model()

if 'db_vtma' not in st.session_state:
    with st.spinner("⏳ Đang tải dữ liệu chuẩn..."):
        df_loaded = load_master_data()
        if df_loaded is not None: st.session_state.db_vtma = df_loaded
        else: st.stop()

# Khởi tạo session
if 'confirmed_nsx' not in st.session_state: st.session_state.confirmed_nsx = []
if 'brand_step_skipped' not in st.session_state: st.session_state.brand_step_skipped = False
if 'brand_suggestions' not in st.session_state: st.session_state.brand_suggestions = []

# --- SIDEBAR TỪ FILE 02 (CLEANER) ---
with st.sidebar:
    st.header("🤖 Cấu hình Gemini AI")
    api_key = st.text_input("Nhập Google API Key", type="password")
    valid_models = []
    if api_key:
        try:
            genai.configure(api_key=api_key)
            all_models = genai.list_models()
            for m in all_models:
                if 'generateContent' in m.supported_generation_methods:
                     valid_models.append(m.name.replace("models/", ""))
        except: st.error("API Key lỗi!")

    if valid_models:
        default_ix = valid_models.index('gemini-1.5-flash') if 'gemini-1.5-flash' in valid_models else 0
        selected_model = st.selectbox("Chọn Model AI:", valid_models, index=default_ix)
        st.session_state.gemini = GeminiAgent(api_key, selected_model)
        st.success("✅ AI Sẵn sàng")
    else:
        st.info("Nhập API Key để dùng tính năng AI sửa lỗi.")
        st.session_state.gemini = GeminiAgent(None, None)

    st.divider()
    st.header("⚙️ Cấu hình Map")
    min_score = st.slider("Min Score (%)", 0, 100, 60) 
    top_n = st.number_input("Top N", 1, 10, 3)
    threshold_ai = st.number_input("Ngưỡng kích hoạt Deep Search", 0, 100, 70)

st.title("🧬 PharmaMaster Ultimate: Intelligent Mapping")

# --- TAB WORKFLOW: KẾT HỢP 3 BƯỚC ---
tab1, tab_brand, tab3, tab4 = st.tabs(["1️⃣ Upload & Test", "2️⃣ Chọn Bộ Lọc (NSX)", "3️⃣ Chạy Full & Fix Lỗi", "4️⃣ Training Model"])

# --- TAB 1: UPLOAD & TEST SAMPLE (TỪ FILE 01) ---
with tab1:
    st.subheader("Bước 1: Tải dữ liệu & Phân tích mẫu")
    uploaded = st.file_uploader("Upload file Excel/CSV cần map", type=['xlsx', 'csv'])
    
    if uploaded:
        if uploaded.name.endswith('.csv'): df_in = pd.read_csv(uploaded)
        else: df_in = pd.read_excel(uploaded)
        
        st.session_state.df_input = df_in # Lưu vào session để dùng cho Tab 3
        st.write(f"Đã nhận {len(df_in)} dòng dữ liệu.")
        col_target = st.selectbox("Chọn cột Tên thuốc:", df_in.columns, key="col_target")
        st.session_state.col_target = col_target

        if st.button("🧪 CHẠY THỬ 3 MẪU & GỢI Ý NSX"):
            sample_3 = df_in.head(3)
            temp_results = []
            for i, row in sample_3.iterrows():
                inp = str(row[col_target])
                # Chạy không lọc để tìm NSX tiềm năng
                matches = search_product_v3(inp, st.session_state.db_vtma, st.session_state.brain, 30, 1)
                if matches:
                    temp_results.append({'Input': inp, 'NSX_Gợi_Ý': matches[0]['NSX'], 'Mã': matches[0]['Mã VTMA']})
            
            st.session_state.brand_suggestions = temp_results
            st.success("✅ Đã xong! Hãy chuyển sang Tab 'Chọn Bộ Lọc' để xác nhận các NSX này.")
            st.table(temp_results)

# --- TAB 2: BRAND FILTER (TỪ FILE 01 - TÍNH NĂNG "SÁT THỦ") ---
with tab_brand:
    st.subheader("Bước 2: Xác nhận Nhà Sản Xuất (Bộ lọc)")
    st.info("💡 Việc lọc đúng NSX sẽ giúp loại bỏ 90% kết quả sai và tăng tốc độ xử lý.")

    # 1. Hiển thị gợi ý từ bước 1
    if st.session_state.brand_suggestions:
        suggestions = list(set([item['NSX_Gợi_Ý'] for item in st.session_state.brand_suggestions]))
        st.write("Gợi ý từ dữ liệu mẫu:")
        for nsx in suggestions:
            c1, c2 = st.columns([4, 1])
            c1.info(f"🏭 {nsx}")
            if c2.button("Thêm", key=f"add_{nsx}"):
                if nsx not in st.session_state.confirmed_nsx:
                    st.session_state.confirmed_nsx.append(nsx)
                    st.rerun()

    st.divider()
    
    # 2. Chọn thủ công
    all_vtma_nsx = sorted(st.session_state.db_vtma['nsx_full'].unique().tolist())
    selected_manual = st.selectbox("Tìm & Thêm thủ công:", ["--- Chọn nhà máy ---"] + all_vtma_nsx)
    if st.button("➕ Thêm vào danh sách"):
        if selected_manual != "--- Chọn nhà máy ---" and selected_manual not in st.session_state.confirmed_nsx:
            st.session_state.confirmed_nsx.append(selected_manual)
            st.rerun()

    # 3. Danh sách đã chọn
    st.write("### 📋 Danh sách áp dụng:")
    if st.session_state.confirmed_nsx:
        for nsx in st.session_state.confirmed_nsx:
            st.success(f"✅ {nsx}")
        if st.button("🗑️ Xóa tất cả bộ lọc"):
            st.session_state.confirmed_nsx = []
            st.session_state.brand_step_skipped = False
            st.rerun()
    else:
        st.warning("Chưa có bộ lọc nào.")

    if st.checkbox("⏩ Bỏ qua bước này (Tìm trên toàn bộ Database)", value=st.session_state.brand_step_skipped):
        st.session_state.brand_step_skipped = True
    else:
        st.session_state.brand_step_skipped = False

# --- TAB 3: FULL RUN & AI FIX (KẾT HỢP FILE 01 & 02) ---
with tab3:
    st.subheader("Bước 3: Chạy Mapping & AI Hậu Kiểm")
    
    if 'df_input' not in st.session_state:
        st.error("Vui lòng upload file ở Tab 1 trước.")
    else:
        # Nút chạy chính
        if st.button("🚀 CHẠY FULL MAPPING"):
            filter_list = st.session_state.confirmed_nsx if not st.session_state.brand_step_skipped else None
            
            all_results = []
            bar = st.progress(0)
            df_run = st.session_state.df_input
            col_t = st.session_state.col_target

            for i, row in df_run.iterrows():
                inp = str(row[col_t])
                # Gọi hàm search với filter_list
                matches = search_product_v3(inp, st.session_state.db_vtma, st.session_state.brain, min_score, top_n, filtered_nsx=filter_list)
                
                if matches:
                    for rank, m in enumerate(matches, 1):
                        all_results.append({
                            'Input_Goc': inp, 'Rank': rank, 'Trang_Thai': 'Khớp',
                            'Ma_VTMA': m['Mã VTMA'], 'Ten_VTMA': m['Tên Thuốc (SKU)'],
                            'NSX_Chuan': m['NSX'],'Ham_Luong_Chuan': m['Hàm Lượng'],
                            'Diem_Tong': m['Điểm Tổng'], 'AI_Suggestion': '' 
                        })
                else:
                    # Trường hợp không tìm thấy (Not Found)
                    all_results.append({
                        'Input_Goc': inp, 'Rank': 1, 'Trang_Thai': 'Không tìm thấy',
                        'Ma_VTMA': '', 'Ten_VTMA': '', 'NSX_Chuan': '', 'Ham_Luong_Chuan': '',
                        'Diem_Tong': 0, 'AI_Suggestion': ''
                    })
                
                # Cập nhật thanh tiến trình
                bar.progress((i+1)/len(df_run))
            
            # Lưu kết quả vào Session State
            st.session_state.result_df = pd.DataFrame(all_results)
            st.success("✅ Đã chạy xong Fuzzy Match cơ bản!")

    # --- KHU VỰC 2: AI DEEP SEARCH & DOWNLOAD (TỪ FILE 02) ---
    if 'result_df' in st.session_state:
        st.divider()
        st.subheader("🛠️ Công cụ: AI Rà Soát & Deep Search")
        
        col_ai_1, col_ai_2 = st.columns([2, 1])
        
        with col_ai_1:
            st.info(f"AI sẽ tự động kiểm tra các dòng có Điểm < {threshold_ai} hoặc 'Không tìm thấy'.")
            
            if st.button("🕵️ Kích hoạt AI Rà Soát (Deep Search)"):
                if not st.session_state.gemini.is_ready:
                    st.error("❌ Thiếu API Key! Vui lòng nhập Key ở cột bên trái.")
                else:
                    df_res = st.session_state.result_df
                    # Lọc ra các ca khó cần AI xử lý
                    mask = (df_res['Diem_Tong'] < threshold_ai) | (df_res['Trang_Thai'] == 'Không tìm thấy')
                    # Chỉ lấy Rank 1 để check (tránh check trùng lặp các rank sau)
                    hard_cases = df_res[mask & (df_res['Rank'] == 1)]
                    
                    if hard_cases.empty:
                        st.success("🎉 Dữ liệu quá tốt! Không có dòng nào dưới ngưỡng điểm cần AI sửa.")
                    else:
                        st.write(f"Đang xử lý {len(hard_cases)} trường hợp nghi ngờ...")
                        my_bar = st.progress(0)
                        count = 0
                        
                        # Sử dụng filter hiện tại nếu có
                        current_filter = st.session_state.confirmed_nsx if not st.session_state.brand_step_skipped else None

                        for idx, row in hard_cases.iterrows():
                            # Lấy candidates rộng hơn (limit=20) để AI có nhiều lựa chọn
                            candidates = get_candidates(row['Input_Goc'], st.session_state.db_vtma, limit=20, filtered_nsx=current_filter)
                            
                            # Gọi Gemini Agent (đã có Retry logic)
                            ai_response = st.session_state.gemini.smart_match(row['Input_Goc'], candidates)
                            
                            # Ghi kết quả vào cột AI Suggestion
                            st.session_state.result_df.at[idx, 'AI_Suggestion'] = f"🤖 {ai_response}"
                            
                            # Delay nhẹ để tránh lỗi 429 nếu chạy quá nhanh
                            time.sleep(1.5)
                            
                            count += 1
                            my_bar.progress(count / len(hard_cases))
                        
                        st.success(f"✅ Đã rà soát xong {len(hard_cases)} dòng!")
                        st.rerun() # Load lại trang để hiển thị kết quả mới

        with col_ai_2:
            st.write("### 📥 Xuất Kết Quả")
            # Hiển thị dataframe kết quả
            st.dataframe(st.session_state.result_df, height=300)
            
            # Logic xuất Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                st.session_state.result_df.to_excel(writer, index=False, sheet_name='KetQua')
            
            st.download_button(
                label="Tải file Excel (.xlsx)",
                data=buffer,
                file_name="Pharma_Map_Result_AI.xlsx",
                mime="application/vnd.ms-excel"
            )

# --- TAB 4: TRAINING MODEL (TỪ FILE 01 - GIÚP MÁY KHÔN HƠN) ---
with tab4:
    st.subheader("4️⃣ Huấn luyện AI (Supervised Learning)")
    st.info("Nếu máy nhận diện sai hãng (ví dụ: 'DHG' không ra 'Dược Hậu Giang'), hãy upload file lịch sử đã map đúng để dạy lại máy.")
    
    uploaded_hist = st.file_uploader("Chọn file lịch sử mapping (.xlsx)", key="hist")
    
    if uploaded_hist:
        df_hist = pd.read_excel(uploaded_hist)
        st.write("Dữ liệu mẫu:")
        st.dataframe(df_hist.head(3))
        
        c1, c2 = st.columns(2)
        col_in = c1.selectbox("Cột Tên Gốc (Input) - Ví dụ: Ten_Thuoc", df_hist.columns)
        col_out = c2.selectbox("Cột Hãng Chuẩn (Target) - Ví dụ: NSX_Chuan", df_hist.columns)
        
        if st.button("🎓 BẮT ĐẦU DẠY MÁY"):
            with st.spinner("Đang phân tích quy luật từ ngữ..."):
                # Gọi hàm learn từ Class PharmaBrain
                n_learned = st.session_state.brain.learn(df_hist, col_in, col_out)
                st.session_state.brain.save_model()
            
            st.success(f"🎉 Đã học xong! Máy đã ghi nhớ thêm {n_learned} từ khóa nhận diện hãng mới.")
            
            with st.expander("Xem bộ nhớ (Brand Memory)"):
                st.json(st.session_state.brain.brand_memory)
