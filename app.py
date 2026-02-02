import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process
import unidecode
import re
import os
import pickle
from collections import Counter

# =============================================================================
# 1. CẤU HÌNH TRANG & CLASS MACHINE LEARNING (TRÍ TUỆ NHÂN TẠO)
# =============================================================================
st.set_page_config(page_title="PharmaMaster: AI Matching", layout="wide", page_icon="💊")

class PharmaBrain:
    """
    Bộ não AI: Có khả năng học từ lịch sử mapping cũ để nhận diện Nhà Sản Xuất.
    """
    def __init__(self):
        self.brand_memory = {}  # Bộ nhớ: { 'từ_khóa': 'Tên_Hãng_Chuẩn' }
        self.learned_status = False

    def _tokenize(self, text):
        """Tách chuỗi thành các từ khóa nhỏ (tokens)"""
        if pd.isna(text): return []
        text = unidecode.unidecode(str(text).lower()) # Chuyển về tiếng Việt không dấu
        return re.findall(r"\w+", text) # Tách từ

    def learn(self, history_df, input_col, brand_col):
        """Học quy luật từ file Excel lịch sử"""
        brand_counter = {}
        count_learned = 0
        
        # Duyệt qua từng dòng lịch sử
        for _, row in history_df.iterrows():
            raw_text = row[input_col]
            true_brand = row[brand_col]
            
            if pd.isna(true_brand) or pd.isna(raw_text): continue
            
            tokens = self._tokenize(raw_text)
            for token in tokens:
                # Bỏ qua từ quá ngắn (<2 ký tự) hoặc toàn số
                if len(token) < 2 or token.isdigit(): continue
                
                if token not in brand_counter: brand_counter[token] = Counter()
                brand_counter[token][true_brand] += 1

        # LỌC NHIỄU: Chỉ nhớ quy luật nào có độ tin cậy > 70%
        self.brand_memory = {}
        for token, counts in brand_counter.items():
            most_common_brand, count = counts.most_common(1)[0]
            total = sum(counts.values())
            confidence = count / total
            
            # Quy tắc: Xuất hiện ít nhất 2 lần và độ chính xác > 70%
            if total >= 2 and confidence > 0.7: 
                self.brand_memory[token] = most_common_brand
                count_learned += 1
                
        self.learned_status = True
        return count_learned

    def predict_brand(self, raw_text):
        """Dự đoán hãng sản xuất dựa trên tên thuốc mới nhập"""
        if not self.brand_memory: return None
        
        tokens = self._tokenize(raw_text)
        detected_brands = []
        
        for token in tokens:
            if token in self.brand_memory:
                detected_brands.append(self.brand_memory[token])
        
        # Trả về hãng xuất hiện nhiều nhất trong câu
        if detected_brands:
            return Counter(detected_brands).most_common(1)[0][0]
        return None

    def save_model(self, path="pharma_brain.pkl"):
        """Lưu bộ nhớ ra file để dùng lần sau"""
        with open(path, "wb") as f: pickle.dump(self.brand_memory, f)

    def load_model(self, path="pharma_brain.pkl"):
        """Nạp bộ nhớ từ file đã lưu"""
        if os.path.exists(path):
            with open(path, "rb") as f: self.brand_memory = pickle.load(f)
            self.learned_status = True
            return True
        return False

# =============================================================================
# 2. CÁC HÀM XỬ LÝ DỮ LIỆU & TÍNH ĐIỂM (CORE LOGIC)
# =============================================================================

def normalize_text(text):
    """Chuẩn hóa text: Chữ thường, bỏ dấu, cắt khoảng trắng"""
    if pd.isna(text): return ""
    return unidecode.unidecode(str(text).lower()).strip()

def extract_numbers(text):
    """Lấy tập hợp các con số từ chuỗi (VD: '500mg' -> {500})"""
    if pd.isna(text): return set()
    # Regex lấy số thực (integer hoặc float)
    return set(re.findall(r"\d+\.?\d*", str(text)))

@st.cache_data
def load_master_data():
    """Phiên bản Debug: Tự động dò dấu phân cách và in tên cột ra màn hình"""
    file_path = "data/vtma_standard.csv"
    
    if not os.path.exists(file_path):
        st.error(f"❌ Không tìm thấy file '{file_path}'")
        return None

    try:
        # 1. Đọc file thông minh: Tự động nhận diện dấu phẩy (,) hay Tab (\t)
        # engine='python' giúp tự dò separator
        df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8')
    except:
        try:
            # Nếu lỗi encoding, thử lại với latin1
            df = pd.read_csv(file_path, sep=None, engine='python', encoding='latin1')
        except Exception as e:
            st.error(f"❌ Không đọc được file. Lỗi: {e}")
            return None

    # 2. Chuẩn hóa tên cột: Xóa khoảng trắng thừa, về chữ thường
    # Ví dụ: " VTMA Code " -> "vtma code"
    df.columns = df.columns.str.strip().str.lower()
    
    # -----------------------------------------------------------
    # 🔍 DEBUG: IN RA TÊN CỘT THỰC TẾ ĐỂ BẠN KIỂM TRA
    # -----------------------------------------------------------
    # Nếu code chạy ok thì dòng này sẽ ẩn đi, nếu lỗi nó giúp bạn biết file có gì
    # st.write("🔍 Debug - Các cột tìm thấy trong file:", df.columns.tolist())
    
    # 3. MAPPING LINH HOẠT HƠN
    # Tạo danh sách các tên cột thường gặp để map về chuẩn
    mapping_dict = {
        'ma_vtma': ['vtma code', 'ma thuoc', 'ma_vtma', 'vtma_code', 'code'],
        'ten_thuoc': ['product', 'ten thuoc', 'ten_thuoc', 'name', 'ten'],
        'hoat_chat': ['molecule', 'hoat chat', 'hoat_chat', 'active ingredient'],
        'ten_cong_ty': ['manufacturer', 'corporation', 'ten cong ty', 'nha san xuat', 'hang sx'],
        'ham_luong': ['galenic', 'ham luong', 'nong do', 'strength'],
        'dang_bao_che': ['unit_measure', 'dang bao che', 'dosage form', 'form'],
        'sku_full': ['sku', 'sku name', 'ten day du']
    }

    # Thực hiện đổi tên dựa trên từ điển trên
    final_rename_map = {}
    current_cols = df.columns.tolist()
    
    for standard_col, aliases in mapping_dict.items():
        found = False
        for alias in aliases:
            if alias in current_cols:
                final_rename_map[alias] = standard_col
                found = True
                break # Đã tìm thấy thì dừng, sang cột tiếp theo
    
    if final_rename_map:
        df.rename(columns=final_rename_map, inplace=True)

    # 4. KIỂM TRA LẠI SAU KHI MAP
    required_cols = ['ma_vtma', 'ten_thuoc', 'ten_cong_ty', 'hoat_chat', 'ham_luong', 'dang_bao_che']
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        st.error("⚠️ LỖI CẤU TRÚC FILE CSV")
        st.error(f"Phần mềm cần cột: **{required_cols}**")
        st.warning(f"Nhưng trong file của bạn sau khi đọc chỉ có: {df.columns.tolist()}")
        st.info("💡 Gợi ý: Hãy mở file CSV bằng Excel, sửa dòng đầu tiên (Header) thành: vtma code, product, manufacturer, molecule, galenic, unit_measure")
        st.stop()

    # 5. Xử lý dữ liệu text (Logic cũ)
    df['norm_name'] = df['ten_thuoc'].apply(normalize_text)
    df['norm_brand'] = df['ten_cong_ty'].apply(normalize_text)
    df['norm_active'] = df['hoat_chat'].apply(normalize_text)
    df['norm_form'] = df['dang_bao_che'].apply(normalize_text)
    df['norm_strength'] = df['ham_luong'].apply(normalize_text)

    # Cột hiển thị
    if 'sku_full' in df.columns:
        df['display_name'] = df['sku_full']
    else:
        df['display_name'] = df['ten_thuoc'] + " " + df['ham_luong']

    return df
def calculate_weighted_score(input_str, row_data, ml_predicted_brand=None):
    """
    Tính điểm khớp (0-100) dựa trên 5 tiêu chí + Điểm thưởng AI
    """
    norm_input = normalize_text(input_str)
    
    # 1. Tên thuốc (40%)
    score_name = fuzz.token_set_ratio(norm_input, row_data['norm_name'])
    
    # 2. Thương hiệu/Hãng (20%)
    score_brand = fuzz.partial_ratio(row_data['norm_brand'], norm_input)
    
    # 3. Hoạt chất (20%)
    score_active = fuzz.token_set_ratio(row_data['norm_active'], norm_input)
    
    # 4. Hàm lượng (10%) - Logic đặc biệt: Khớp số
    input_nums = extract_numbers(input_str)
    row_nums = extract_numbers(row_data['ham_luong'])
    if not row_nums: score_strength = 50 # Không có số liệu thì cho điểm trung bình
    elif input_nums.intersection(row_nums): score_strength = 100 # Có số trùng nhau
    else: score_strength = 0 # Có số nhưng khác nhau (lệch hàm lượng)
    
    # 5. Dạng bào chế (10%)
    score_form = fuzz.partial_ratio(row_data['norm_form'], norm_input)
    
    # --- TỔNG ĐIỂM CƠ BẢN ---
    base_score = (score_name*0.4) + (score_brand*0.2) + (score_active*0.2) + (score_strength*0.1) + (score_form*0.1)
    
    # --- 6. ĐIỂM THƯỞNG AI (TRUST BONUS) ---
    ml_bonus = 0
    match_ml = "No"
    
    if ml_predicted_brand:
        # Nếu AI đoán ra hãng, và hãng đó khớp với dữ liệu dòng này (>85%)
        similarity = fuzz.token_set_ratio(normalize_text(ml_predicted_brand), row_data['norm_brand'])
        if similarity > 85:
            ml_bonus = 15 # Cộng 15 điểm
            match_ml = "Yes"
            
    final_score = base_score + ml_bonus
    
    return {
        'total': min(final_score, 100), # Max là 100
        'detail': f"Tên:{int(score_name)} | Hãng:{int(score_brand)} | Số:{int(score_strength)} | ML:+{ml_bonus}",
        'ml_match': match_ml
    }

def search_product(input_text, db_df, brain_model, min_score=50, top_n=1):
    """Hàm tìm kiếm chính"""
    # B1: AI dự đoán hãng
    predicted_brand = brain_model.predict_brand(input_text)
    
    # B2: Lọc thô (Heuristic) - Lấy Top 50 tên giống nhất để tính toán cho nhanh
    norm_input = normalize_text(input_text)
    candidates = process.extract(norm_input, db_df['norm_name'], limit=50, scorer=fuzz.token_set_ratio)
    candidate_indices = [x[2] for x in candidates if x[1] > 30] # Chỉ lấy nếu giống > 30%
    
    if not candidate_indices: return []

    subset = db_df.iloc[candidate_indices].copy()
    results = []
    
    # B3: Chấm điểm chi tiết
    for idx, row in subset.iterrows():
        scoring = calculate_weighted_score(input_text, row, ml_predicted_brand=predicted_brand)
        
        if scoring['total'] >= min_score:
            results.append({
                'Mã VTMA': row['ma_vtma'],
                'Tên Thuốc (SKU)': row['display_name'],
                'NSX Chuẩn': row['ten_cong_ty'],
                'AI Dự Đoán': predicted_brand if predicted_brand else "-",
                'Điểm': round(scoring['total'], 1),
                'Chi Tiết': scoring['detail']
            })
            
    # B4: Sắp xếp & Cắt Top N
    results.sort(key=lambda x: x['Điểm'], reverse=True)
    return results[:top_n]

# =============================================================================
# 3. GIAO DIỆN NGƯỜI DÙNG (STREAMLIT UI)
# =============================================================================

# --- A. KHỞI TẠO STATE ---
if 'brain' not in st.session_state:
    st.session_state.brain = PharmaBrain()
    st.session_state.brain.load_model() # Load bộ não cũ nếu có

if 'db_vtma' not in st.session_state:
    with st.spinner("⏳ Đang tải Master Data..."):
        df_loaded = load_master_data()
        if df_loaded is not None:
            st.session_state.db_vtma = df_loaded
        else:
            st.stop() # Dừng App nếu không load được Data

# --- B. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu hình Mapping")
    min_score = st.slider("Ngưỡng điểm tối thiểu", 0, 100, 60, help="Dưới điểm này coi như không tìm thấy")
    top_n = st.number_input("Số lượng kết quả (Top N)", 1, 10, 1)
    st.divider()
    st.write(f"📊 Database: **{len(st.session_state.db_vtma)}** SKU")
    st.write(f"🧠 Trạng thái AI: **{'Đã học' if st.session_state.brain.learned_status else 'Chưa học'}**")

st.title("💊 PharmaMaster: Hệ Thống Mapping Thuốc Thông Minh")

# --- C. MAIN TABS ---
tab1, tab2 = st.tabs(["🚀 Chạy Mapping (Run)", "🧠 Dạy AI (Train)"])

# TAB 1: CHẠY MAPPING
with tab1:
    st.subheader("1. Mapping Dữ Liệu Mới")
    
    # Input Text (Test nhanh)
    col_search, col_btn = st.columns([3, 1])
    with col_search:
        test_txt = st.text_input("Nhập tên thuốc để test nhanh:", placeholder="Ví dụ: Hapacol 650 dhg")
    
    if test_txt:
        res = search_product(test_txt, st.session_state.db_vtma, st.session_state.brain, min_score, top_n)
        if res:
            df_res = pd.DataFrame(res)
            # Highlight dòng được cộng điểm AI
            st.dataframe(df_res.style.apply(lambda x: ['background-color: #d1e7dd' if "ML:+" in str(x['Chi Tiết']) else '' for i in x], axis=1), use_container_width=True)
        else:
            st.warning("Không tìm thấy kết quả phù hợp.")

    st.divider()
    
    # Upload File (Chạy hàng loạt)
    st.write("📂 **Upload file Excel danh sách cần Map:**")
    uploaded_file = st.file_uploader("Chọn file (.xlsx, .csv)", type=['xlsx', 'csv'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'): df_in = pd.read_csv(uploaded_file)
            else: df_in = pd.read_excel(uploaded_file)
            
            st.write(f"Đã nhận {len(df_in)} dòng dữ liệu.")
            col_target = st.selectbox("Chọn cột chứa tên thuốc:", df_in.columns)
            
            if st.button("🚀 BẮT ĐẦU MAPPING HÀNG LOẠT"):
                results_batch = []
                progress_bar = st.progress(0)
                
                for i, row in df_in.iterrows():
                    input_val = str(row[col_target])
                    matches = search_product(input_val, st.session_state.db_vtma, st.session_state.brain, min_score, 1)
                    
                    if matches:
                        match = matches[0] # Lấy Top 1
                        results_batch.append({
                            'Input_Goc': input_val,
                            'Ma_VTMA': match['Mã VTMA'],
                            'Ten_VTMA': match['Tên Thuốc (SKU)'],
                            'Diem': match['Điểm'],
                            'Ghi_Chu': match['Chi Tiết']
                        })
                    else:
                        results_batch.append({'Input_Goc': input_val, 'Ma_VTMA': '', 'Ten_VTMA': 'Không tìm thấy', 'Diem': 0})
                    
                    progress_bar.progress((i + 1) / len(df_in))
                
                df_out = pd.DataFrame(results_batch)
                st.success("✅ Hoàn tất!")
                st.dataframe(df_out)
                
                # Download
                csv = df_out.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Tải kết quả (CSV)", csv, "ket_qua_map.csv", "text/csv")
                
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")

# TAB 2: DẠY MÁY HỌC
with tab2:
    st.subheader("2. Huấn luyện AI (Supervised Learning)")
    st.info("Upload file lịch sử đã map đúng để máy học cách nhận diện Nhà Sản Xuất từ tên viết tắt.")
    
    uploaded_hist = st.file_uploader("Chọn file lịch sử (.xlsx)", key="hist")
    
    if uploaded_hist:
        df_hist = pd.read_excel(uploaded_hist)
        st.dataframe(df_hist.head(3))
        
        c1, c2 = st.columns(2)
        col_in = c1.selectbox("Cột Tên Gốc (Input)", df_hist.columns)
        col_out = c2.selectbox("Cột Hãng Chuẩn (Target)", df_hist.columns)
        
        if st.button("🎓 BẮT ĐẦU DẠY MÁY"):
            with st.spinner("Đang phân tích dữ liệu..."):
                n_learned = st.session_state.brain.learn(df_hist, col_in, col_out)
                st.session_state.brain.save_model()
            
            st.success(f"🎉 Đã học xong! Máy đã ghi nhớ {n_learned} quy luật nhận diện hãng mới.")
            st.json(st.session_state.brain.brand_memory)
