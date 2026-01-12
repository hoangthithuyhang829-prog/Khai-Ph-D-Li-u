__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ================== 1. CẤU HÌNH ĐƯỜNG DẪN & API ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# Kiểm tra API KEY
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ Chưa cấu hình GOOGLE_API_KEY trong Streamlit Secrets")
    st.stop()

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ================== 2. KHỞI TẠO EMBEDDING & COLLECTION ==================
# Lưu ý: Phải dùng ĐÚNG model mà bạn đã dùng ở máy Local (Colab)
# Theo ảnh bạn gửi là BAAI/bge-m3
@st.cache_resource
def load_collection():
    # 1. Ép sử dụng đường dẫn tuyệt đối
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "chroma_db")
    
    chroma_client = chromadb.PersistentClient(path=db_path)

    # 2. Phải dùng ĐÚNG model embedding đã dùng lúc tạo database
    # Trong ảnh bạn gửi là BAAI/bge-m3, hãy dùng nó
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3"
    )

    # 3. Sử dụng get_collection (không dùng get_or_create) để kiểm tra
    # Phải khớp tên "dichvucong_rag"
    collection = chroma_client.get_collection(
        name="dichvucong_rag", 
        embedding_function=embedding_func
    )

    return collection

# ================== 3. HÀM XỬ LÝ TRUY VẤN (RAG) ==================
def query_rag(query: str, top_k: int):
    if not collection:
        return "Database chưa được tải thành công."

    # Truy vấn dữ liệu
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Kiểm tra nếu không có kết quả phù hợp
    if not results["documents"] or len(results["documents"][0]) == 0:
        return "Xin lỗi! Câu hỏi của bạn không nằm trong phạm vi hỗ trợ của tôi."

    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        hierarchy = meta.get('hierarchy', 'Thông tin')
        url = meta.get('url', 'Không có nguồn')
        context_parts.append(f"[{hierarchy}]\n{doc}\n(Nguồn: {url})")

    context = "\n\n".join(context_parts)

    prompt = f"""
Bạn là trợ lý tư vấn thủ tục hành chính công của Việt Nam.
Bạn chỉ trả lời câu hỏi.
KHÔNG được viết lại, diễn đạt lại hay sửa đổi câu hỏi của người dùng.
KHÔNG nhắc lại câu hỏi.
PHẠM VI ÁP DỤNG:
- Ưu tiên tư vấn các thủ tục hành chính liên quan đến trẻ em dưới 6 tuổi.
- Nếu CONTEXT không đề cập rõ độ tuổi nhưng nội dung thuộc thủ tục thường áp dụng cho trẻ em,
  bạn được phép trả lời dựa trên thông tin hiện có và nêu rõ phạm vi áp dụng nếu được đề cập.

NGUYÊN TẮC TRẢ LỜI:
- Chỉ sử dụng thông tin có trong CONTEXT bên dưới.
- Không sử dụng kiến thức bên ngoài.
- Không tự bổ sung thông tin không có trong CONTEXT.
- Không tự thay đổi câu hỏi của người dùng.

CÁCH TRẢ LỜI:
- Chỉ trả lời các nội dung LIÊN QUAN TRỰC TIẾP đến câu hỏi.
- Có thể tổng hợp nhiều đoạn trong CONTEXT nếu chúng cùng mô tả một thủ tục.
- Trình bày ngắn gọn, rõ ràng, đúng trọng tâm.

TRƯỜNG HỢP KHÔNG TRẢ LỜI:
Chỉ trả lời đúng câu sau nếu:
- CONTEXT hoàn toàn không chứa thông tin liên quan đến câu hỏi.

Câu trả lời trong trường hợp này PHẢI CHÍNH XÁC:
"Xin lỗi! Câu hỏi của bạn không nằm trong phạm vi hỗ trợ của tôi."

YÊU CẦU ĐỊNH DẠNG:
- Trả lời bằng tiếng Việt.
- Nếu có nhiều ý, trình bày bằng gạch đầu dòng hoặc đánh số.
- Giữ nguyên trích dẫn nguồn nếu có trong CONTEXT.

    Context:
    {context}

    Câu hỏi: {query}

    Trả lời bằng tiếng Việt, có đánh số nếu là danh sách, và trích dẫn nguồn rõ ràng (tên block, URL):
    """
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# ================== 4. GIAO DIỆN STREAMLIT ==================
st.set_page_config(page_title="Chatbot TTHC Trẻ Em", page_icon="🤖")

# Hiệu ứng hoa rơi (Giữ nguyên CSS của bạn)
st.markdown("""
<style>
.stApp { background: #fff0f5; }
.flower { position: fixed; top: -40px; font-size: 22px; animation: fall 8s linear infinite; z-index: 0; }
@keyframes fall { to { transform: translateY(110vh) rotate(360deg); } }
</style>
<div class="flower" style="left:5%;  --x-move:-80px; animation-duration:6s;">🌸</div>
<div class="flower" style="left: 20%; --x-move:-100px; animation-duration: 4s;">🧨</div>
<div class="flower" style="left:15%; --x-move:120px; animation-duration:7s;">🌷</div>
<div class="flower" style="left:30%; --x-move:-60px; animation-duration:7.5s;">💐</div>
<div class="flower" style="left:37%; --x-move:70px; animation-duration:8s;">✨</div>
<div class="flower" style="left:25%; --x-move:-150px; animation-duration:8s;">🌼</div>
<div class="flower" style="left: 50%; --x-move:-100px; animation-duration: 4s;">🧨</div>
<div class="flower" style="left:35%; --x-move:90px; animation-duration:6.5s;">🌺</div>
<div class="flower" style="left: 85%; --x-move:130px; animation-duration: 15s;">🍀</div>
<div class="flower" style="left:45%; --x-move:-60px; animation-duration:7.5s;">💐</div>
<div class="flower" style="left:55%; --x-move:140px; animation-duration:9s;">🌸</div>
<div class="flower" style="left: 85%; --x-move:130px; animation-duration: 15s;">🍀</div>
<div class="flower" style="left:65%; --x-move:-120px; animation-duration:6.8s;">🌷</div>
<div class="flower" style="left: 81%; --x-move:-100px; animation-duration: 4s;">🧨</div>
<div class="flower" style="left:75%; --x-move:70px; animation-duration:8.2s;">🌼</div>
<div class="flower" style="left:40%; --x-move:-100px; animation-duration:7.2s;">🌺</div>
<div class="flower" style="left:99%; --x-move:70px; animation-duration:8s;">✨</div>
""",
unsafe_allow_html=True
)


st.title("🤖 Tư vấn TTHC Trẻ em dưới 6 tuổi")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Cấu hình")
    top_k = st.slider("Số lượng chunk lấy về", 1, 10, 3)
    st.divider()
    st.subheader("ℹ️ Thông tin hệ thống")
    if collection:
        st.success(f"✅ Đã kết nối Database")
        st.write(f"🧩 Số chunk: {collection.count()}")
    else:
        st.error("❌ Chưa tìm thấy dữ liệu")

# Lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý nhập liệu
if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm dữ liệu..."):
            answer = query_rag(prompt, top_k)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
