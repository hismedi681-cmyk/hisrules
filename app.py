import streamlit as st
import pandas as pd
import numpy as np
import re
from supabase import create_client, Client, ClientOptions
from httpx import Timeout
import httpx 
from sentence_transformers import SentenceTransformer
from streamlit_pdf_viewer import pdf_viewer # ★ [핵심] 전용 뷰어 임포트

# --- 1. 페이지 설정 ---
st.set_page_config(
    page_title="병원 규정 AI 검색기",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. Supabase 및 AI 모델 연결 ---
@st.cache_resource
def init_connections():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["anon_key"]
        
        default_timeout = Timeout(10.0, read=10.0)
        supabase_options = ClientOptions(
            httpx_client=httpx.Client(timeout=default_timeout)
        )
        supabase = create_client(url, key, options=supabase_options)
        ai_model = SentenceTransformer('jhgan/ko-sbert-nli')
        return supabase, ai_model
    except Exception as e:
        st.error(f"❌ [오류] 서비스 연결 실패: {e}")
        return None, None

@st.cache_data(ttl=600)
def load_map_data(_supabase: Client):
    try:
        response = _supabase.table("regulations_map").select(
            "id, ch_name, std_id, std_name, me_id, me_name, pdf_filename, pdf_url"
        ).order("id").execute()
        
        df = pd.DataFrame(response.data)
        if df.empty: return pd.DataFrame()
        
        def create_sort_key(std_id_str):
            try:
                parts = re.split(r'[.-]', str(std_id_str))
                return tuple(int(p) for p in parts if p.isdigit())
            except ValueError:
                return (0,)
        df['std_sort_key'] = df['std_id'].apply(create_sort_key)
        return df.sort_values(by=['std_sort_key', 'me_id'])
    except Exception:
        return pd.DataFrame()

# --- 3. 핵심 기능 함수 ---
def run_ai_search(query_text, search_mode, _supabase, _model):
    if not query_text: return [], None
    try:
        query_vector = _model.encode(query_text).tolist()
        if search_mode == "[AI] 제목/분류 검색":
            st.session_state.ai_status = "✅ '제목/분류' 검색 중..."
            response = _supabase.rpc('match_map', {
                'query_vector': query_vector, 'match_threshold': 0.3, 'match_count': 10
            }).execute()
            return response.data, "map"
        else: 
            st.session_state.ai_status = "✅ '본문 전체' 검색 중..."
            response = _supabase.rpc('match_chunks_all', {
                'query_vector': query_vector, 'match_threshold': 0.5, 'match_count': 5
            }).execute()
            return response.data, "chunks"
    except Exception:
        return [], None

# ★★★ [핵심 수정] PDF 뷰어 함수 (라이브러리 사용) ★★★
@st.cache_data(ttl=3600) # PDF 데이터를 캐싱하여 속도 향상
def download_pdf_data(url: str):
    """ Supabase URL에서 PDF 바이너리 데이터를 다운로드합니다. """
    try:
        # HTTPS 강제 변환 (보안 이슈 방지)
        if url.startswith("http://"):
            url = url.replace("http://", "https://")
            
        response = httpx.get(url, timeout=10.0)
        if response.status_code == 200:
            return response.content
        return None
    except Exception:
        return None

def render_pdf_viewer(pdf_url: str, page: int = 1):
    """ streamlit-pdf-viewer 라이브러리로 안전하게 PDF 표시 """
    if not pdf_url:
        st.warning("PDF URL이 없습니다.")
        return

    with st.spinner("📄 PDF 문서를 불러오는 중..."):
        pdf_data = download_pdf_data(pdf_url)
        
    if pdf_data:
        # width를 설정하면 반응형으로 꽉 차게 보입니다.
        # resolution을 높이면 글자가 선명해집니다.
        pdf_viewer(input=pdf_data, width=700, height=1000, resolution_boost=1.5)
        
        # (참고) 이 라이브러리는 아직 특정 페이지로 자동 스크롤하는 기능이 불안정하여
        # 전체 문서를 보여주되, 사용자가 스크롤하도록 유도합니다.
        if page > 1:
            st.caption(f"💡 **{page}페이지**를 참고하세요.")
    else:
        st.error("❌ PDF 파일을 다운로드할 수 없습니다. (URL 오류 또는 권한 문제)")
        st.link_button("↗️ 새 창에서 직접 열기", pdf_url)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

def set_pdf_url(url: str, page: int):
    st.session_state.current_pdf_url = url
    st.session_state.current_pdf_page = page
    st.session_state.view_mode = "preview" 

# --- 4. UI 구성 ---

# (보안 체크)
def check_password():
    if "password" not in st.session_state: return
    if st.session_state["password"] == st.secrets["app_security"]["common_password"]:
        st.session_state["is_authenticated"] = True
        del st.session_state["password"]
    else:
        st.error("비밀번호 오류")

if "is_authenticated" not in st.session_state: st.session_state.is_authenticated = False
if "view_mode" not in st.session_state: st.session_state.view_mode = "preview"
if "current_pdf_url" not in st.session_state: st.session_state.current_pdf_url = None
if "current_pdf_page" not in st.session_state: st.session_state.current_pdf_page = 1
if "ai_status" not in st.session_state: st.session_state.ai_status = ""

if not st.session_state.is_authenticated:
    st.title("🏥 병원 규정 AI 검색기")
    with st.container(border=True):
        st.text_input("비밀번호", type="password", on_change=check_password, key="password")
    st.stop()

# (메인 앱)
supabase, ai_model = init_connections()
if not supabase or not ai_model: st.stop()
map_data = load_map_data(supabase)

st.title("🏥 병원 규정 AI 검색기")

col_nav, col_viewer = st.columns([1, 1.5]) # 뷰어 공간 확보를 위해 비율 조정

with col_nav:
    st.header("탐색")
    search_mode = st.radio("모드", ["[AI] 제목/분류 검색", "[AI] 본문 내용 검색", "제목 검색 (키워드)"])
    search_query = st.text_input("검색어", placeholder="예: 낙상")
    
    st.subheader("규정 목록")
    
    # (리스트/아코디언 로직 - 간소화하여 유지)
    target_df = map_data
    ai_result_type = None
    
    if search_query:
        if "[AI]" in search_mode:
            with st.spinner("AI 검색 중..."):
                ai_results, ai_result_type = run_ai_search(search_query, search_mode, supabase, ai_model)
                if ai_results:
                    if ai_result_type == "map":
                         ids = [r['id'] for r in ai_results]
                         target_df = map_data[map_data['id'].isin(ids)]
                    # 본문 검색은 별도 리스트로 표시
                    elif ai_result_type == "chunks":
                        url_map = map_data.drop_duplicates('pdf_filename').set_index('pdf_filename')['pdf_url'].to_dict()
                        for row in ai_results:
                            with st.container(border=True):
                                st.caption(f"유사도: {row['similarity']:.0%}")
                                chunk = row['context_chunk'].split("[본문]")[-1] if "[본문]" in row['context_chunk'] else row['context_chunk']
                                st.markdown(f"...{chunk[:100]}...")
                                pdf_url = url_map.get(row['pdf_filename'])
                                if pdf_url:
                                    st.button(f"📄 {row['pdf_filename']} (p.{row['page_num']})", 
                                              key=f"c_{row['id']}", 
                                              on_click=set_pdf_url, args=(pdf_url, row['page_num']))
                        target_df = pd.DataFrame() # 아코디언 숨김

        elif "키워드" in search_mode:
            q = search_query.lower()
            target_df = map_data[map_data['me_name'].str.lower().str.contains(q) | map_data['std_name'].str.lower().str.contains(q)]

    # (아코디언 렌더링)
    if not target_df.empty:
        for ch, ch_df in target_df.groupby('ch_name', sort=False):
            with st.expander(f"📂 {ch}", expanded=bool(search_query)):
                for std, std_df in ch_df.groupby('std_name', sort=False):
                    std_id = std_df.iloc[0]['std_id']
                    st.caption(f"📙 {std_id} {std}")
                    for _, row in std_df.iterrows():
                        st.button(f"📄 {row['me_name']}", key=f"btn_{row['id']}", 
                                  on_click=set_pdf_url, args=(row['pdf_url'], 1))

with col_viewer:
    st.header("미리보기")
    if st.session_state.current_pdf_url:
        # ★ 여기서 새로운 뷰어 함수 호출
        render_pdf_viewer(st.session_state.current_pdf_url, st.session_state.current_pdf_page)
    else:
        st.info("왼쪽에서 규정을 선택하세요.")
