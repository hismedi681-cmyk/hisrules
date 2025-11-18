import streamlit as st
import pandas as pd
import numpy as np
import re
import base64 # ★ [핵심] 데이터를 직접 주입하기 위한 라이브러리
from supabase import create_client, Client, ClientOptions
from httpx import Timeout
import httpx 
from sentence_transformers import SentenceTransformer

# --- 1. 페이지 설정 ---
st.set_page_config(
    page_title="병원 규정 AI 검색기",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed" # 관리자 패널 닫힘 상태 시작
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

# ★★★ [핵심 전략] Base64 인코딩을 통한 보안 우회 뷰어 ★★★
@st.cache_data(ttl=3600)
def get_pdf_base64(url: str):
    """ PDF URL을 받아 Base64 문자열로 변환합니다. (보안 우회) """
    try:
        if url.startswith("http://"): url = url.replace("http://", "https://")
        response = httpx.get(url, timeout=10.0)
        if response.status_code == 200:
            # 바이너리 데이터를 base64 문자열로 인코딩
            return base64.b64encode(response.content).decode('utf-8')
    except:
        pass
    return None

def render_native_pdf(pdf_url: str, page: int = 1):
    """ 브라우저 자체 PDF 뷰어를 강제로 활성화하는 HTML 생성 """
    if not pdf_url:
        st.info("규정을 선택하세요.")
        return

    with st.spinner("📄 PDF 뷰어 로딩 중..."):
        # 1. 서버에서 PDF 데이터를 직접 가져옴 (CORS 우회)
        base64_pdf = get_pdf_base64(pdf_url)
    
    if base64_pdf:
        # 2. 데이터를 브라우저에게 '내부 데이터'인 것처럼 속여서 주입 (data:application/pdf;base64)
        # '#page=N' 태그를 사용하여 해당 페이지로 이동
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page}" width="100%" height="1000px" type="application/pdf" style="border:none;"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.error("❌ PDF 데이터를 불러올 수 없습니다.")
        st.link_button("↗️ 새 창에서 열기", pdf_url)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

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

# (전체 화면 모드)
if st.session_state.view_mode == "fullscreen":
    st.button("🔙 목록 보기", on_click=lambda: st.session_state.update(view_mode="preview"), width='stretch')
    if st.session_state.current_pdf_url:
        # ★ 수정된 네이티브 뷰어 호출
        render_native_pdf(st.session_state.current_pdf_url, st.session_state.current_pdf_page)

# (분할 화면 모드)
else:
    col_nav, col_viewer = st.columns([1, 1.5]) 

    with col_nav:
        st.header("탐색")
        search_mode = st.radio("모드", ["[AI] 제목/분류 검색", "[AI] 본문 내용 검색", "제목 검색 (키워드)"])
        search_query = st.text_input("검색어", placeholder="예: 낙상")
        
        st.subheader("규정 목록")
        
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
            # ★ 수정된 네이티브 뷰어 호출
            render_native_pdf(st.session_state.current_pdf_url, st.session_state.current_pdf_page)
        else:
            st.info("왼쪽에서 규정을 선택하세요.")

        st.divider()
        st.button(
            "↗️ 전체 화면으로 보기", 
            on_click=lambda: st.session_state.update(view_mode="fullscreen"), 
            width='stretch',
            disabled=(st.session_state.current_pdf_url is None)
        )
