import streamlit as st
import pandas as pd
import numpy as np
import re
from supabase import create_client, Client, ClientOptions
from httpx import Timeout
import httpx 
from sentence_transformers import SentenceTransformer

# --- 1. 페이지 설정 (수정됨) ---
st.set_page_config(
    page_title="병원 규정 AI 검색기",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed" # ★ [수정] 사이드바 기본 닫힘 설정
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
        st.error(f"❌ [오류] 서비스 연결에 실패했습니다: {e}")
        return None, None

@st.cache_data(ttl=600)
def load_map_data(_supabase: Client):
    try:
        response = _supabase.table("regulations_map").select(
            "id, ch_name, std_id, std_name, me_id, me_name, pdf_filename, pdf_url"
        ).order("id").execute()
        
        df = pd.DataFrame(response.data)
        if df.empty:
            st.error("❌ [오류] 데이터가 없습니다.")
            return pd.DataFrame()
        
        def create_sort_key(std_id_str):
            try:
                parts = re.split(r'[.-]', str(std_id_str))
                return tuple(int(p) for p in parts if p.isdigit())
            except ValueError:
                return (0,)
                
        df['std_sort_key'] = df['std_id'].apply(create_sort_key)
        df = df.sort_values(by=['std_sort_key', 'me_id']) 
        return df
    except Exception as e:
        st.error(f"❌ [오류] 데이터를 불러오는 중 문제가 발생했습니다: {e}")
        return pd.DataFrame()

# --- 3. 핵심 기능 함수 ---

def run_ai_search(query_text: str, search_mode: str, _supabase: Client, _model: SentenceTransformer):
    if not query_text or not _supabase or not _model:
        return [], None
        
    try:
        query_vector = _model.encode(query_text).tolist()
        
        if search_mode == "[AI] 제목/분류 검색":
            st.session_state.ai_status = "✅ '제목/분류'에서 AI 검색 중..."
            response = _supabase.rpc('match_map', {
                'query_vector': query_vector,
                'match_threshold': 0.3, 
                'match_count': 10 
            }).execute()
            return response.data, "map" 
            
        else: 
            st.session_state.ai_status = "✅ '본문 전체'에서 AI 검색 중..."
            response = _supabase.rpc('match_chunks_all', {
                'query_vector': query_vector,
                'match_threshold': 0.5, 
                'match_count': 5
            }).execute()
            return response.data, "chunks" 

    except Exception as e:
        st.error(f"❌ [오류] AI 검색 중 문제가 발생했습니다: {e}")
        return [], None

# ★★★ [수정] 강력해진 PDF 뷰어 함수 ★★★
def get_pdf_embed_html(pdf_url: str, page: int = 1) -> str:
    """ 
    Chrome 차단 문제를 해결하기 위해 <embed> 태그 사용 및 HTTPS 강제 적용 
    """
    if not pdf_url:
        return "<p>PDF URL이 없습니다.</p>"
    
    # 1. HTTPS 강제 (Mixed Content 차단 방지)
    if pdf_url.startswith("http://"):
        pdf_url = pdf_url.replace("http://", "https://")

    page_to_show = max(1, page)
    # 캐시 버스팅 및 페이지 점프
    final_url = f"{pdf_url}?v={page_to_show}#page={page_to_show}"
    
    # 2. <embed> 태그 사용 (iframe보다 호환성 좋음)
    return f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
            <a href="{pdf_url}" target="_blank" style="background-color: #ff4b4b; color: white; padding: 5px 10px; text-decoration: none; border-radius: 5px; font-size: 0.8rem;">
                ↗️ 새 창에서 PDF 열기 (오류 시 클릭)
            </a>
        </div>
        <embed src="{final_url}" type="application/pdf" width="100%" height="1000px" />
    """
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★

def set_pdf_url(url: str, page: int):
    st.session_state.current_pdf_url = url
    st.session_state.current_pdf_page = page
    st.session_state.view_mode = "preview" 

# --- 4. Streamlit UI 구성 ---

# (보안 체크)
def check_password():
    if "password" not in st.session_state or st.session_state.password == "":
        st.session_state.is_authenticated = False
        return
        
    if st.session_state["password"] == st.secrets["app_security"]["common_password"]:
        st.session_state["is_authenticated"] = True
        del st.session_state["password"]
    else:
        st.session_state["is_authenticated"] = False
        st.error("비밀번호가 올바르지 않습니다.")

if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "preview"
if "current_pdf_url" not in st.session_state:
    st.session_state.current_pdf_url = None
if "current_pdf_page" not in st.session_state:
    st.session_state.current_pdf_page = 1
if "ai_status" not in st.session_state:
    st.session_state.ai_status = ""

# --- 로그인 화면 ---
if not st.session_state.is_authenticated:
    st.title("🏥 병원 규정 AI 검색기")
    with st.container(border=True): 
        st.subheader("로그인")
        st.markdown("병원 공통 비밀번호를 입력하세요.")
        st.text_input(
            "비밀번호", 
            type="password", 
            on_change=check_password, 
            key="password"
        )
    st.stop() 

# --- 메인 앱 ---
supabase, ai_model = init_connections()
if not supabase or not ai_model:
    st.stop()

map_data = load_map_data(supabase)
if map_data.empty:
    st.stop()
    
try:
    combined_pdf_url = supabase.storage.from_("regulations").get_public_url("combined_regulations.pdf")
except Exception:
    combined_pdf_url = None

st.title("🏥 병원 규정 AI 검색기")

# --- 뷰어 로직 ---
if st.session_state.view_mode == "fullscreen":
    st.button("🔙 목록 보기", on_click=lambda: st.session_state.update(view_mode="preview"), width='stretch')
    
    if st.session_state.current_pdf_url:
        st.markdown(
            get_pdf_embed_html(st.session_state.current_pdf_url, st.session_state.current_pdf_page), 
            unsafe_allow_html=True
        )
    else:
        st.info("표시할 PDF가 선택되지 않았습니다.")

else:
    col_nav, col_viewer = st.columns([1, 2]) 

    with col_nav:
        st.header("탐색")
        
        if combined_pdf_url:
            st.button(
                "📂 [전체 합본 보기]", 
                on_click=set_pdf_url, 
                args=(combined_pdf_url, 1),
                key="btn_combined_pdf",
                width='stretch'
            )
        
        st.divider()
        
        search_mode = st.radio(
            "검색 모드", 
            ["[AI] 제목/분류 검색", "[AI] 본문 내용 검색", "제목 검색 (키워드)"], 
            horizontal=True
        )
        
        search_query = st.text_input("🔍 검색어 입력", placeholder="예: 낙상 평가도구, HIS-1.1")
        
        st.subheader("규정 목록")
        
        if not search_query:
            st.session_state.ai_status = "" 
            for ch_name, ch_df in map_data.groupby('ch_name', sort=False):
                with st.expander(f"📂 {ch_name}"):
                    for std_name, std_df in ch_df.groupby('std_name', sort=False):
                        std_id = std_df.iloc[0]['std_id']
                        with st.expander(f"📙 {std_id} {std_name}"):
                            for _, row in std_df.iterrows():
                                st.button(
                                    f"📄 [{row['me_id']}] {row['me_name']}",
                                    key=f"btn_me_{row['id']}", 
                                    on_click=set_pdf_url,
                                    args=(row['pdf_url'], 1) 
                                )
        else:
            ai_results = []
            result_type = None
            filtered_df = pd.DataFrame() 

            if search_mode == "[AI] 제목/분류 검색":
                with st.spinner("🧠 AI가 '제목/분류'(을)를 검색 중입니다..."):
                    st.session_state.ai_status = "..." 
                    ai_results, result_type = run_ai_search(search_query, search_mode, supabase, ai_model)
                
                if not ai_results:
                    st.info(f"ℹ️ 결과가 없습니다.")
                else:
                    filtered_df = pd.DataFrame(ai_results)
                    st.markdown(f"**'{search_query}'(와)과 유사한 {len(filtered_df)}건을 찾았습니다.**")

            elif search_mode == "[AI] 본문 내용 검색":
                with st.spinner("🧠 AI가 '본문 전체'(을)를 검색 중입니다..."):
                    st.session_state.ai_status = "..." 
                    ai_results, result_type = run_ai_search(search_query, search_mode, supabase, ai_model)
                
                if not ai_results:
                    st.info(f"ℹ️ 결과가 없습니다.")
                else:
                    st.markdown(f"**'{search_query}'(와)과 유사한 {len(ai_results)}건의 본문을 찾았습니다.**")
                    
                    url_map = map_data.drop_duplicates(subset=['pdf_filename'])
                    url_map = pd.Series(url_map.pdf_url.values, index=url_map.pdf_filename).to_dict()

                    for row in ai_results:
                        st.markdown(f"---")
                        st.info(f"**(p.{row['page_num']}에서 발견)** (유사도: {row['similarity']:.0%})")
                        
                        chunk_content = row['context_chunk']
                        if "[본문] " in chunk_content:
                            chunk_content = chunk_content.split("[본문] ", 1)[-1]
                            
                        highlighted_text = chunk_content.replace(search_query, f"**{search_query}**") 
                        st.markdown(f"> {highlighted_text}...")
                        
                        result_filename = row['pdf_filename']
                        pdf_url_to_open = url_map.get(result_filename) 
                        
                        if pdf_url_to_open:
                            st.button(
                                f"↗️ 규정 보기 ({result_filename}, {row['page_num']}p.로 이동)",
                                key=f"ai_btn_chunk_{row['id']}",
                                on_click=set_pdf_url,
                                args=(pdf_url_to_open, row['page_num'])
                            )

            elif search_mode == "제목 검색 (키워드)":
                st.session_state.ai_status = "" 
                query = search_query.lower()
                mask = (
                    map_data['ch_name'].str.lower().str.contains(query, na=False) |
                    map_data['std_name'].str.lower().str.contains(query, na=False) |
                    map_data['me_name'].str.lower().str.contains(query, na=False) |
                    map_data['std_id'].str.lower().str.contains(query, na=False) |
                    map_data['me_id'].str.lower().str.contains(query, na=False)
                )
                filtered_df = map_data[mask]
                
                if filtered_df.empty:
                    st.info("ℹ️ 검색 결과가 없습니다.")
                else:
                    st.markdown(f"**'{search_query}'(으)로 {len(filtered_df)}건을 찾았습니다.**")

            if not filtered_df.empty:
                if result_type == "map":
                     result_ids = filtered_df['id'].tolist()
                     filtered_df = map_data[map_data['id'].isin(result_ids)].set_index('id').loc[result_ids].reset_index()

                for ch_name, ch_df in filtered_df.groupby('ch_name', sort=False):
                    with st.expander(f"📂 {ch_name}", expanded=True):
                        for std_name, std_df in ch_df.groupby('std_name', sort=False):
                            std_id = std_df.iloc[0]['std_id']
                            with st.expander(f"📙 {std_id} {std_name}", expanded=True):
                                for _, row in std_df.iterrows():
                                    st.button(
                                        f"📄 [{row['me_id']}] {row['me_name']}",
                                        key=f"btn_me_filtered_{row['id']}", 
                                        on_click=set_pdf_url,
                                        args=(row['pdf_url'], 1) 
                                    )

    with col_viewer:
        st.button(
            "↗️ 전체 화면으로 보기", 
            on_click=lambda: st.session_state.update(view_mode="fullscreen"), 
            width='stretch',
            disabled=(st.session_state.current_pdf_url is None)
        )
        st.divider()

        if st.session_state.current_pdf_url:
            st.markdown(
                get_pdf_embed_html(st.session_state.current_pdf_url, st.session_state.current_pdf_page), 
                unsafe_allow_html=True
            )
        else:
            st.info("규정을 선택하면 여기에 미리보기가 표시됩니다.")

# --- 관리자 패널 ---
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False

st.sidebar.title("관리자 패널")
if st.session_state.is_admin:
    st.sidebar.success("관리자 모드 활성화")
    st.sidebar.markdown("---")
    st.sidebar.dataframe(map_data.head())
else:
    admin_pw = st.sidebar.text_input("관리자 암호:", type="password")
    if admin_pw:
        if admin_pw == st.secrets["app_security"]["admin_password"]:
            st.session_state.is_admin = True
            st.rerun()
