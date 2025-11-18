import streamlit as st
import pandas as pd
import numpy as np
import re
# Base64 import는 더 이상 필요 없습니다.
from supabase import create_client, Client, ClientOptions
from httpx import Timeout
import httpx 
from sentence_transformers import SentenceTransformer
# streamlit_pdf_viewer import는 더 이상 필요 없습니다.

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
        if df.empty: 
            st.warning("⚠️ '지도' 데이터가 없습니다. admin_sync.py를 실행했는지 확인하세요.")
            return pd.DataFrame()
        
        def create_sort_key(std_id_str):
            try:
                parts = re.split(r'[.-]', str(std_id_str))
                return tuple(int(p) for p in parts if p.isdigit())
            except ValueError:
                return (0,)
        df['std_sort_key'] = df['std_id'].apply(create_sort_key)
        return df.sort_values(by=['std_sort_key', 'me_id'])
    except Exception as e:
        st.error(f"❌ [오류] '지도' 데이터를 불러오는 중 문제가 발생했습니다: {e}")
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
    except Exception as e:
        st.error(f"❌ [오류] AI 검색 중 문제가 발생했습니다: {e}")
        return [], None

# --- PDF 뷰어 함수 (대안 1) ---

# get_pdf_bytes 함수는 더 이상 사용하지 않습니다.

def render_pdf_with_anchor_jump(pdf_url: str, page: int = 1):
    """ 
    [대안 1] Native Browser 뷰어 (URL #Anchor) 사용
    - 스크롤 자유 + 자동 점프 기능 구현
    """
    if not pdf_url:
        st.info("규정을 선택하세요.")
        return

    target_page = int(page)
    
    st.markdown(f"**📍 AI가 찾은 페이지:** {target_page} 페이지 (스크롤하여 전체 문맥을 확인하세요)")
    
    # [안전장치] 원본 링크 제공
    st.markdown(f"""
    <a href="{pdf_url}#page={target_page}" target="_blank" style="
        display: inline-block;
        background-color: #f0f2f6;
        color: #31333F;
        padding: 6px 12px;
        border-radius: 4px;
        text-decoration: none;
        font-size: 14px;
        margin-bottom: 10px;
        border: 1px solid #d6d6d8;">
        ↗️ 새 창에서 PDF 원본 열기 (자동 점프 포함)
    </a>
    """, unsafe_allow_html=True)
    
    st.divider()

    # ★ 핵심 수정: iframe으로 URL을 직접 로드하고, #page=을 붙여 점프 기능 활성화
    st.markdown(f"""
    <iframe src="{pdf_url}#page={target_page}" 
            width="100%" 
            height="1000px" 
            type="application/pdf" 
            style="border:none;">
    </iframe>
    """, unsafe_allow_html=True)

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

# 합본 PDF URL 가져오기
try:
    combined_pdf_url = supabase.storage.from_("regulations").get_public_url("combined_regulations.pdf")
except Exception:
    combined_pdf_url = None

st.title("🏥 병원 규정 AI 검색기")

# (전체 화면 모드)
if st.session_state.view_mode == "fullscreen":
    st.button("🔙 목록 보기", on_click=lambda: st.session_state.update(view_mode="preview"), width='stretch')
    if st.session_state.current_pdf_url:
        render_pdf_with_anchor_jump(st.session_state.current_pdf_url, st.session_state.current_pdf_page)

# (분할 화면 모드)
else:
    col_nav, col_viewer = st.columns([1, 1.5]) 

    with col_nav:
        if combined_pdf_url:
            st.button(
                "📂 [전체 합본 보기]", 
                on_click=set_pdf_url, 
                args=(combined_pdf_url, 1),
                key="btn_combined_pdf",
                width='stretch'
            )
        
        st.divider()
        
        search_mode = st.radio("모드", ["[AI] 제목/분류 검색", "[AI] 본문 내용 검색", "제목 검색 (키워드)"])
        search_query = st.text_input("검색어", placeholder="예: 낙상")
        
        st.markdown("### 규정 목록")
        
        target_df = map_data
        ai_result_type = None
        
        if search_query:
            if "[AI]" in search_mode:
                with st.spinner(st.session_state.ai_status if st.session_state.ai_status else "AI 검색 중..."):
                    ai_results, ai_result_type = run_ai_search(search_query, search_mode, supabase, ai_model)
                    
                    if not ai_results:
                        st.info("ℹ️ 결과가 없습니다.")
                        target_df = pd.DataFrame()
                    else:
                        if ai_result_type == "map":
                             ids = [r['id'] for r in ai_results]
                             target_df = map_data[map_data['id'].isin(ids)]
                        elif ai_result_type == "chunks":
                            st.markdown(f"##### 🔍 '{search_query}' 관련 본문 검색 결과 ({len(ai_results)}건)")
                            url_map = map_data.drop_duplicates(subset=['pdf_filename'])
                            url_map = pd.Series(url_map.pdf_url.values, index=url_map.pdf_filename).to_dict()

                            for row in ai_results:
                                with st.container(border=True):
                                    c1, c2 = st.columns([4, 1])
                                    c1.markdown(f"**📄 {row['pdf_filename']}** (p.{row['page_num']})")
                                    score = row['similarity']
                                    color = "green" if score >= 0.6 else "orange" if score >= 0.5 else "gray"
                                    c2.markdown(f":{color}[**{score:.0%}**]")
                                    
                                    raw_text = row['context_chunk']
                                    clean_text = raw_text.replace("[본문]", "").strip()
                                    if clean_text.startswith("...Ÿ"): clean_text = clean_text.replace("...Ÿ", "...")
                                    if search_query:
                                        clean_text = clean_text.replace(search_query, f":red[**{search_query}**]")
                                    st.markdown(f"...{clean_text}...")
                                    
                                    pdf_url = url_map.get(row['pdf_filename'])
                                    if pdf_url:
                                        st.button(
                                            "👉 이 페이지 바로 보기",
                                            key=f"btn_chunk_{row['id']}",
                                            on_click=set_pdf_url,
                                            args=(pdf_url, row['page_num']),
                                            use_container_width=True
                                        )
                            target_df = pd.DataFrame()

            elif "키워드" in search_mode:
                q = search_query.lower()
                target_df = map_data[map_data['ch_name'].str.lower().str.contains(q) | 
                                     map_data['std_name'].str.lower().str.contains(q) | 
                                     map_data['me_name'].str.lower().str.contains(q)]
                if target_df.empty: st.info("결과가 없습니다.")

        if not target_df.empty:
            should_expand = True if search_query else False
            
            for ch_name, ch_df in target_df.groupby('ch_name', sort=False):
                with st.expander(f"📂 {ch_name}", expanded=should_expand):
                    for std_name, std_df in ch_df.groupby('std_name', sort=False):
                        std_id = std_df.iloc[0]['std_id']
                        with st.expander(f"📙 {std_id} {std_name}", expanded=should_expand):
                            for _, row in std_df.iterrows():
                                st.button(f"📄 {row['me_name']}", key=f"btn_{row['id']}", 
                                          on_click=set_pdf_url, args=(row['pdf_url'], 1))

    with col_viewer:
        st.button(
            "↗️ 전체 화면으로 보기", 
            on_click=lambda: st.session_state.update(view_mode="fullscreen"), 
            width='stretch',
            disabled=(st.session_state.current_pdf_url is None)
        )
        
        st.divider()

        if st.session_state.current_pdf_url:
            # ★★★ 핵심 변경: 대안 1 함수 호출 ★★★
            render_pdf_with_anchor_jump(st.session_state.current_pdf_url, st.session_state.current_pdf_page)
        else:
            st.info("왼쪽에서 규정을 선택하세요.")

# --- 관리자 패널 ---
if 'is_admin' not in st.session_state: st.session_state.is_admin = False
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
        else:
            st.sidebar.error("암호가 틀렸습니다.")



