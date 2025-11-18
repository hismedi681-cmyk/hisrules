import streamlit as st
import pandas as pd
import numpy as np
import re
from supabase import create_client, Client, ClientOptions
from httpx import Timeout
import httpx 
from sentence_transformers import SentenceTransformer

# --- 1. 페이지 설정 (파일 최상단) ---
st.set_page_config(
    page_title="병원 규정 AI 검색기",
    page_icon="🏥",
    layout="wide"
)

# --- 2. Supabase 및 AI 모델 연결 ---
@st.cache_resource
def init_connections():
    """
    secrets.toml에서 연결 정보를 읽어 Supabase 클라이언트와 AI 모델을 초기화합니다.
    """
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
    """
    [2-Track 수정] Supabase DB에서 '지도(map)' 데이터만 로드합니다. (아코디언 UI용)
    """
    try:
        # ★★★ 'match_map'이 반환하는 모든 컬럼을 가져오도록 수정 ★★★
        response = _supabase.table("regulations_map").select(
            "id, ch_name, std_id, std_name, me_id, me_name, pdf_filename, pdf_url"
        ).order("id").execute()
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        
        df = pd.DataFrame(response.data)
        if df.empty:
            st.error("❌ [오류] '지도(regulations_map)' 데이터를 불러오지 못했습니다. admin_sync.py를 먼저 실행하세요.")
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
        st.error(f"❌ [오류] '지도' 데이터를 불러오는 중 문제가 발생했습니다: {e}")
        return pd.DataFrame()

# --- 3. 핵심 기능 함수 ---

def run_ai_search(query_text: str, search_mode: str, _supabase: Client, _model: SentenceTransformer):
    """
    "2-Track" 전략에 따라 올바른 Supabase 함수를 호출합니다.
    """
    if not query_text or not _supabase or not _model:
        return [], None
        
    try:
        query_vector = _model.encode(query_text).tolist()
        
        if search_mode == "[AI] 제목/분류 검색":
            st.session_state.ai_status = "✅ '제목/분류'에서 AI 검색 중..."
            response = _supabase.rpc('match_map', {
                'query_vector': query_vector,
                'match_threshold': 0.3, 
                'match_count': 10 # <-- 아코디언 구성을 위해 더 많이 가져옵니다.
            }).execute()
            return response.data, "map" 
            
        else: # "[AI] 본문 내용 검색"
            st.session_state.ai_status = "✅ '본문 전체'에서 AI 검색 중..."
            response = _supabase.rpc('match_chunks_all', {
                'query_vector': query_vector,
                'match_threshold': 0.5, 
                'match_count': 5
            }).execute()
            return response.data, "chunks" 

    except Exception as e:
        st.error(f"❌ [오류] AI 검색 중 문제가 발생했습니다: {e}")
        st.exception(e)
        return [], None

def get_pdf_embed_html(pdf_url: str, page: int = 1) -> str:
    """ PDF 임베드 HTML 생성 (페이지 점프 기능 포함) """
    if not pdf_url:
        return "<p>PDF URL이 없습니다.</p>"
    page_to_show = max(1, page)
    final_url = f"{pdf_url}?v={page_to_show}#page={page_to_show}"
    
    return f"""
        <iframe src="{final_url}&navpanes=0&toolbar=0" width="100%" height="1000px" style="border:none;">
            <p>PDF를 표시할 수 없습니다.</p>
        </iframe>
    """

def set_pdf_url(url: str, page: int):
    """ PDF 뷰어 상태 변경 콜백 """
    st.session_state.current_pdf_url = url
    st.session_state.current_pdf_page = page
    st.session_state.view_mode = "preview" 

# --- 4. Streamlit UI 구성 ---

# --- 4-0. 앱 보안 (공통 비밀번호) ---
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

# (세션 상태 초기화)
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

# --- 비밀번호 입력 화면 ---
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

# --- 비밀번호 통과 시, 메인 앱 로드 ---

# (서비스 연결)
supabase, ai_model = init_connections()
if not supabase or not ai_model:
    st.stop()

# (데이터 로드)
map_data = load_map_data(supabase) # (아코디언용 원본 데이터)
if map_data.empty:
    st.stop()
    
# (합본 PDF URL 미리 가져오기)
try:
    combined_pdf_url = supabase.storage.from_("regulations").get_public_url("combined_regulations.pdf")
except Exception:
    combined_pdf_url = None

st.title("🏥 병원 규정 AI 검색기")

# --- 전체 화면 로직 ---
if st.session_state.view_mode == "fullscreen":
    st.button("🔙 목록 보기", on_click=lambda: st.session_state.update(view_mode="preview"), width='stretch')
    
    if st.session_state.current_pdf_url:
        st.markdown(
            get_pdf_embed_html(st.session_state.current_pdf_url, st.session_state.current_pdf_page), 
            unsafe_allow_html=True
        )
    else:
        st.info("표시할 PDF가 선택되지 않았습니다. '목록 보기'로 돌아가세요.")

else:
    # --- [미리보기 모드 (기본)] ---
    col_nav, col_viewer = st.columns([1, 2]) # 1:2 비율

    # --- 좌측 네비게이터 (col_nav) ---
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
            ["[AI] 제목/분류 검색", "[AI] 본문 내용 검색", "제목 검색 (키워드)"], # <-- 3개 모드
            horizontal=True,
            help="""
            - **[AI] 제목/분류 검색:** '환자 확인'처럼 특정 기준(ME)이나 규정집 제목을 AI로 찾습니다. (아코디언 필터링)
            - **[AI] 본문 내용 검색:** '손씻기 절차'처럼 규정집 본문의 상세 내용을 AI로 찾습니다. (본문 조각 리스트)
            - **제목 검색 (키워드):** 'HIS-1.1'처럼 정확한 키워드로 아코디언을 필터링합니다.
            """
        )
        
        search_query = st.text_input("🔍 검색어 입력", placeholder="예: 낙상 평가도구, 개방형 질문, HIS-1.1")
        
        st.subheader("규정 목록")
        
        # (1. 검색어가 없을 때 - 기본 아코디언)
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
        
        # (2. 검색어가 있을 때 - 필터링된 결과)
        else:
            ai_results = []
            result_type = None
            filtered_df = pd.DataFrame() # 아코디언을 그릴 DataFrame

            # --- [AI] 제목/분류 검색 로직 (아코디언 필터링) ---
            if search_mode == "[AI] 제목/분류 검색":
                with st.spinner("🧠 AI가 '제목/분류'(을)를 검색 중입니다..."):
                    st.session_state.ai_status = "..." 
                    ai_results, result_type = run_ai_search(search_query, search_mode, supabase, ai_model)
                
                if not ai_results:
                    st.info(f"ℹ️ 'AI 제목/분류' 검색 결과가 없습니다.")
                else:
                    # ★★★ [의도 수정] 결과를 DataFrame으로 변환 ★★★
                    filtered_df = pd.DataFrame(ai_results)
                    st.markdown(f"**'{search_query}'(와)과 유사한 {len(filtered_df)}건의 항목을 찾았습니다.**")

            # --- [AI] 본문 내용 검색 로직 (새 리스트) ---
            elif search_mode == "[AI] 본문 내용 검색":
                with st.spinner("🧠 AI가 '본문 전체'(을)를 검색 중입니다..."):
                    st.session_state.ai_status = "..." 
                    ai_results, result_type = run_ai_search(search_query, search_mode, supabase, ai_model)
                
                if not ai_results:
                    st.info(f"ℹ️ 'AI 본문 내용' 검색 결과가 없습니다.")
                else:
                    st.markdown(f"**'{search_query}'(와)과 유사한 {len(ai_results)}건의 본문 조각을 찾았습니다.**")
                    
                    # (본문 검색은 URL 맵이 필요함)
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
                        else:
                            st.error(f"오류: {result_filename}의 URL을 '지도(map)'에서 찾을 수 없습니다.")

            # --- 제목 검색 (키워드) 로직 (아코디언 필터링) ---
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
                    st.info("ℹ️ '제목 (키워드)' 검색 결과가 없습니다.")
                else:
                    st.markdown(f"**'{search_query}'(으)로 {len(filtered_df)}건의 항목을 찾았습니다.**")

            # --- ★★★ [의도 수정] 아코디언 렌더링 로직 (공통) ★★★ ---
            # 'filtered_df'에 내용이 있으면 (AI 제목 검색 또는 키워드 제목 검색이 성공하면)
            if not filtered_df.empty:
                # (match_map 결과는 'std_sort_key'가 없으므로 'ch_name', 'std_name'으로 그룹화)
                # (match_map 결과에는 'ch_name' 등이 없으므로, 원본 map_data와 join해야 함)
                
                # 'match_map' 결과 (ai_results)는 'id'만 있습니다. 
                # 이 'id'를 사용해 원본 'map_data'에서 전체 정보를 가져옵니다.
                if result_type == "map":
                     result_ids = filtered_df['id'].tolist()
                     # AI가 찾은 ID 순서대로 원본 map_data에서 행을 필터링하고 정렬
                     filtered_df = map_data[map_data['id'].isin(result_ids)].set_index('id').loc[result_ids].reset_index()

                # (공통 아코디언 렌더링)
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

    # --- 우측 뷰어 (col_viewer) ---
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
            st.info("좌측 '탐색' 메뉴에서 규정을 선택하거나 'AI 검색'을 실행하세요.")

# --- 관리자 패널 (st.sidebar) ---
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False

st.sidebar.title("관리자 패널")
if st.session_state.is_admin:
    st.sidebar.success("관리자 모드 활성화")
    st.sidebar.markdown("---")
    st.sidebar.subheader("앱 상태")
    st.sidebar.dataframe(map_data.head())
    st.sidebar.caption(f"총 {len(map_data)}개의 '지도(ME)' 항목 로드됨")
else:
    admin_pw = st.sidebar.text_input("관리자 암호:", type="password")
    if admin_pw:
        if admin_pw == st.secrets["app_security"]["admin_password"]:
            st.session_state.is_admin = True
            st.rerun()
        else:
            st.sidebar.error("암호가 틀렸습니다.")