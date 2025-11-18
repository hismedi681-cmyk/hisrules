import streamlit as st
import pandas as pd
import numpy as np
import re
from supabase import create_client, Client, ClientOptions
from httpx import Timeout
import httpx 
from sentence_transformers import SentenceTransformer
from streamlit_pdf_viewer import pdf_viewer 

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

@st.cache_data(ttl=3600)
def get_pdf_bytes(url: str):
    """ PDF URL을 받아 바이너리(bytes) 데이터로 반환합니다. """
    try:
        if url.startswith("http://"): url = url.replace("http://", "https://")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = httpx.get(url, headers=headers, timeout=15.0)
        
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"❌ PDF 다운로드 실패: HTTP {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ PDF 다운로드 오류: {e}")
        return None


# ★★★ [NEW] JavaScript 스크롤 헬퍼 함수 정의 (픽셀 점프) ★★★
def js_scroll_to_page_relative(scroll_index):
    """ PDF 뷰어의 내부 스크롤 컨테이너를 찾아서 상대적 인덱스 위치로 이동시키는 JS 코드를 삽입합니다. """
    
    js_code = f"""
    <script>
        function attemptScroll() {{
            const viewer = document.querySelector('.streamlit-container .st-emotion-base:last-child');
            
            if (viewer) {{
                const scrollableContainer = viewer.querySelector('.react-pdf__Document'); 
                const firstPage = viewer.querySelector('.react-pdf__Page'); // 첫 번째 페이지 요소를 찾습니다.

                if (scrollableContainer && firstPage) {{
                    const pageHeight = firstPage.offsetHeight; // 첫 페이지의 픽셀 높이를 측정합니다.
                    // 스크롤 위치 = 인덱스 * 측정된 높이
                    const scrollAmount = {scroll_index} * pageHeight;
                    
                    scrollableContainer.scrollTop = scrollAmount;
                    console.log('PDF Scrolled to index: {scroll_index}, Height: ' + pageHeight);
                }} else {{
                    // 컨테이너/페이지가 아직 로드되지 않았으면 0.1초 뒤에 재시도
                    setTimeout(attemptScroll, 100); 
                }}
            }}
        }}

        // 페이지가 로드된 후 스크롤 시도 (0.5초 대기)
        setTimeout(attemptScroll, 500); 
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)


# ★★★ [NEW] 최종 안정화 뷰어 함수: 듀얼 모드 (전체/맥락) ★★★
def render_pdf_viewer_mode(pdf_url: str, page: int = 1):
    """ 
    [듀얼 모드] target_page에 따라 로드 방식을 결정하고, AI 검색 시 픽셀 점프를 시도합니다.
    """
    # 1. 입력 페이지 번호를 확실하게 int로 변환 (TypeError 방지)
    target_page = int(page) 
    
    if not pdf_url:
        st.info("규정을 선택하세요.")
        return

    # 2. 로딩 모드 결정 및 페이지 계산
    if target_page == 1:
        # 일반 규정 목록 또는 합본 PDF 클릭 시: 전체 로드 시도
        pages_to_load = [] # 빈 리스트는 전체 로드 효과를 냅니다.
        spinner_text = "📄 전체 문서를 로딩 중..."
        jump_index = 0 # 점프 불필요
    else:
        # AI 검색 결과 클릭 시: 맥락 창 로드 (±20 페이지)
        context_range = 20 
        start = int(max(1, target_page - context_range))
        end = int(target_page + context_range)
        
        pages_to_load = list(range(start, end + 1))
        
        # ★★★ 점프 인덱스 계산: 타겟 페이지가 로드된 페이지 리스트 내에서 몇 번째 인덱스인지 계산 ★★★
        # (예: start=30, target_page=50. 인덱스 = 50 - 30 = 20)
        jump_index = target_page - start
        spinner_text = f"📄 AI 검색 문맥 창 ({start}p ~ {end}p) 로딩 및 {target_page}p로 점프 중..."

    # 3. PDF 렌더링
    with st.spinner(spinner_text):
        pdf_data = get_pdf_bytes(pdf_url)
    
    if pdf_data:
        pdf_viewer(
            input=pdf_data, 
            width=700, 
            height=1000,
            pages_to_render=pages_to_load
        )
        
        # 4. 렌더링 성공 후, AI 검색 모드일 때만 JS 스크롤 실행
        if target_page > 1 and jump_index > 0:
            js_scroll_to_page_relative(jump_index)
            
    else:
        st.error("❌ PDF 문서를 로딩할 수 없습니다.")


def set_pdf_url(url: str, page: int):
    st.session_state.current_pdf_url = url
    st.session_state.current_pdf_page = page
    st.session_state.view_mode = "preview" 

# --- 4. UI 구성 (메인 루프) ---

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
        render_pdf_viewer_mode(st.session_state.current_pdf_url, st.session_state.current_pdf_page)

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
            # ★★★ 함수 호출
            render_pdf_viewer_mode(st.session_state.current_pdf_url, st.session_state.current_pdf_page)
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




