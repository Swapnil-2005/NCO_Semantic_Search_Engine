import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import speech_recognition as sr
from googletrans import Translator
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Load Models and Data ---
@st.cache_resource # Cache the model so it's loaded only once
def load_sentence_transformer():
    """Loads the Sentence Transformer model."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Failed to load sentence transformer model: {e}")
        return None

@st.cache_resource
def load_faiss_index():
    """Loads the Faiss vector index."""
    try:
        index = faiss.read_index("nco_vector.index")
        return index
    except Exception as e:
        st.error(f"Failed to load vector index 'nco_vector.index': {e}")
        return None

@st.cache_data
def load_nco_data():
    """Loads the NCO data from the pickle file."""
    try:
        with open('nco_metadata.pkl', 'rb') as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)
        for col in ['Occupation Title', 'Description', 'processed_text']:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('None')
        return df
    except FileNotFoundError:
        st.error("Error: 'nco_metadata.pkl' not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading 'nco_metadata.pkl': {e}")
        return pd.DataFrame()

# --- Translation Function ---
@st.cache_data
def translate_text(text_to_translate, dest_lang, src_lang=None):
    """Translates text to the destination language, with caching."""
    if not text_to_translate or dest_lang == src_lang:
        return text_to_translate
    try:
        translator = Translator()
        if src_lang:
            translated = translator.translate(text_to_translate, dest=dest_lang, src=src_lang)
        else:
            translated = translator.translate(text_to_translate, dest=dest_lang)
        
        if translated and translated.text:
            return translated.text
        else:
            st.warning(f"Translation returned empty for '{text_to_translate}'. Using original text.")
            return text_to_translate
    except Exception as e:
        st.warning(f"Translation error for '{text_to_translate}': {e}. Using original text.")
        return text_to_translate # Return original text on error

# --- Database Setup ---
def init_db():
    """Initializes the SQLite database."""
    with sqlite3.connect('app_data.db') as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS audit_trail (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                user TEXT,
                action TEXT,
                details TEXT
            )
        ''')

def add_audit_log(user, action, details):
    """Adds a new entry to the audit trail."""
    with sqlite3.connect('app_data.db') as conn:
        c = conn.cursor()
        c.execute("INSERT INTO audit_trail (timestamp, user, action, details) VALUES (?, ?, ?, ?)",
                  (datetime.now(), user, action, details))

def get_trending_searches():
    """Retrieves and counts search terms."""
    with sqlite3.connect('app_data.db') as conn:
        try:
            query = """
                SELECT details AS "Search Term", COUNT(details) AS "Count"
                FROM audit_trail WHERE action = 'search' AND details != ''
                GROUP BY details ORDER BY "Count" DESC
            """
            return pd.read_sql_query(query, conn)
        except Exception:
            return pd.DataFrame(columns=["Search Term", "Count"])

def get_audit_trail():
    """Retrieves all entries from the audit trail."""
    with sqlite3.connect('app_data.db') as conn:
        return pd.read_sql_query("SELECT * FROM audit_trail", conn)

def get_daily_usage_data(start_date, end_date):
    """Retrieves daily search usage within a date range."""
    with sqlite3.connect('app_data.db') as conn:
        try:
            query = "SELECT timestamp FROM audit_trail WHERE action = 'search' AND date(timestamp) BETWEEN ? AND ?"
            df = pd.read_sql_query(query, conn, params=(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_counts = df.groupby('date').size().reset_index(name='Searches').set_index('date')
            all_days = pd.date_range(start=start_date, end=end_date, freq='D').date
            return daily_counts.reindex(all_days, fill_value=0)
        except Exception:
            return pd.DataFrame()

def delete_search_history(search_term):
    """Deletes all occurrences of a specific search term from the audit trail."""
    with sqlite3.connect('app_data.db') as conn:
        c = conn.cursor()
        c.execute("DELETE FROM audit_trail WHERE action = 'search' AND details = ?", (search_term,))

# --- Multilingual Support ---
translations = {
    "en": {
        "title": "NCO Search Application", "nav_search": "Search", "nav_dashboard": "Dashboard", "nav_audit": "Audit Trail",
        "search_placeholder": "Search occupation typing a keyword...", "dashboard_header": "Trending Searches Dashboard",
        "usage_graph_header": "Daily Search Usage", "date_range_label": "Select Date Range:", "audit_trail_header": "Full Audit Trail",
        "download_csv": "Download as CSV", "language_select": "Select Language", "voice_input_button": "🎤",
        "voice_prompt": "Listening...", "search_results_header": "Occupation List", "nsqf_level_label": "NSQF Level",
        "isco_group_label": "ISCO Group", "all_option": "All", "found_results_for": "Found {count} results for: {query}",
        "select_occupation_prompt": "Select an Occupation Title to View Details:", "translated_title_header": "Translated Title",
        "relevance_score_header": "Relevance Score (%)", "searching_for_translation": "Searching for English translation: '{query}'",
        "delete_history_header": "Delete Search History", "select_term_to_delete_label": "Select a search term to delete:",
        "delete_button_label": "Delete", "delete_success_message": "Deleted all entries for '{term}'."
    },
    "hi": {
        "title": "एनसीओ खोज एप्लीकेशन", "nav_search": "खोज", "nav_dashboard": "डैशबोर्ड", "nav_audit": "ऑडिट ट्रायल",
        "search_placeholder": "कीवर्ड टाइप करके पेशा खोजें...", "dashboard_header": "ट्रेंडिंग खोज डैशबोर्ड",
        "usage_graph_header": "दैनिक खोज उपयोग", "date_range_label": "दिनांक सीमा चुनें:", "audit_trail_header": "पूर्ण ऑडिट ट्रायल",
        "download_csv": "सीएसवी के रूप में डाउनलोड करें", "language_select": "भाषा चुनें", "voice_input_button": "🎤",
        "voice_prompt": "सुन रहा हूँ...", "search_results_header": "पेशा सूची", "nsqf_level_label": "एनएसक्यूएफ स्तर",
        "isco_group_label": "आईएससीओ समूह", "all_option": "सभी", "found_results_for": "{query} के लिए {count} परिणाम मिले:",
        "select_occupation_prompt": "विवरण देखने के लिए एक पेशा शीर्षक चुनें:", "translated_title_header": "अनुवादित शीर्षक",
        "relevance_score_header": "प्रासंगिकता स्कोर (%)", "searching_for_translation": "अंग्रेजी अनुवाद के लिए खोजा जा रहा है: '{query}'",
        "delete_history_header": "खोज इतिहास हटाएं", "select_term_to_delete_label": "हटाने के लिए एक खोज शब्द चुनें:",
        "delete_button_label": "हटाएं", "delete_success_message": "'{term}' के लिए सभी प्रविष्टियाँ हटा दी गईं।"
    },
    # Add full translations for other languages here...
}

# --- Voice Input ---
def recognize_speech(lang_code='en-IN'):
    """Recognizes speech from the microphone."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info(translations.get(st.session_state.lang, translations['en'])["voice_prompt"])
        audio = r.listen(source)
    try:
        return r.recognize_google(audio, language=lang_code)
    except (sr.UnknownValueError, sr.RequestError):
        return ""

# --- Main App ---
def main():
    """Main function to run the Streamlit app."""
    init_db()
    
    st.session_state.setdefault('lang', 'en')
    st.session_state.setdefault('search_text', "")

    # --- Language Selection ---
    st.sidebar.title("Settings")
    lang_options = {
        "en": "English", "hi": "हिंदी", "bn": "বাংলা", "ta": "தமிழ்", "te": "తెలుగు",
        "mr": "मराठी", "gu": "ગુજરાતી", "kn": "ಕನ್ನಡ", "ml": "മലയാളം", "pa": "ਪੰਜਾਬੀ"
    }
    lang_code = st.sidebar.selectbox("Language", options=list(lang_options.keys()), format_func=lang_options.get)
    st.session_state.lang = lang_code
    t = translations.get(st.session_state.lang, translations['en'])

    st.title(t["title"])

    # --- Page Navigation using Tabs ---
    tab1, tab2, tab3 = st.tabs([t["nav_search"], t["nav_dashboard"], t["nav_audit"]])

    with tab1:
        run_search_page(t)
    
    with tab2:
        run_dashboard_page(t)

    with tab3:
        run_audit_page(t)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        'This project is made by Team "Blank Coders" ft. "Arka", "Debjit", "Rishav", "Saptarshi", "Swapnil".'
    )


def run_search_page(t):
    """Renders the search page."""
    nco_df = load_nco_data()
    model = load_sentence_transformer()
    index = load_faiss_index()

    if nco_df.empty or model is None or index is None:
        st.warning("Search is unavailable. Please check file paths and model loading.")
        return

    # --- Search Filters ---
    col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
    with col1:
        search_query = st.text_input("", key="search_text_input", value=st.session_state.search_text, placeholder=t["search_placeholder"], label_visibility="collapsed")
    with col2:
        if st.button(t["voice_input_button"]):
            lang_map = {'en': 'en-IN', 'hi': 'hi-IN', 'bn': 'bn-IN', 'ta': 'ta-IN', 'te': 'te-IN', 'mr': 'mr-IN', 'gu': 'gu-IN', 'kn': 'kn-IN', 'ml': 'ml-IN', 'pa': 'pa-IN'}
            voice_query = recognize_speech(lang_map.get(st.session_state.lang, 'en-IN'))
            if voice_query:
                st.session_state.search_text = voice_query
                st.rerun()

    nsqf_level = t["all_option"]
    isco_group = t["all_option"]
    with col3:
        if 'NSQF Level' in nco_df.columns:
            nsqf_options = [t["all_option"]] + sorted(nco_df['NSQF Level'].astype(str).unique().tolist())
            nsqf_level = st.selectbox(t["nsqf_level_label"], nsqf_options, label_visibility="collapsed")
    with col4:
        if 'ISCO Group' in nco_df.columns:
            isco_options = [t["all_option"]] + sorted(nco_df['ISCO Group'].astype(str).unique().tolist())
            isco_group = st.selectbox(t["isco_group_label"], isco_options, label_visibility="collapsed")

    # --- Filtering and Searching ---
    filtered_df = nco_df.copy()
    if nsqf_level != t["all_option"] and 'NSQF Level' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['NSQF Level'].astype(str) == nsqf_level]
    if isco_group != t["all_option"] and 'ISCO Group' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['ISCO Group'].astype(str) == isco_group]

    if search_query:
        add_audit_log("user", "search", search_query)
        
        if st.session_state.lang != 'en':
            english_query = translate_text(search_query, 'en', src_lang=st.session_state.lang)
            st.info(t['searching_for_translation'].format(query=english_query))
        else:
            english_query = search_query

        candidate_indices = filtered_df.index.to_numpy()
        if len(candidate_indices) > 0:
            candidate_vectors = np.array([index.reconstruct(int(i)) for i in candidate_indices]).astype('float32')
            faiss.normalize_L2(candidate_vectors)
            temp_index = faiss.IndexFlatIP(candidate_vectors.shape[1])
            temp_index.add(candidate_vectors)
            
            query_vector = model.encode([english_query]).astype('float32')
            faiss.normalize_L2(query_vector)
            
            k = min(len(candidate_indices), 50)
            similarities, temp_indices = temp_index.search(query_vector, k)
            
            original_indices = [candidate_indices[i] for i in temp_indices[0]]
            results_df = nco_df.loc[original_indices].copy()
            
            results_df[t['relevance_score_header']] = (similarities[0] * 100).astype(int)
            results_df = results_df.sort_values(by=t['relevance_score_header'], ascending=False)
        else:
            results_df = pd.DataFrame()
    else:
        results_df = filtered_df

    # --- Display Results ---
    if not results_df.empty:
        display_results(results_df, t, search_query)
    else:
        st.warning("No results found.")

def display_results(results_df, t, search_query):
    """Renders the search results table and dropdown."""
    st.markdown(f"**{t['found_results_for'].format(count=len(results_df), query=search_query if search_query else 'all occupations')}**")
    
    display_df = results_df.copy()

    if search_query and st.session_state.lang != 'en':
        display_df[t['translated_title_header']] = display_df['Occupation Title'].apply(lambda x: translate_text(x, st.session_state.lang))
        occupation_options = [t["select_occupation_prompt"]] + display_df[t['translated_title_header']].tolist()
    else:
        occupation_options = [t["select_occupation_prompt"]] + display_df['Occupation Title'].tolist()
    
    selected_occupation = st.selectbox("", occupation_options, label_visibility="collapsed")
    st.header(t["search_results_header"])

    if selected_occupation != t["select_occupation_prompt"]:
        key_col = t['translated_title_header'] if (search_query and st.session_state.lang != 'en') else 'Occupation Title'
        display_table = display_df[display_df[key_col] == selected_occupation]
    else:
        display_table = display_df
    
    # Reorder columns for display
    display_cols = display_table.columns.tolist()
    new_order = []
    if t['relevance_score_header'] in display_cols: new_order.append(t['relevance_score_header'])
    if t['translated_title_header'] in display_cols: new_order.append(t['translated_title_header'])
    if 'Occupation Title' in display_cols: new_order.append('Occupation Title')
    
    remaining_cols = [col for col in display_cols if col not in new_order and col != 'processed_text']
    st.dataframe(display_table[new_order + remaining_cols])

def run_dashboard_page(t):
    """Renders the dashboard page."""
    st.header(t["dashboard_header"])
    trending_df = get_trending_searches()
    if not trending_df.empty:
        st.dataframe(trending_df)
        csv = trending_df.to_csv(index=False).encode('utf-8')
        st.download_button(label=t["download_csv"], data=csv, file_name='trending_searches.csv', mime='text/csv')
        
        st.markdown("---")
        st.subheader(t["delete_history_header"])
        search_terms_to_delete = trending_df["Search Term"].tolist()
        term_to_delete = st.selectbox(t["select_term_to_delete_label"], search_terms_to_delete)
        
        if st.button(t["delete_button_label"]):
            if term_to_delete:
                delete_search_history(term_to_delete)
                st.success(t["delete_success_message"].format(term=term_to_delete))
                st.rerun()

    else:
        st.info("No search data available yet to show trends.")

    st.markdown("---")
    st.header(t["usage_graph_header"])
    today = datetime.now().date()
    default_start = today - timedelta(days=6)
    date_range = st.date_input(t["date_range_label"], (default_start, today), max_value=today)

    if len(date_range) == 2:
        start_date, end_date = date_range
        usage_df = get_daily_usage_data(start_date, end_date)
        if not usage_df.empty:
            st.bar_chart(usage_df)
        else:
            st.info("No usage data available for the selected period.")
    else:
        st.warning("Please select a valid date range.")

def run_audit_page(t):
    """Renders the audit trail page."""
    st.header(t["audit_trail_header"])
    audit_df = get_audit_trail()
    if not audit_df.empty:
        st.dataframe(audit_df)
    else:
        st.info("No audit trail data available.")

if __name__ == '__main__':
    main()
