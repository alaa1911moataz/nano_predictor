import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Nano-Drug Delivery Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Theme Management Initialization
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

is_dark = st.session_state.theme == "dark"

# 3. Dynamic UI Theme Styling Rules
if is_dark:
    bg = "#0A0C12"
    bg_gradient = "radial-gradient(circle at 15% 10%, rgba(124, 77, 255, 0.15) 0%, transparent 40%), radial-gradient(circle at 85% 90%, rgba(0, 224, 198, 0.12) 0%, transparent 40%), #0A0C12"
    text_color = "#E8E9ED"
    subtitle_color = "#8B92A6"
    card_bg = "linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015))"
    card_border = "rgba(124, 77, 255, 0.10)"
    card_border_hover = "rgba(124, 77, 255, 0.28)"
    input_bg = "#15171F"
    input_border = "rgba(124, 77, 255, 0.14)"
    label_color = "#9AA1B4"
    section_title_color = "#EDEDF2"
    footer_color = "#565C6E"
    shadow = "0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03)"
else:
    bg = "#F7F7FB"
    bg_gradient = "radial-gradient(circle at 15% 10%, rgba(124, 77, 255, 0.08) 0%, transparent 45%), radial-gradient(circle at 85% 90%, rgba(0, 191, 165, 0.08) 0%, transparent 45%), #F7F7FB"
    text_color = "#2B2E3B"
    subtitle_color = "#6B7280"
    card_bg = "linear-gradient(180deg, rgba(255,255,255,0.9), rgba(250,250,253,0.85))"
    card_border = "rgba(124, 77, 255, 0.10)"
    card_border_hover = "rgba(124, 77, 255, 0.30)"
    input_bg = "#FFFFFF"
    input_border = "rgba(124, 77, 255, 0.12)"
    label_color = "#6B7280"
    section_title_color = "#2B2E3B"
    footer_color = "#A3A7B5"
    shadow = "0 10px 26px rgba(124, 77, 255, 0.08), inset 0 1px 0 rgba(255,255,255,0.5)"

# 4. Inject Premium Custom CSS Styles
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Poppins', sans-serif; }}
    .stApp {{ background: {bg_gradient}; color: {text_color}; }}
    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{ padding: 2rem 1rem 1rem 1rem; max-width: 1100px; }}
    div[data-testid="column"] {{ padding: 0 0.4rem; }}
    div[data-baseweb="select"], div[data-baseweb="select"] > div, .stNumberInput, .stTextInput {{ width: 100% !important; }}
    
    .hero-wrap {{ text-align: center; margin-bottom: 2.2rem; }}
    .hero-badge {{ display: inline-block; padding: 0.35rem 1rem; border-radius: 999px; background: rgba(124, 77, 255, 0.10); border: 1px solid rgba(124, 77, 255, 0.30); color: #7C4DFF; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 1rem; }}
    .hero-title {{ font-size: 2.6rem; font-weight: 800; line-height: 1.15; background: linear-gradient(100deg, #7C4DFF 0%, #00BFA5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.6rem; }}
    .hero-subtitle {{ color: {subtitle_color}; font-size: 1.02rem; font-weight: 400; max-width: 560px; margin: 0 auto; line-height: 1.6; }}

    .glass-card {{ background: {card_bg}; backdrop-filter: blur(12px); border: 1px solid {card_border}; border-radius: 20px; padding: 1.7rem 1.8rem 0.5rem 1.8rem; margin-bottom: 1.4rem; box-shadow: {shadow}; transition: border 0.25s ease; }}
    .glass-card:hover {{ border: 1px solid {card_border_hover}; }}
    .section-label {{ font-size: 1rem; font-weight: 600; color: {section_title_color}; margin-bottom: 1.1rem; display: flex; align-items: center; gap: 0.55rem; padding-bottom: 0.8rem; border-bottom: 1px solid {card_border}; }}
    .section-icon {{ width: 32px; height: 32px; border-radius: 9px; display: inline-flex; align-items: center; justify-content: center; background: linear-gradient(135deg, rgba(124,77,255,0.20), rgba(0,191,165,0.18)); font-size: 1rem; }}

    div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {{ background-color: {input_bg} !important; border: 1px solid {input_border} !important; border-radius: 11px !important; color: {text_color} !important; font-size: 0.92rem !important; }}
    label {{ color: {label_color} !important; font-weight: 500 !important; font-size: 0.86rem !important; }}

    div.stButton > button {{ width: 100%; background: linear-gradient(100deg, #7C4DFF, #00BFA5); color: white; border: none; border-radius: 13px; padding: 0.85rem 1rem; font-size: 1.02rem; font-weight: 700; display: flex; align-items: center; justify-content: center; transition: all 0.25s ease; box-shadow: 0 6px 20px rgba(124, 77, 255, 0.25); }}
    div.stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 10px 28px rgba(124, 77, 255, 0.45); }}
    
    .result-title {{ text-align: center; font-size: 1.4rem; font-weight: 700; color: {section_title_color}; margin: 1.8rem 0 1rem 0; }}
    .metric-card-custom {{ background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.005)); border: 1px solid {card_border}; border-radius: 16px; padding: 1.5rem 1rem; text-align: center; box-shadow: {shadow}; }}
    .metric-label-custom {{ font-size: 0.88rem; color: {label_color}; font-weight: 500; margin-bottom: 0.4rem; }}
    .status-high {{ color: #00E0C6 !important; font-size: 1.8rem; font-weight: 700; text-shadow: 0 0 15px rgba(0,224,198,0.3); }}
    .status-low {{ color: #A3A7B5 !important; font-size: 1.8rem; font-weight: 700; opacity: 0.6; }}
    .footer-note {{ text-align: center; color: {footer_color}; font-size: 0.78rem; margin-top: 2.5rem; padding-bottom: 1rem; }}
</style>
""", unsafe_allow_html=True)

# 5. Header Components and Theme Toggle Button
toggle_col1, toggle_col2 = st.columns([6, 1])
with toggle_col2:
    st.button("🌙 Dark" if not is_dark else "☀️ Light", on_click=toggle_theme, use_container_width=True)

st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">🧬 AI-Powered Smart Classifier</div>
    <div class="hero-title">Nano-Particle Biodistribution<br>Classifier</div>
    <div class="hero-subtitle">
        Predict if nano-particles will achieve <b style="color:#7C4DFF;">High Tumor Retention</b> and optimal <b style="color:#00A896;">Selectivity</b>.
    </div>
</div>
""", unsafe_allow_html=True)

# 6. Optimized Data Loading Resource Cache
@st.cache_resource
def load_nano_resources():
    model = joblib.load('xgboost_nano_classifier_model.pkl')
    encoding_maps = joblib.load('nano_preprocessor.pkl')
    return model, encoding_maps

try:
    model, encoding_maps = load_nano_resources()

    # 7. Web Interactive Feature Form Construction
    # Card A: Physical Properties
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label"><span class="section-icon">⚙️</span> Physical Properties</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        size = st.number_input("Size (nm)", min_value=1.0, max_value=1000.0, value=100.0)
        zeta_mv = st.text_input("Zeta Potential (mv) — optional", value="")
    with col2:
        shape = st.selectbox("Shape", ["", "Spherical", "Rod", "Cylinder", "Discoid", "Cubical"])
        zeta_cat = st.selectbox("Zeta Category", ["", "Positive", "Negative", "Neutral"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Card B: Composition & Coating
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label"><span class="section-icon">🧪</span> Composition & Coating</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        np_class = st.selectbox("NP Class", ["", "Organic", "Inorganic"])
        has_peg = st.selectbox("Has PEG", ["", "Yes", "No"])
    with col4:
        shell_type = st.selectbox("Shell Type", ["", "PEG", "Cellulose", "Dextran", "Fuc", "HA", "HPMA", "No Stealth Effect", "PKP"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Card C: Dosing & Target Context
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label"><span class="section-icon">💉</span> Dosing & Target</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        dosage = st.number_input("Administration Dosage (mg/kg)", min_value=0.0, max_value=500.0, value=5.0)
        time_point = st.number_input("Time Point (h)", min_value=0.0, max_value=1000.0, value=24.0)
    with col6:
        tumor_site = st.selectbox("Tumor Site", ["", "Cervix", "Brain", "Breast", "Colon", "Liver", "Lungs", "Lymphoma", "Ovary", "Pancreas", "Prostate", "Sarcoma", "Skin"])
    st.markdown('</div>', unsafe_allow_html=True)

    # 8. Execution Handle on Screening Button Trigger
    predict_clicked = st.button(" Classification Screening", type="primary", use_container_width=True)

    if predict_clicked:
        # Building clean evaluation dictionary mapping
        input_dict = {
            'NP_Class': str(np_class).strip() if np_class != "" else "nan",
            'INPs_Core': "nan", 
            'Shape': str(shape).strip() if shape != "" else "nan",
            'Size (nm)': float(size),
            'Size_Category': "nan",
            'Zeta Potential (mv)': float(zeta_mv) if zeta_mv.strip() != "" else np.nan,
            'Zeta_Category': str(zeta_cat).strip() if zeta_cat != "" else "nan",
            'Organ or tissue': "nan",
            'HAS_PEG': str(has_peg).strip() if has_peg != "" else "nan",
            'Shell Type': str(shell_type).strip() if shell_type != "" else "nan",
            'Administration Dosages (mg/kg)': float(dosage),
            'Time point (h)': float(time_point),
            'Tumor Site': str(tumor_site).strip() if tumor_site != "" else "nan"
        }

        # Safe feature text mapping based on Pandas custom dictionary maps
        encoded_dict = {}
        categorical_features = ['NP_Class', 'INPs_Core', 'Shape', 'Size_Category', 'Zeta_Category', 'Organ or tissue', 'HAS_PEG', 'Shell Type', 'Tumor Site']
        
        for col, val in input_dict.items():
            if col in categorical_features:
                encoded_dict[col] = encoding_maps[col].get(val, np.nan)
            else:
                encoded_dict[col] = val

        # Reindexing to match rigorous algorithmic structural feature order
        ordered_features = [
            'NP_Class', 'INPs_Core', 'Shape', 'Size (nm)', 'Size_Category',
            'Zeta Potential (mv)', 'Zeta_Category', 'Organ or tissue', 'HAS_PEG', 
            'Shell Type', 'Administration Dosages (mg/kg)', 'Time point (h)', 'Tumor Site'
        ]
        input_df = pd.DataFrame([encoded_dict])[ordered_features]
        
        # Parallel model inference execution
        prediction = model.predict(input_df)

        # Parsing targets status definitions
        tumor_res = "High Retention" if prediction[0][0] == 1 else "Low Retention"
        selectivity_res = "High Selectivity" if prediction[0][1] == 1 else "Low Selectivity"
        tumor_class = "status-high" if prediction[0][0] == 1 else "status-low"
        selectivity_class = "status-high" if prediction[0][1] == 1 else "status-low"

        # Rendering screening glassmorphic results widgets
        st.markdown('<div class="result-title"> AI Screening Analysis</div>', unsafe_allow_html=True)
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown(f'<div class="metric-card-custom"><div class="metric-label-custom"> Tumor Retention Potential</div><div class="{tumor_class}">{tumor_res}</div></div>', unsafe_allow_html=True)
        with res_col2:
            st.markdown(f'<div class="metric-card-custom"><div class="metric-label-custom"> Targeting Selectivity Index</div><div class="{selectivity_class}">{selectivity_res}</div></div>', unsafe_allow_html=True)

        # 9. Reference Guide Context Section (Scientific Support Documentation)
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("###  Evaluation Reference Guide")
        
        ref_col1, ref_col2 = st.columns(2)
        with ref_col1:
            st.info(
                " **Tumor Retention Classification:**\n\n"
                "* **High Retention:** The particle achieves a bio-distribution efficiency **above or equal to the dataset median**. "
                "This indicates highly promising tumor accumulation suitable for therapeutic delivery.\n"
                "* **Low Retention:** The accumulation drops **below the dataset median**, suggesting the formulation might need "
                "surface optimization (e.g., adjusting PEGylation or Size)."
            )
            
        with ref_col2:
            st.success(
                " **Targeting Selectivity Index Classification:**\n\n"
                "* **High Selectivity:** The ratio of nano-particle accumulation in the tumor versus healthy organs is **optimal (>= Median)**, "
                "signaling minimized off-target side effects.\n"
                "* **Low Selectivity:** The off-target accumulation is relatively high. Consider modifying the **Shell Type** or targeting ligands "
                "to enhance specificity."
            )
            
        st.caption(
            "*Note: Threshold boundaries are dynamically computed based on the median values of the validated historical Nano-Drug Delivery Silver Dataset.*"
        )

    # 10. Footer Attribution Signature
    st.markdown('<div class="footer-note">Powered by XGBoost Classifier · Designed by Alaa Moataz</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"⚠️ An error occurred while running the model: {e}")
