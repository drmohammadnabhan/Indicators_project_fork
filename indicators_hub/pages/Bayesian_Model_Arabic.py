import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

# Page Configuration
st.set_page_config(layout="wide", page_title="Adaptive Bayesian Estimation Proposal")

# --- Global variable for language (simplified) ---
# We'll set this from the sidebar. Default to English.
# This replaces the complex LANGUAGES dictionary for now to ensure parsing.
current_lang = "en"

# --- Helper Functions for Plotting (no text directly from global dict) ---
def plot_beta_distribution(alpha, beta, label_en, label_ar, ax, is_discounted_label=False, discount_factor_val=None):
    """Plots a Beta distribution with internationalized and formatted label."""
    base_label = label_ar if current_lang == "ar" else label_en
    
    formatted_label = ""
    if is_discounted_label:
        formatted_label = f"{base_label} (δ={discount_factor_val:.1f}, α={alpha:.2f}, β={beta:.2f})"
    else:
        formatted_label = f"{base_label} (α={alpha:.2f}, β={beta:.2f})"

    x_vals = np.linspace(0.001, 0.999, 500)
    y_vals = stats.beta.pdf(x_vals, alpha, beta)
    ax.plot(x_vals, y_vals, label=formatted_label)
    ax.fill_between(x_vals, y_vals, alpha=0.2)
    ax.set_ylim(bottom=0)

def update_beta_parameters(prior_alpha, prior_beta, successes, failures):
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + failures
    return posterior_alpha, posterior_beta

def get_credible_interval(alpha, beta, conf_level=0.95):
    if alpha <=0 or beta <=0:
        return (0.0,0.0)
    try:
        return stats.beta.interval(conf_level, alpha, beta)
    except ValueError:
        return (0.0,0.0)

# --- Rendering Helpers (Simplified) ---
def display_header(english_text, arabic_text, level=1):
    text_to_display = arabic_text if current_lang == "ar" else english_text
    header_tag = f"h{level}"
    if current_lang == "ar":
        st.markdown(f"<{header_tag} dir='rtl' style='text-align: right;'>{text_to_display}</{header_tag}>", unsafe_allow_html=True)
    else:
        st.markdown(f"<{header_tag}>{text_to_display}</{header_tag}>", unsafe_allow_html=True)

def display_markdown(english_text, arabic_text, unsafe_html=True):
    text_to_display = arabic_text if current_lang == "ar" else english_text
    if current_lang == "ar":
        # Ensure the div wrapper if not already present (though we try to add it here)
        if isinstance(text_to_display, str) and not text_to_display.strip().startswith("<div dir='rtl'>"):
             text_to_display = f"<div dir='rtl'>{text_to_display}</div>"
        st.markdown(text_to_display, unsafe_allow_html=unsafe_html)
    else:
        st.markdown(text_to_display, unsafe_allow_html=unsafe_html)

def display_write(english_text, arabic_text):
    text_to_display = arabic_text if current_lang == "ar" else english_text
    if current_lang == "ar":
        st.markdown(f"<div dir='rtl' style='text-align: right;'>{text_to_display}</div>", unsafe_allow_html=True)
    else:
        st.write(text_to_display)

# --- Proposal Content Functions (using direct strings or simple bilingual variables) ---

def introduction_objectives_content():
    display_header("1. Introduction & Objectives", "١. مقدمة وأهداف", level=2)
    
    en_intro = """This proposal outlines an **Adaptive Bayesian Estimation framework** designed to enhance the process of gathering and analyzing survey data for Hajj pilgrim satisfaction and the assessment of services provided by various companies.
The current practice of developing satisfaction metrics month over month faces complexities, such as delays in pilgrim arrivals or non-uniformity across different months, making it difficult to consistently achieve high-confidence and low-error confidence intervals for key indicators. This proposal aims to introduce a more dynamic, efficient, and robust methodology."""
    ar_intro = """يحدد هذا المقترح إطارًا لـ **تقدير Bayesian التكيفي** مصممًا لتعزيز عملية جمع وتحليل بيانات الاستطلاع الخاصة برضا حجاج بيت الله الحرام وتقييم الخدمات المقدمة من مختلف الشركات.
تواجه الممارسة الحالية المتمثلة في تطوير مقاييس الرضا شهرًا بعد شهر تعقيدات، مثل التأخير في وصول الحجاج أو عدم الانتظام عبر الأشهر المختلفة، مما يجعل من الصعب تحقيق فترات ثقة عالية الدقة ومنخفضة الخطأ للمؤشرات الرئيسية بشكل مستمر. يهدف هذا المقترح إلى تقديم منهجية أكثر ديناميكية وكفاءة وقوة."""
    display_markdown(en_intro, ar_intro)

    display_header("1.1. Primary Objectives", "١.١. الأهداف الأساسية", level=3)
    en_objectives = """
* **Achieve Desired Precision Efficiently:** To obtain satisfaction metrics and service provider assessments with pre-defined levels of precision (i.e., narrow credible intervals at a specific confidence level) using optimized sample sizes.
* **Dynamic Sampling Adjustments:** To iteratively adjust sampling efforts based on accumulating evidence.
* **Timely and Reliable Estimates:** To provide decision-makers with more timely and statistically robust estimates.
* **Incorporate Prior Knowledge:** To formally integrate knowledge from previous survey waves or expert opinions.
* **Adapt to Changing Conditions:** To develop a system that can adapt to changes in satisfaction levels or service provider performance.
* **Enhanced Subgroup Analysis:** To facilitate more reliable analysis of specific pilgrim subgroups.
    """
    ar_objectives = """
* **تحقيق الدقة المطلوبة بكفاءة:** الحصول على مقاييس الرضا وتقييمات مقدمي الخدمات بمستويات دقة محددة مسبقًا.
* **تعديلات أخذ العينات الديناميكية:** تعديل جهود أخذ العينات بشكل متكرر بناءً على الأدلة المتراكمة.
* **تقديرات موثوقة وفي الوقت المناسب:** تزويد صانعي القرار بتقديرات أكثر موثوقية.
* **دمج المعرفة المسبقة:** دمج المعرفة من موجات الاستطلاع السابقة أو آراء الخبراء.
* **التكيف مع الظروف المتغيرة:** تطوير نظام يمكنه التكيف مع التغيرات في مستويات الرضا.
* **تحليل محسن للمجموعات الفرعية:** تسهيل تحليل أكثر موثوقية لمجموعات الحجاج الفرعية.
    """
    display_markdown(en_objectives, ar_objectives)

def challenges_addressed_content():
    display_header("2. Challenges Addressed", "٢. التحديات المعالجة", level=2)
    # ... (Content for this section using display_markdown/display_header with direct Eng/Ar strings)
    en_text = "Details of challenges addressed..."
    ar_text = "تفاصيل التحديات التي تم تناولها..."
    display_markdown(en_text, ar_text)

    display_header("Why Not Just Combine All Data (Data Lumping)?", "لماذا لا نكتفي بدمج جميع البيانات (تجميع البيانات)؟", level=3)
    en_lumping = "Explanation about data lumping issues..."
    ar_lumping = "شرح حول مشاكل تجميع البيانات..."
    display_markdown(en_lumping, ar_lumping)


def bayesian_adaptive_methodology_content():
    display_header("3. Core Methodology: Bayesian Adaptive Estimation", "٣. المنهجية الأساسية: تقدير Bayesian التكيفي", level=2)
    # ... (Content for this section)
    display_markdown("Details on Bayesian methodology...", "تفاصيل منهجية Bayesian...")


def implementation_roadmap_content():
    display_header("4. Implementation Roadmap", "٤. خارطة طريق التنفيذ", level=2)
    # ... (Content for this section)
    display_markdown("Roadmap details...", "تفاصيل خارطة الطريق...")
    # For table, you might need to construct it more manually or use a simpler format
    data_en = {'Phase': ['Phase 1'], 'Step': ['Step 1'], 'Description': ['Desc 1']}
    data_ar = {'المرحلة': ['المرحلة ١'], 'الخطوة': ['الخطوة ١'], 'الوصف': ['وصف ١']}
    df_display = pd.DataFrame(data_ar if current_lang == "ar" else data_en)
    if current_lang == 'ar':
        st.markdown(f"<div dir='rtl'>{df_display.to_html(index=False)}</div>", unsafe_allow_html=True)
    else:
        st.dataframe(df_display)


def note_to_practitioners_content():
    display_header("5. Note to Practitioners", "٥. ملاحظة للممارسين", level=2)
    # ... (Content for this section)
    display_markdown("Notes for practitioners...", "ملاحظات للممارسين...")

def interactive_illustration_content():
    display_header("6. Interactive Illustration", "٦. توضيح تفاعلي", level=2)
    display_markdown("Interactive Beta-Binomial model illustration.", "توضيح تفاعلي لنموذج Beta-Binomial.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        display_header("Prior Beliefs", "المعتقدات المسبقة", level=3)
        prior_alpha = st.slider("Prior Alpha (α₀)" if current_lang=="en" else "ألفا المسبقة (α₀)", 0.1, 50.0, 1.0, 0.1, key="ia_prior_a")
        prior_beta = st.slider("Prior Beta (β₀)" if current_lang=="en" else "بيتا المسبقة (β₀)", 0.1, 50.0, 1.0, 0.1, key="ia_prior_b")
        # ... (rest of interactive content, using bilingual labels directly)

    with col2:
        display_header("New Survey Data", "بيانات الاستطلاع الجديدة", level=3)
        num_surveys = st.slider("Number of New Surveys (n)" if current_lang=="en" else "عدد الاستطلاعات الجديدة (n)", 1, 500, 50, 1, key="ia_n")
        num_satisfied = st.slider("Number Satisfied (k)" if current_lang=="en" else "عدد الراضين (k)", 0, num_surveys, int(num_surveys/2), 1, key="ia_k")

    # Simplified plotting call (labels defined in plot_beta_distribution)
    fig, ax = plt.subplots()
    if prior_alpha > 0 and prior_beta > 0:
        plot_beta_distribution(prior_alpha, prior_beta, "Prior", "المسبق", ax)
    
    # Dummy posterior for example
    posterior_alpha, posterior_beta = update_beta_parameters(prior_alpha, prior_beta, num_satisfied, num_surveys - num_satisfied)
    if posterior_alpha > 0 and posterior_beta > 0:
        plot_beta_distribution(posterior_alpha, posterior_beta, "Posterior", "اللاحق", ax)
    
    ax.set_title("Prior and Posterior Distributions" if current_lang=="en" else "التوزيعات المسبقة واللاحقة")
    ax.set_xlabel("Satisfaction Rate (θ)" if current_lang=="en" else "معدل الرضا (θ)")
    ax.set_ylabel("Density" if current_lang=="en" else "الكثافة")
    if (prior_alpha > 0 and prior_beta > 0) or (posterior_alpha > 0 and posterior_beta > 0):
        ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def conclusion_content():
    display_header("7. Conclusion", "٧. الخلاصة", level=2)
    # ... (Content for this section)
    display_markdown("Conclusion of the proposal...", "خلاصة المقترح...")


# --- Streamlit App Structure ---

# Language Selection
selected_lang_option = st.sidebar.selectbox(
    label="Select Language / اختر اللغة",
    options=["English", "العربية"],
    index=0
)
current_lang = "ar" if selected_lang_option == "العربية" else "en"


# App Title
app_title_en = "Proposal: Adaptive Bayesian Estimation"
app_title_ar = "مقترح: تقدير Bayesian التكيفي"
display_header(app_title_en, app_title_ar, level=1)


# Sidebar Navigation
PAGES_SETUP = {
    "Introduction & Objectives": {"en": "1. Introduction & Objectives", "ar": "١. مقدمة وأهداف", "func": introduction_objectives_content},
    "Challenges Addressed": {"en": "2. Challenges Addressed", "ar": "٢. التحديات المعالجة", "func": challenges_addressed_content},
    "Core Methodology": {"en": "3. Core Methodology", "ar": "٣. المنهجية الأساسية", "func": bayesian_adaptive_methodology_content},
    "Implementation Roadmap": {"en": "4. Implementation Roadmap", "ar": "٤. خارطة طريق التنفيذ", "func": implementation_roadmap_content},
    "Note to Practitioners": {"en": "5. Note to Practitioners", "ar": "٥. ملاحظة للممارسين", "func": note_to_practitioners_content},
    "Interactive Illustration": {"en": "6. Interactive Illustration", "ar": "٦. توضيح تفاعلي", "func": interactive_illustration_content},
    "Conclusion": {"en": "7. Conclusion", "ar": "٧. الخلاصة", "func": conclusion_content}
}

sidebar_display_options = [PAGES_SETUP[key][current_lang] for key in PAGES_SETUP]
sidebar_keys = list(PAGES_SETUP.keys())

st.sidebar.title("Proposal Sections" if current_lang == "en" else "أقسام المقترح")
selected_page_display_name = st.sidebar.radio(
    f"navigation_radio_{current_lang}", 
    sidebar_display_options
)

selected_page_key = None
for key in sidebar_keys:
    if PAGES_SETUP[key][current_lang] == selected_page_display_name:
        selected_page_key = key
        break

if selected_page_key:
    page_function_to_call = PAGES_SETUP[selected_page_key]["func"]
    page_function_to_call()
else:
    st.error("Page loading error." if current_lang == "en" else "خطأ في تحميل الصفحة.")

st.sidebar.markdown("---")
st.sidebar.info("Developed by Dr. Mohammad Nabhan." if current_lang == "en" else "تطوير د. محمد نبهان.")
