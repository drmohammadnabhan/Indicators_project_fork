import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

# Page Configuration
st.set_page_config(layout="wide", page_title="Adaptive Bayesian Estimation Proposal")

# --- Global variable for language ---
current_lang = "en" # Default

# --- Helper Functions for Plotting ---
def plot_beta_distribution(alpha, beta, label_en, label_ar, ax, is_discounted_label=False, discount_factor_val=None):
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
    return prior_alpha + successes, prior_beta + failures

def get_credible_interval(alpha, beta, conf_level=0.95):
    if alpha <= 0 or beta <= 0: return (0.0, 0.0)
    try: return stats.beta.interval(conf_level, alpha, beta)
    except ValueError: return (0.0, 0.0)

# --- Rendering Helpers ---
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
        # Ensure the div wrapper. Using st.markdown for the inner content allows it to parse markdown.
        st.markdown(f"<div dir='rtl'>{text_to_display}</div>", unsafe_allow_html=unsafe_html)
    else:
        st.markdown(text_to_display, unsafe_allow_html=unsafe_html) # unsafe_html might not be needed for Eng if no HTML

def display_write(english_text, arabic_text): # Typically for simple text, not markdown
    text_to_display = arabic_text if current_lang == "ar" else english_text
    if current_lang == "ar":
        st.markdown(f"<div dir='rtl' style='text-align: right;'>{text_to_display}</div>", unsafe_allow_html=True)
    else:
        st.write(text_to_display) # st.write can also interpret some markdown

# --- Proposal Content Functions ---

def introduction_objectives_content():
    display_header("1. Introduction & Objectives", "١. مقدمة وأهداف", level=2)
    
    en_intro = """This proposal outlines an **Adaptive Bayesian Estimation framework**.
It is designed to enhance the process of gathering and analyzing survey data for Hajj pilgrim satisfaction.
Here is **another bolded phrase**. And *italic text*.
    """
    ar_intro = """يحدد هذا المقترح إطارًا لـ **تقدير Bayesian التكيفي**.
إنه مصمم لتعزيز عملية جمع وتحليل بيانات الاستطلاع الخاصة برضا حجاج بيت الله الحرام.
هذه **عبارة أخرى بخط عريض**. وهذا *نص مائل*.
    """
    display_markdown(en_intro, ar_intro)

    display_header("1.1. Primary Objectives", "١.١. الأهداف الأساسية", level=3)
    en_objectives = """
* **Achieve Desired Precision Efficiently:** To obtain metrics with pre-defined precision.
* **Dynamic Sampling Adjustments:** To iteratively adjust sampling efforts.
* **Another Objective:** This is a **key goal**.
    """
    ar_objectives = """
* **تحقيق الدقة المطلوبة بكفاءة:** الحصول على مقاييس بدقة محددة مسبقًا.
* **تعديلات أخذ العينات الديناميكية:** تعديل جهود أخذ العينات بشكل متكرر.
* **هدف آخر:** هذا **هدف رئيسي**.
    """
    display_markdown(en_objectives, ar_objectives)

def challenges_addressed_content():
    display_header("2. Challenges Addressed", "٢. التحديات المعالجة", level=2)
    en_text = "Details of **challenges** addressed by this **important** methodology."
    ar_text = "تفاصيل **التحديات** التي تم تناولها بواسطة هذه المنهجية **الهامة**."
    display_markdown(en_text, ar_text)

    display_header("Why Not Just Combine All Data (Data Lumping)?", "لماذا لا نكتفي بدمج جميع البيانات (تجميع البيانات)؟", level=3)
    en_lumping = "Explanation about **data lumping** issues. This is **critical** to understand."
    ar_lumping = "شرح حول مشاكل **تجميع البيانات**. من **الأهمية بمكان** فهم هذا."
    display_markdown(en_lumping, ar_lumping)


def bayesian_adaptive_methodology_content():
    display_header("3. Core Methodology: Bayesian Adaptive Estimation", "٣. المنهجية الأساسية: تقدير Bayesian التكيفي", level=2)
    en_method = r"""
The core of Bayesian inference involves:
* **Prior Distribution ($P(\theta)$):** Initial belief.
* **Likelihood ($P(D|\theta)$):** How data $D$ supports $\theta$.
* **Posterior Distribution ($P(\theta|D)$):** Updated belief, calculated via Bayes' Theorem:
    $$ P(\theta|D) \propto P(D|\theta) \times P(\theta) $$
This posterior is **fundamental** for decision making.
    """
    ar_method = r"""
جوهر استدلال Bayesian يتضمن:
* **التوزيع المسبق ($P(\theta)$):** الاعتقاد الأولي.
* **دالة الإمكان ($P(D|\theta)$):** كيف تدعم البيانات $D$ المعلمة $\theta$.
* **التوزيع اللاحق ($P(\theta|D)$):** الاعتقاد المحدث، المحسوب عبر نظرية Bayes:
    $$ P(\theta|D) \propto P(D|\theta) \times P(\theta) $$
هذا التوزيع اللاحق **أساسي** لاتخاذ القرار.
    """
    display_markdown(en_method, ar_method)


def implementation_roadmap_content():
    display_header("4. Implementation Roadmap", "٤. خارطة طريق التنفيذ", level=2)
    en_roadmap = "Details of the **implementation phases** will go here."
    ar_roadmap = "تفاصيل **مراحل التنفيذ** ستوضع هنا."
    display_markdown(en_roadmap, ar_roadmap)
    
    data_en = {'Phase': ['**Phase 1**: Foundation'], 'Step': ['**Step 1**: Define Metrics'], 'Description': ['**Crucial** initial definitions.']}
    data_ar = {'المرحلة': ['**المرحلة ١**: التأسيس'], 'الخطوة': ['**الخطوة ١**: تعريف المقاييس'], 'الوصف': ['تعريفات أولية **حاسمة**']}
    df_display = pd.DataFrame(data_ar if current_lang == "ar" else data_en)
    
    # To render markdown in DataFrame cells, it's more complex.
    # Simplest for now is to use st.markdown for the table if complex formatting is needed,
    # or accept that st.dataframe might not render markdown in cells directly.
    # For this example, I'll show how to do it with st.markdown and HTML table.
    
    html_table = df_display.to_html(escape=False, index=False) # escape=False allows HTML/Markdown
    if current_lang == 'ar':
        st.markdown(f"<div dir='rtl' style='text-align: right;'>{html_table}</div>", unsafe_allow_html=True)
    else:
        st.markdown(html_table, unsafe_allow_html=True)


def note_to_practitioners_content():
    display_header("5. Note to Practitioners", "٥. ملاحظة للممارسين", level=2)
    en_notes = "Important **notes** and **considerations** for practitioners."
    ar_notes = "**ملاحظات** و **اعتبارات** هامة للممارسين."
    display_markdown(en_notes, ar_notes)

def interactive_illustration_content():
    display_header("6. Interactive Illustration", "٦. توضيح تفاعلي", level=2)
    display_markdown("Interactive **Beta-Binomial model** illustration.", 
                     "توضيح تفاعلي لنموذج **Beta-Binomial**.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        display_header("Prior Beliefs", "المعتقدات المسبقة", level=3)
        prior_alpha = st.slider("Prior Alpha (α₀)" if current_lang=="en" else "ألفا المسبقة (α₀)", 0.1, 50.0, 1.0, 0.1, key="ia_prior_a_v2")
        prior_beta = st.slider("Prior Beta (β₀)" if current_lang=="en" else "بيتا المسبقة (β₀)", 0.1, 50.0, 1.0, 0.1, key="ia_prior_b_v2")
        
        prior_mean = prior_alpha / (prior_alpha + prior_beta) if (prior_alpha + prior_beta) > 0 else 0
        prior_mean_en = f"**Prior Mean**: {prior_mean:.3f}"
        prior_mean_ar = f"**المتوسط المسبق**: {prior_mean:.3f}"
        display_markdown(prior_mean_en, prior_mean_ar)

        prior_ci = get_credible_interval(prior_alpha, prior_beta)
        prior_ci_en = f"**95% Credible Interval (Prior)**: [{prior_ci[0]:.3f}, {prior_ci[1]:.3f}]"
        prior_ci_ar = f"**فترة الموثوقية ٩٥٪ (المسبقة)**: [{prior_ci[0]:.3f}, {prior_ci[1]:.3f}]"
        display_markdown(prior_ci_en, prior_ci_ar)


    with col2:
        display_header("New Survey Data", "بيانات الاستطلاع الجديدة", level=3)
        num_surveys = st.slider("Number of New Surveys (n)" if current_lang=="en" else "عدد الاستطلاعات الجديدة (n)", 1, 500, 50, 1, key="ia_n_v2")
        num_satisfied = st.slider("Number Satisfied (k)" if current_lang=="en" else "عدد الراضين (k)", 0, num_surveys, int(num_surveys/2), 1, key="ia_k_v2")
        # ... (other interactive elements)

    posterior_alpha, posterior_beta = update_beta_parameters(prior_alpha, prior_beta, num_satisfied, num_surveys - num_satisfied)

    fig, ax = plt.subplots()
    if prior_alpha > 0 and prior_beta > 0:
        plot_beta_distribution(prior_alpha, prior_beta, "Prior", "المسبق", ax)
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
    en_conc = "This is the **final conclusion** of the proposal. It summarizes **key takeaways**."
    ar_conc = "هذه هي **الخلاصة النهائية** للمقترح. إنها تلخص **النقاط الرئيسية**."
    display_markdown(en_conc, ar_conc)


# --- Streamlit App Structure ---
selected_lang_option = st.sidebar.selectbox(
    label="Select Language / اختر اللغة",
    options=["English", "العربية"],
    index=0
)
current_lang = "ar" if selected_lang_option == "العربية" else "en"

app_title_en = "Proposal: Adaptive Bayesian Estimation"
app_title_ar = "مقترح: تقدير Bayesian التكيفي"
# Use st.markdown for main title to control alignment if needed, h1 is default for st.title
if current_lang == "ar":
    st.markdown(f"<h1 style='text-align: center; direction: rtl;'>{app_title_ar}</h1>", unsafe_allow_html=True)
else:
    st.markdown(f"<h1 style='text-align: center;'>{app_title_en}</h1>", unsafe_allow_html=True)


PAGES_SETUP = {
    "Intro": {"en": "1. Introduction & Objectives", "ar": "١. مقدمة وأهداف", "func": introduction_objectives_content},
    "Challenges": {"en": "2. Challenges Addressed", "ar": "٢. التحديات المعالجة", "func": challenges_addressed_content},
    "Methodology": {"en": "3. Core Methodology", "ar": "٣. المنهجية الأساسية", "func": bayesian_adaptive_methodology_content},
    "Roadmap": {"en": "4. Implementation Roadmap", "ar": "٤. خارطة طريق التنفيذ", "func": implementation_roadmap_content},
    "Practitioners": {"en": "5. Note to Practitioners", "ar": "٥. ملاحظة للممارسين", "func": note_to_practitioners_content},
    "Illustration": {"en": "6. Interactive Illustration", "ar": "٦. توضيح تفاعلي", "func": interactive_illustration_content},
    "Conclusion": {"en": "7. Conclusion", "ar": "٧. الخلاصة", "func": conclusion_content}
}

sidebar_display_options = [PAGES_SETUP[key][current_lang] for key in PAGES_SETUP]
sidebar_keys = list(PAGES_SETUP.keys())

st.sidebar.title("Proposal Sections" if current_lang == "en" else "أقسام المقترح")
selected_page_display_name = st.sidebar.radio(
    f"navigation_radio_{current_lang}", 
    sidebar_display_options,
    key=f"nav_radio_main_{current_lang}" # Added a more unique key
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
    st.error("Page loading error." if current_lang == "en" else "<div dir='rtl'>خطأ في تحميل الصفحة.</div>")

st.sidebar.markdown("---")
st.sidebar.info("Developed by Dr. Mohammad Nabhan." if current_lang == "en" else "تطوير د. محمد نبهان.")
