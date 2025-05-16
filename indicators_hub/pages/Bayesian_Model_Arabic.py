import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Adaptive Bayesian Estimation Proposal", layout="wide")

# --- Global variable for language ---
current_lang = "en"

# --- Helper Functions for Plotting (Unchanged from previous working version) ---
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

# --- Rendering Helpers (Adjusted for hybrid approach) ---
def display_header(english_text, arabic_text, level=1): # arabic_text is simple text for header
    text_to_display = arabic_text if current_lang == "ar" else english_text
    header_tag = f"h{level}"
    if current_lang == "ar":
        st.markdown(f"<{header_tag} dir='rtl' style='text-align: right;'>{text_to_display}</{header_tag}>", unsafe_allow_html=True)
    else:
        # English headers can use markdown directly if needed, or just be plain text.
        # If headers might contain markdown/KaTeX, use st.markdown for English too.
        st.markdown(f"<{header_tag}>{text_to_display}</{header_tag}>", unsafe_allow_html=True)


# NEW: Helper to render segmented content for the hybrid approach
def display_segmented_content(english_markdown_text, arabic_segments_list):
    if current_lang == "ar":
        st.markdown("<div dir='rtl' style='text-align: right;'>", unsafe_allow_html=True)
        for segment in arabic_segments_list:
            if segment["type"] == "html":
                st.markdown(segment["content"], unsafe_allow_html=True)
            elif segment["type"] == "latex":
                st.latex(segment["content"])
            elif segment["type"] == "markdown": # For general markdown in Arabic if needed
                st.markdown(segment["content"]) # Streamlit's KaTeX won't work here reliably
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(english_markdown_text) # English markdown handles its own KaTeX

# --- Proposal Content Functions (Example with Bayesian Methodology) ---

def introduction_objectives_content():
    display_header("1. Introduction & Objectives", "١. مقدمة وأهداف", level=2)
    en_intro_md = """This proposal outlines an **Adaptive Bayesian Estimation framework**.
And a formula: $$x^2 + y^2 = z^2$$"""
    ar_intro_segments = [
        {"type": "html", "content": "<p>يحدد هذا المقترح إطارًا لـ <b>تقدير Bayesian التكيفي</b>.</p>"},
        {"type": "html", "content": "<p>و معادلة:</p>"},
        {"type": "latex", "content": r"x^2 + y^2 = z^2"}
    ]
    display_segmented_content(en_intro_md, ar_intro_segments)
    # ... (rest of the intro content)

def bayesian_adaptive_methodology_content():
    display_header("3. Core Methodology: Bayesian Adaptive Estimation", "٣. المنهجية الأساسية: تقدير Bayesian التكيفي", level=2)
    
    english_methodology_md = r"""
At its heart, Bayesian inference combines prior knowledge with observed data to arrive at an updated understanding, known as the posterior distribution.

* **Prior Distribution ($P(\theta)$):** This represents our initial belief about a parameter $\theta$ (e.g., the proportion of satisfied pilgrims) *before* observing new data.
* **Likelihood ($P(D|\theta)$):** This quantifies how probable the observed data ($D$) is, given a particular value of the parameter $\theta$.
* **Posterior Distribution ($P(\theta|D)$):** This is our updated belief about $\theta$ *after* observing the data. It is calculated using Bayes' Theorem:
    $$ P(\theta|D) = \frac{P(D|\theta) \times P(\theta)}{P(D)} $$
    Where $P(D)$ is the marginal likelihood of the data, acting as a normalizing constant. In practice, we often focus on the proportionality:
    $$ P(\theta|D) \propto P(D|\theta) \times P(\theta) $$
* **Credible Interval:** In Bayesian statistics, a credible interval is a range of values that contains the parameter $\theta$ with a certain probability (e.g., 95%).
This posterior is **fundamental** for decision making.
    """

    arabic_methodology_segments = [
        {"type": "html", "content": "<p>في جوهره، يجمع استدلال Bayesian بين المعرفة المسبقة والبيانات المرصودة للوصول إلى فهم محدث، يُعرف بالتوزيع اللاحق.</p>"},
        {"type": "html", "content": "<ul>"},
        {"type": "html", "content": "<li><b>التوزيع المسبق ($P(\\theta)$):</b> يمثل هذا اعتقادنا الأولي حول معلمة $\\theta$ (على سبيل المثال، نسبة الحجاج الراضين) <i>قبل</i> ملاحظة البيانات الجديدة. (يمكن استخدام &alpha; للرموز البسيطة).</li>"},
        {"type": "html", "content": "<li><b>دالة الإمكان ($P(D|\\theta)$):</b> تحدد هذه الدالة مدى احتمالية البيانات المرصودة ($D$)، بالنظر إلى قيمة معينة للمعلومة $\\theta$.</li>"},
        {"type": "html", "content": "<li><b>التوزيع اللاحق ($P(\\theta|D)$):</b> هذا هو اعتقادنا المحدث حول $\\theta$ <i>بعد</i> ملاحظة البيانات. يتم حسابه باستخدام نظرية Bayes:</li>"},
        {"type": "latex", "content": r"P(\theta|D) = \frac{P(D|\theta) P(\theta)}{P(D)}"},
        {"type": "html", "content": "<li>حيث $P(D)$ هو الإمكان الهامشي للبيانات، ويعمل كثابت تسوية. في الممارسة العملية، غالبًا ما نركز على التناسب:</li>"},
        {"type": "latex", "content": r"P(\theta|D) \propto P(D|\theta) P(\theta)"},
        {"type": "html", "content": "<li><b>فترة الموثوقية:</b> في إحصاءات Bayesian، فترة الموثوقية هي نطاق من القيم يحتوي على المعلمة $\\theta$ باحتمال معين (على سبيل المثال، ٩٥٪).</li>"},
        {"type": "html", "content": "</ul>"},
        {"type": "html", "content": "<p>هذا التوزيع اللاحق <b>أساسي</b> لاتخاذ القرار.</p>"}
    ]
    
    display_segmented_content(english_methodology_md, arabic_methodology_segments)

    # --- Example for Modeling Satisfaction (Section 3.3) using the hybrid approach ---
    display_header("3.3. Modeling Satisfaction (e.g., using Beta-Binomial Model)", 
                   "٣.٣. نمذجة الرضا (على سبيل المثال، باستخدام نموذج Beta-Binomial)", level=3)

    english_modeling_md = r"""
For satisfaction metrics that are proportions, the Beta-Binomial model is suitable.
* Parameter of Interest: $\theta$
* Prior: $Beta(\alpha_0, \beta_0)$
* Likelihood: $P(k, n | \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k}$
* Posterior: $\theta | k, n \sim Beta(\alpha_0 + k, \beta_0 + n - k)$
* Point Estimate (Posterior Mean): $$ E[\theta | D] = \frac{\alpha_0 + k}{\alpha_0 + k + \beta_0 + n - k} $$
    """
    arabic_modeling_segments = [
        {"type": "html", "content": "<p>بالنسبة لمقاييس الرضا التي هي عبارة عن نسب، فإن نموذج Beta-Binomial مناسب جدًا.</p>"},
        {"type": "html", "content": "<ul><li>المعلمة ذات الأهمية: &theta; (ثيتا)</li></ul>"}, # Using HTML entity for simple Greek letter
        {"type": "html", "content": "<ul><li>التوزيع المسبق: $Beta(\\alpha_0, \\beta_0)$ (يمكن عرض هذا مباشرة إذا كان بسيطًا، أو عبر st.latex إذا كان معقدًا)</li></ul>"},
        # For complex formulas, use st.latex
        {"type": "html", "content": "<p>دالة الإمكان:</p>"},
        {"type": "latex", "content": r"P(k, n | \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k}"},
        {"type": "html", "content": "<p>التوزيع اللاحق:</p>"},
        {"type": "latex", "content": r"\theta | k, n \sim Beta(\alpha_0 + k, \beta_0 + n - k)"},
        {"type": "html", "content": "<p>التقدير النقطي (المتوسط اللاحق):</p>"},
        {"type": "latex", "content": r"E[\theta | D] = \frac{\alpha_0 + k}{\alpha_0 + k + \beta_0 + n - k}"}
    ]
    display_segmented_content(english_modeling_md, arabic_modeling_segments)


# --- Dummy stubs for other content functions ---
def challenges_addressed_content():
    display_header("2. Challenges Addressed", "٢. التحديات المعالجة", level=2)
    # ... convert to segmented display ...

def implementation_roadmap_content():
    display_header("4. Implementation Roadmap", "٤. خارطة طريق التنفيذ", level=2)
    # ... convert to segmented display ...

def note_to_practitioners_content():
    display_header("5. Note to Practitioners", "٥. ملاحظة للممارسين", level=2)
    # ... convert to segmented display ...

def interactive_illustration_content():
    display_header("6. Interactive Illustration", "٦. توضيح تفاعلي", level=2)
    # ... (This section primarily uses Streamlit widgets and plots, less heavy on mixed text/KaTeX)
    # ... Text within can be handled by display_segmented_content if needed
    st.markdown("---") # Example placeholder
    display_segmented_content(
        "**Prior Beliefs** (English Markdown)",
        [{"type": "html", "content": "<b>المعتقدات المسبقة</b> (HTML عربي)"}]
    )
    # ... rest of interactive elements

def conclusion_content():
    display_header("7. Conclusion", "٧. الخلاصة", level=2)
    # ... convert to segmented display ...

# --- Streamlit App Structure (Sidebar etc. remains similar) ---
selected_lang_option = st.sidebar.selectbox(
    label="Select Language / اختر اللغة",
    options=["English", "العربية"],
    index=0
)
current_lang = "ar" if selected_lang_option == "العربية" else "en"

app_title_en = "Proposal: Adaptive Bayesian Estimation"
app_title_ar = "مقترح: تقدير Bayesian التكيفي"

if current_lang == "ar":
    st.markdown(f"<h1 style='text-align: center; direction: rtl;'>{app_title_ar}</h1>", unsafe_allow_html=True)
else:
    st.markdown(f"<h1 style='text-align: center;'>{app_title_en}</h1>", unsafe_allow_html=True)

PAGES_SETUP = {
    "Intro": {"en": "1. Introduction & Objectives", "ar": "١. مقدمة وأهداف", "func": introduction_objectives_content},
    "Challenges": {"en": "2. Challenges Addressed", "ar": "٢. التحديات التي تعالجها هذه المنهجية", "func": challenges_addressed_content}, # Corrected Arabic key
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
    f"navigation_radio_{current_lang}_v_hybrid", 
    sidebar_display_options,
    key=f"nav_radio_main_{current_lang}_v_hybrid"
)

selected_page_key = None
for key_iter in sidebar_keys:
    if PAGES_SETUP[key_iter][current_lang] == selected_page_display_name:
        selected_page_key = key_iter
        break

if selected_page_key:
    page_function_to_call = PAGES_SETUP[selected_page_key]["func"]
    page_function_to_call()
else:
    st.error("Page loading error." if current_lang == "en" else "<div dir='rtl'>خطأ في تحميل الصفحة.</div>")

st.sidebar.markdown("---")
st.sidebar.info("Developed by Dr. Mohammad Nabhan." if current_lang == "en" else "تطوير د. محمد نبهان.")
