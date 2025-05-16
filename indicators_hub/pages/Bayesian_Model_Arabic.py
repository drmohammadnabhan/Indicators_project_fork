import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Adaptive Bayesian Estimation Proposal", layout="wide")

# --- Global variable for language ---
current_lang = "en" # Default

# --- MathJax Configuration ---
# This script tells MathJax to look for $...$ and \(...\) for inline math,
# and $$...$$ and \[...\] for display math.
# It will be injected once at the beginning of the app.
mathjax_script = """
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      displayMath: [['$$','$$'], ['\\[','\\]']],
      processEscapes: true,
      skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // Skip MathJax processing in these tags
    },
    TeX: {
        equationNumbers: { autoNumber: "AMS" },
        extensions: ["AMSmath.js", "AMSsymbols.js", "noErrors.js", "noUndefined.js"]
    }
  });
</script>
"""
st.components.v1.html(mathjax_script, height=0)


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

# --- Rendering Helpers (Arabic strings can now contain HTML and LaTeX) ---
def display_header(english_text_with_latex, arabic_text_with_latex_and_html, level=1):
    header_tag = f"h{level}"
    if current_lang == "ar":
        # Arabic text might contain HTML for bold/italic AND LaTeX for MathJax to process
        st.markdown(f"<{header_tag} dir='rtl' style='text-align: right;'>{arabic_text_with_latex_and_html}</{header_tag}>", unsafe_allow_html=True)
    else:
        # English text uses Streamlit's native Markdown and KaTeX processing
        st.markdown(f"<{header_tag}>{english_text_with_latex}</{header_tag}>", unsafe_allow_html=True)


def display_markdown(english_markdown_with_latex, arabic_text_with_latex_and_html):
    if current_lang == "ar":
        # Pass the Arabic string (which can have HTML and LaTeX) to markdown with unsafe_allow_html=True
        # MathJax will scan this HTML block.
        st.markdown(f"<div dir='rtl'>{arabic_text_with_latex_and_html}</div>", unsafe_allow_html=True)
    else:
        # English text uses Streamlit's native Markdown and KaTeX processing
        st.markdown(english_markdown_with_latex) # unsafe_allow_html=False (default) or True if Eng needs HTML

# --- Proposal Content Functions (Strings now mix HTML and LaTeX freely) ---

def introduction_objectives_content():
    display_header("1. Introduction & Objectives", "١. مقدمة وأهداف", level=2)
    
    en_intro_md = """This proposal outlines an **Adaptive Bayesian Estimation framework**.
It is designed to enhance the process of gathering and analyzing survey data for Hajj pilgrim satisfaction.
Here is **another bolded phrase**. And *italic text*.
And a formula: $$x^2 + y^2 = z^2$$
    """
    # Arabic now uses HTML for formatting and standard LaTeX for MathJax
    ar_intro_html_latex = """يحدد هذا المقترح إطارًا لـ <b>تقدير Bayesian التكيفي</b>.
إنه مصمم لتعزيز عملية جمع وتحليل بيانات الاستطلاع الخاصة برضا حجاج بيت الله الحرام.
هذه <b>عبارة أخرى بخط عريض</b>. وهذا <i>نص مائل</i>.
و معادلة: $$x^2 + y^2 = z^2$$
المعادلة أعلاه مثال.
    """
    display_markdown(en_intro_md, ar_intro_html_latex)

    display_header("1.1. Primary Objectives", "١.١. الأهداف الأساسية", level=3)
    en_objectives_md = """
* **Achieve Desired Precision Efficiently:** To obtain metrics with pre-defined precision.
* **Dynamic Sampling Adjustments:** To iteratively adjust sampling efforts.
* **Another Objective:** This is a **key goal**.
    """
    ar_objectives_html_latex = """
<ul>
    <li><b>تحقيق الدقة المطلوبة بكفاءة:</b> الحصول على مقاييس بدقة محددة مسبقًا.</li>
    <li><b>تعديلات أخذ العينات الديناميكية:</b> تعديل جهود أخذ العينات بشكل متكرر.</li>
    <li><b>هدف آخر:</b> هذا <b>هدف رئيسي</b>.</li>
</ul>
    """
    display_markdown(en_objectives_md, ar_objectives_html_latex)


def bayesian_adaptive_methodology_content():
    display_header("3. Core Methodology: Bayesian Adaptive Estimation", "٣. المنهجية الأساسية: تقدير Bayesian التكيفي", level=2)
    
    en_method_md = r"""
The core of Bayesian inference involves:
* **Prior Distribution ($P(\theta)$):** Initial belief.
* **Likelihood ($P(D|\theta)$):** How data $D$ supports $\theta$.
* **Posterior Distribution ($P(\theta|D)$):** Updated belief, calculated via Bayes' Theorem:
    $$ P(\theta|D) = \frac{P(D|\theta) \times P(\theta)}{P(D)} $$
    Where $P(D)$ is the marginal likelihood. We often use:
    $$ P(\theta|D) \propto P(D|\theta) \times P(\theta) $$
This posterior is **fundamental** for decision making.
    """
    ar_method_html_latex = r"""
<p>في جوهره، يجمع استدلال Bayesian بين المعرفة المسبقة والبيانات المرصودة للوصول إلى فهم محدث، يُعرف بالتوزيع اللاحق.</p>
<ul>
    <li><b>التوزيع المسبق ($P(\theta)$):</b> يمثل هذا اعتقادنا الأولي حول معلمة $\theta$.</li>
    <li><b>دالة الإمكان ($P(D|\theta)$):</b> تحدد هذه الدالة مدى احتمالية البيانات $D$.</li>
    <li><b>التوزيع اللاحق ($P(\theta|D)$):</b> هذا هو اعتقادنا المحدث. يتم حسابه باستخدام نظرية Bayes:
        $$ P(\theta|D) = \frac{P(D|\theta) P(\theta)}{P(D)} $$</li>
    <li>حيث $P(D)$ هو الإمكان الهامشي للبيانات. في الممارسة العملية، غالبًا ما نركز على التناسب:
        $$ P(\theta|D) \propto P(D|\theta) P(\theta) $$</li>
    <li><b>فترة الموثوقية:</b> في إحصاءات Bayesian، فترة الموثوقية هي نطاق من القيم يحتوي على المعلمة $\theta$ باحتمال معين.</li>
</ul>
<p>هذا التوزيع اللاحق <b>أساسي</b> لاتخاذ القرار.</p>
    """
    display_markdown(en_method_md, ar_method_html_latex)

# --- Dummy content for other sections for brevity ---
def challenges_addressed_content():
    display_header("2. Challenges Addressed", "٢. التحديات التي تعالجها هذه المنهجية", level=2)
    en_text = "Details of **challenges** and math $$\\sum x_i$$"
    ar_text = "تفاصيل <b>التحديات</b> ورياضيات $$\\sum x_i$$"
    display_markdown(en_text, ar_text)

def implementation_roadmap_content():
    display_header("4. Implementation Roadmap (Conceptual)", "٤. خارطة طريق التنفيذ (مفاهيمية)", level=2)
    en_text = "Roadmap with $N_0 = \\alpha + \\beta$."
    ar_text = "خارطة الطريق مع $N_0 = \\alpha + \\beta$."
    display_markdown(en_text, ar_text)

def note_to_practitioners_content():
    display_header("5. Note to Practitioners", "٥. ملاحظة للممارسين", level=2)
    en_text = "Notes involving $p$-values (just kidding, it's Bayesian!) and formula $p(1-p)$."
    ar_text = "ملاحظات تتضمن $p$-values (أمزح، إنه Bayesian!) وصيغة $p(1-p)$."
    display_markdown(en_text, ar_text)

def interactive_illustration_content():
    display_header("6. Interactive Illustration: Beta-Binomial Model", "٦. توضيح تفاعلي: نموذج Beta-Binomial", level=2)
    en_text = "Interactive Beta-Binomial model. Posterior is $Beta(\\alpha_0+k, \\beta_0+n-k)$."
    ar_text = "توضيح تفاعلي لنموذج Beta-Binomial. التوزيع اللاحق هو $Beta(\\alpha_0+k, \\beta_0+n-k)$."
    display_markdown(en_text, ar_text)

    # ... (rest of interactive illustration logic, can also use display_markdown for its text parts)
    # For plots, labels do not need complex LaTeX, so previous plot function is fine.

def conclusion_content():
    display_header("7. Conclusion", "٧. الخلاصة", level=2)
    en_text = "Final **conclusion** with perhaps some integral $\\int f(x) dx$."
    ar_text = "<b>الخلاصة</b> النهائية مع احتمال وجود تكامل $\\int f(x) dx$."
    display_markdown(en_text, ar_text)

# --- Streamlit App Structure ---
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
    f"navigation_radio_{current_lang}_v_mathjax", 
    sidebar_display_options,
    key=f"nav_radio_main_{current_lang}_v_mathjax"
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
