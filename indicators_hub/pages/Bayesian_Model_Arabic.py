import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Adaptive Bayesian Estimation Proposal", layout="wide")

# --- Global variable for language ---
current_lang = "en"

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

# --- Rendering Helpers (Modified for segmented Arabic content) ---
def display_header(english_text, arabic_text, level=1): # arabic_text here is simple text for header
    text_to_display = arabic_text if current_lang == "ar" else english_text
    header_tag = f"h{level}"
    if current_lang == "ar":
        st.markdown(f"<{header_tag} dir='rtl' style='text-align: right;'>{text_to_display}</{header_tag}>", unsafe_allow_html=True)
    else:
        st.markdown(f"<{header_tag}>{text_to_display}</{header_tag}>", unsafe_allow_html=True)

def display_content(english_markdown_text, arabic_segments_list):
    if current_lang == "ar":
        st.markdown("<div dir='rtl' style='text-align: right;'>", unsafe_allow_html=True) # Apply RTL and text-align to the whole block
        for segment in arabic_segments_list:
            if segment["type"] == "html":
                st.markdown(segment["content"], unsafe_allow_html=True)
            elif segment["type"] == "latex":
                st.latex(segment["content"]) # st.latex is specifically for KaTeX
            elif segment["type"] == "markdown": # If you still want to use markdown for some Arabic parts
                 st.markdown(segment["content"])
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(english_markdown_text) # English uses markdown which handles KaTeX

# --- Proposal Content Functions ---

def introduction_objectives_content():
    display_header("1. Introduction & Objectives", "١. مقدمة وأهداف", level=2)
    
    en_intro_md = """This proposal outlines an **Adaptive Bayesian Estimation framework**.
It is designed to enhance the process of gathering and analyzing survey data for Hajj pilgrim satisfaction.
Here is **another bolded phrase**. And *italic text*.
And a formula: $$x^2 + y^2 = z^2$$
    """
    ar_intro_segments = [
        {"type": "html", "content": "يحدد هذا المقترح إطارًا لـ <b>تقدير Bayesian التكيفي</b>."},
        {"type": "html", "content": "إنه مصمم لتعزيز عملية جمع وتحليل بيانات الاستطلاع الخاصة برضا حجاج بيت الله الحرام."},
        {"type": "html", "content": "هذه <b>عبارة أخرى بخط عريض</b>. وهذا <i>نص مائل</i>."},
        {"type": "latex", "content": r"x^2 + y^2 = z^2"},
        {"type": "html", "content": "المعادلة أعلاه مثال."}
    ]
    display_content(en_intro_md, ar_intro_segments)

    # ... (rest of the content functions would need similar updates for Arabic parts)
    # For brevity, I will only update bayesian_adaptive_methodology_content fully as an example

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
    ar_method_segments = [
        {"type": "html", "content": "<p>في جوهره، يجمع استدلال Bayesian بين المعرفة المسبقة والبيانات المرصودة للوصول إلى فهم محدث، يُعرف بالتوزيع اللاحق.</p>"},
        {"type": "html", "content": "<ul>"},
        {"type": "html", "content": "<li><b>التوزيع المسبق ($P(\\theta)$):</b> يمثل هذا اعتقادنا الأولي حول معلمة $\\theta$ (على سبيل المثال، نسبة الحجاج الراضين) <i>قبل</i> ملاحظة البيانات الجديدة.</li>"},
        {"type": "html", "content": "<li><b>دالة الإمكان ($P(D|\\theta)$):</b> تحدد هذه الدالة مدى احتمالية البيانات المرصودة ($D$)، بالنظر إلى قيمة معينة للمعلومة $\\theta$.</li>"},
        {"type": "html", "content": "<li><b>التوزيع اللاحق ($P(\\theta|D)$):</b> هذا هو اعتقادنا المحدث حول $\\theta$ <i>بعد</i> ملاحظة البيانات. يتم حسابه باستخدام نظرية Bayes:</li>"},
        {"type": "latex", "content": r"P(\theta|D) = \frac{P(D|\theta) P(\theta)}{P(D)}"},
        {"type": "html", "content": "<li>حيث $P(D)$ هو الإمكان الهامشي للبيانات. في الممارسة العملية، غالبًا ما نركز على التناسب:</li>"},
        {"type": "latex", "content": r"P(\theta|D) \propto P(D|\theta) P(\theta)"},
        {"type": "html", "content": "<li><b>فترة الموثوقية:</b> في إحصاءات Bayesian، فترة الموثوقية هي نطاق من القيم يحتوي على المعلمة $\\theta$ باحتمال معين.</li>"},
        {"type": "html", "content": "</ul>"},
        {"type": "html", "content": "<p>هذا التوزيع اللاحق <b>أساسي</b> لاتخاذ القرار.</p>"}
    ]
    display_content(en_method_md, ar_method_segments)

# --- Dummy content for other sections to make the app runnable ---
def challenges_addressed_content():
    display_header("2. Challenges Addressed", "٢. التحديات المعالجة", level=2)
    display_content("English challenges **markdown**.", [{"type": "html", "content": "<b>التحديات</b> العربية."}])

def implementation_roadmap_content():
    display_header("4. Implementation Roadmap", "٤. خارطة طريق التنفيذ", level=2)
    display_content("English roadmap.", [{"type": "html", "content": "خارطة الطريق."}])
    data_en = {'Phase': ['**Phase 1**'], 'Description': ['Description 1']}
    data_ar = [{'المرحلة': '<b>المرحلة ١</b>', 'الوصف': 'وصف ١'}] # HTML in data
    df_display_data = data_ar if current_lang == "ar" else data_en
    df_display = pd.DataFrame(df_display_data)
    
    html_table = df_display.to_html(escape=False, index=False, classes='dataframe table table-striped', border=0)
    if current_lang == 'ar':
        st.markdown(f"<div dir='rtl' style='text-align: right;'>{html_table}</div>", unsafe_allow_html=True)
    else:
        st.markdown(html_table, unsafe_allow_html=True)


def note_to_practitioners_content():
    display_header("5. Note to Practitioners", "٥. ملاحظة للممارسين", level=2)
    display_content("English notes.", [{"type": "html", "content": "ملاحظات للممارسين."}])

def interactive_illustration_content():
    display_header("6. Interactive Illustration", "٦. توضيح تفاعلي", level=2)
    display_content(
        "Interactive Beta-Binomial model illustration using **markdown for English**.",
        [{"type": "html", "content": "توضيح تفاعلي لنموذج <b>Beta-Binomial</b>."}]
    )
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        display_header("Prior Beliefs", "المعتقدات المسبقة", level=3)
        prior_alpha = st.slider("Prior Alpha (α₀)" if current_lang=="en" else "ألفا المسبقة (α₀)", 0.1, 50.0, 1.0, 0.1, key="ia_prior_a_v4")
        prior_beta = st.slider("Prior Beta (β₀)" if current_lang=="en" else "بيتا المسبقة (β₀)", 0.1, 50.0, 1.0, 0.1, key="ia_prior_b_v4")
        
        prior_mean = prior_alpha / (prior_alpha + prior_beta) if (prior_alpha + prior_beta) > 0 else 0
        prior_mean_en_md = f"**Prior Mean**: {prior_mean:.3f}"
        prior_mean_ar_html = [{"type": "html", "content": f"<b>المتوسط المسبق</b>: {prior_mean:.3f}"}]
        display_content(prior_mean_en_md, prior_mean_ar_html)

        prior_ci = get_credible_interval(prior_alpha, prior_beta)
        prior_ci_en_md = f"**95% Credible Interval (Prior)**: [{prior_ci[0]:.3f}, {prior_ci[1]:.3f}]"
        prior_ci_ar_html = [{"type": "html", "content": f"<b>فترة الموثوقية ٩٥٪ (المسبقة)</b>: [{prior_ci[0]:.3f}, {prior_ci[1]:.3f}]"}]
        display_content(prior_ci_en_md, prior_ci_ar_html)

    with col2:
        display_header("New Survey Data", "بيانات الاستطلاع الجديدة", level=3)
        num_surveys = st.slider("Number of New Surveys (n)" if current_lang=="en" else "عدد الاستطلاعات الجديدة (n)", 1, 500, 50, 1, key="ia_n_v4")
        num_satisfied = st.slider("Number Satisfied (k)" if current_lang=="en" else "عدد الراضين (k)", 0, num_surveys, int(num_surveys/2), 1, key="ia_k_v4")

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
    display_content(
        "This is the **final conclusion**.",
        [{"type": "html", "content": "هذه هي <b>الخلاصة النهائية</b>."}]
    )

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
    f"navigation_radio_{current_lang}_v3", 
    sidebar_display_options,
    key=f"nav_radio_main_{current_lang}_v3"
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
