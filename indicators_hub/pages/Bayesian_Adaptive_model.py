import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import os # For image path if you add one locally

# Page Configuration
st.set_page_config(layout="wide", page_title="Adaptive Bayesian Estimation Proposal")

# --- Global variable for language ---
current_lang = "en" # Default

# --- Helper Functions for Plotting ---
def plot_beta_distribution(alpha, beta, label_en, label_ar, ax): # Simplified label arguments
    """Plots a Beta distribution."""
    label = label_ar if current_lang == "ar" else label_en
    x = np.linspace(0.001, 0.999, 500) # Avoid 0 and 1 for pdf
    y = stats.beta.pdf(x, alpha, beta)
    ax.plot(x, y, label=f'{label} (α={alpha:.2f}, β={beta:.2f})')
    ax.fill_between(x, y, alpha=0.2)
    ax.set_ylim(bottom=0)


def update_beta_parameters(prior_alpha, prior_beta, successes, failures):
    """Updates Beta parameters given new data."""
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + failures
    return posterior_alpha, posterior_beta

def get_credible_interval(alpha, beta, conf_level=0.95):
    """Calculates the credible interval for a Beta distribution."""
    if alpha <= 0 or beta <= 0: # Invalid parameters
        return (0.0, 0.0)
    try:
        return stats.beta.interval(conf_level, alpha, beta)
    except ValueError:
        return (0.0, 0.0)

# --- Rendering Helpers (Hybrid Approach) ---
def display_header(english_text, arabic_text, level=1):
    text_to_display = arabic_text if current_lang == "ar" else english_text
    header_tag = f"h{level}"
    if current_lang == "ar":
        st.markdown(f"<{header_tag} dir='rtl' style='text-align: right;'>{text_to_display}</{header_tag}>", unsafe_allow_html=True)
    else:
        st.markdown(f"<{header_tag}>{english_text}</{header_tag}>", unsafe_allow_html=True)

def display_text_content(english_content, arabic_segments_list):
    """
    Renders content. English content is a single markdown string.
    Arabic content is a list of segments, each typed 'html' or 'latex'.
    """
    if current_lang == "ar":
        st.markdown("<div dir='rtl' style='text-align: right;'>", unsafe_allow_html=True)
        for segment in arabic_segments_list:
            if segment["type"] == "html":
                st.markdown(segment["content"], unsafe_allow_html=True)
            elif segment["type"] == "latex":
                st.latex(segment["content"])
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(english_content) # Assumes English content handles its own KaTeX

# --- Proposal Content (Original English + Arabic Segments) ---

def introduction_objectives_content():
    display_header("1. Introduction & Objectives", "١. مقدمة وأهداف", level=1) # Changed to h1 for main section titles

    english_intro_md = """
This proposal outlines an **Adaptive Bayesian Estimation framework** designed to enhance the process of gathering and analyzing survey data for Hajj pilgrim satisfaction and the assessment of services provided by various companies.

The current practice of developing satisfaction metrics month over month faces complexities, such as delays in pilgrim arrivals or non-uniformity across different months, making it difficult to consistently achieve high-confidence and low-error confidence intervals for key indicators. This proposal aims to introduce a more dynamic, efficient, and robust methodology.
    """
    arabic_intro_segments = [
        {"type": "html", "content": "<p>يحدد هذا المقترح إطارًا لـ <b>تقدير Bayesian التكيفي</b> مصممًا لتعزيز عملية جمع وتحليل بيانات الاستطلاع الخاصة برضا حجاج بيت الله الحرام وتقييم الخدمات المقدمة من مختلف الشركات.</p>"},
        {"type": "html", "content": "<p>تواجه الممارسة الحالية المتمثلة في تطوير مقاييس الرضا شهرًا بعد شهر تعقيدات، مثل التأخير في وصول الحجاج أو عدم الانتظام عبر الأشهر المختلفة، مما يجعل من الصعب تحقيق فترات ثقة عالية الدقة ومنخفضة الخطأ للمؤشرات الرئيسية بشكل مستمر. يهدف هذا المقترح إلى تقديم منهجية أكثر ديناميكية وكفاءة وقوة.</p>"}
    ]
    display_text_content(english_intro_md, arabic_intro_segments)

    display_header("1.1. Primary Objectives", "١.١. الأهداف الأساسية", level=2) # Changed to h2
    english_objectives_md = """
* **Achieve Desired Precision Efficiently:** To obtain satisfaction metrics and service provider assessments with pre-defined levels of precision (i.e., narrow credible intervals at a specific confidence level) using optimized sample sizes.
* **Dynamic Sampling Adjustments:** To iteratively adjust sampling efforts based on accumulating evidence. This means collecting more data only when and where it's needed to meet precision targets, avoiding over-sampling or under-sampling.
* **Timely and Reliable Estimates:** To provide decision-makers with more timely and statistically robust estimates, allowing for quicker responses to emerging issues or trends in pilgrim satisfaction.
* **Incorporate Prior Knowledge:** To formally integrate knowledge from previous survey waves, historical data, or expert opinions into the estimation process, leading to more informed starting points and potentially faster convergence to precise estimates.
* **Adapt to Changing Conditions:** To develop a system that can adapt to changes in satisfaction levels or service provider performance over time, for instance, by adjusting the influence of older data.
* **Enhanced Subgroup Analysis:** To facilitate more reliable analysis of specific pilgrim subgroups or service aspects by adaptively ensuring sufficient data is collected for these segments.
    """
    arabic_objectives_segments = [
        {"type": "html", "content": "<ul>"},
        {"type": "html", "content": "<li><b>تحقيق الدقة المطلوبة بكفاءة:</b> الحصول على مقاييس الرضا وتقييمات مقدمي الخدمات بمستويات دقة محددة مسبقًا (أي فترات موثوقية ضيقة عند مستوى ثقة معين) باستخدام أحجام عينات مُحسَّنة.</li>"},
        {"type": "html", "content": "<li><b>تعديلات أخذ العينات الديناميكية:</b> تعديل جهود أخذ العينات بشكل متكرر بناءً على الأدلة المتراكمة. هذا يعني جمع المزيد من البيانات فقط عند الحاجة إليها وحيثما تكون هناك حاجة إليها لتحقيق أهداف الدقة، وتجنب الإفراط في أخذ العينات أو نقصها.</li>"},
        {"type": "html", "content": "<li><b>تقديرات موثوقة وفي الوقت المناسب:</b> تزويد صانعي القرار بتقديرات أكثر موثوقية من الناحية الإحصائية وفي الوقت المناسب، مما يسمح باستجابات أسرع للقضايا الناشئة أو الاتجاهات في رضا الحجاج.</li>"},
        {"type": "html", "content": "<li><b>دمج المعرفة المسبقة:</b> دمج المعرفة من موجات الاستطلاع السابقة أو البيانات التاريخية أو آراء الخبراء رسميًا في عملية التقدير، مما يؤدي إلى نقاط انطلاق أكثر استنارة واحتمال تقارب أسرع للتقديرات الدقيقة.</li>"},
        {"type": "html", "content": "<li><b>التكيف مع الظروف المتغيرة:</b> تطوير نظام يمكنه التكيف مع التغيرات في مستويات الرضا أو أداء مقدمي الخدمات بمرور الوقت، على سبيل المثال، عن طريق تعديل تأثير البيانات القديمة.</li>"},
        {"type": "html", "content": "<li><b>تحليل محسن للمجموعات الفرعية:</b> تسهيل تحليل أكثر موثوقية لمجموعات فرعية محددة من الحجاج أو جوانب الخدمة من خلال ضمان جمع بيانات كافية لهذه الشرائح بشكل تكيفي.</li>"},
        {"type": "html", "content": "</ul>"}
    ]
    display_text_content(english_objectives_md, arabic_objectives_segments)

def challenges_addressed_content():
    display_header("2. Challenges Addressed by this Methodology", "٢. التحديات التي تعالجها هذه المنهجية", level=1)
    # ... Populate with your full English markdown and corresponding Arabic segments ...
    english_challenges_md = """
The proposed Bayesian adaptive estimation framework directly addresses several key challenges currently faced in the Hajj survey process:

* **Difficulty in Obtaining Stable Confidence Intervals:**
    * **Challenge:** Operational complexities like staggered pilgrim arrivals...
    * **Bayesian Solution:** The adaptive nature allows sampling to continue...
* **Inefficiency of Fixed Sample Size Approaches:** (etc.)
    """ # Keep this content identical to the English app provided
    arabic_challenges_segments = [
        {"type": "html", "content": "<p>يعالج إطار تقدير Bayesian التكيفي المقترح بشكل مباشر العديد من التحديات الرئيسية التي تواجه حاليًا في عملية استطلاع الحج:</p>"},
        {"type": "html", "content": "<ul><li><b>صعوبة الحصول على فترات ثقة مستقرة:</b><ul><li><b>التحدي:</b> التعقيدات التشغيلية...</li><li><b>حل Bayesian:</b> تسمح الطبيعة التكيفية...</li></ul></li></ul>"} # Example structure
        # ... Add all other challenges in this segmented format for Arabic
    ]
    display_text_content(english_challenges_md, arabic_challenges_segments)


def bayesian_adaptive_methodology_content():
    display_header("3. Core Methodology: Bayesian Adaptive Estimation", "٣. المنهجية الأساسية: تقدير Bayesian التكيفي", level=1)
    
    english_methodology_intro_md = """
The Bayesian adaptive estimation framework is an iterative process that leverages Bayes' theorem to update our beliefs about pilgrim satisfaction or service performance as new survey data is collected. This allows for dynamic adjustments to the sampling strategy.
    """
    arabic_methodology_intro_segments = [
        {"type": "html", "content": "<p>إطار تقدير Bayesian التكيفي هو عملية تكرارية تستفيد من نظرية Bayes لتحديث معتقداتنا حول رضا الحجاج أو أداء الخدمة عند جمع بيانات استطلاع جديدة. هذا يسمح بإجراء تعديلات ديناميكية على استراتيجية أخذ العينات.</p>"}
    ]
    display_text_content(english_methodology_intro_md, arabic_methodology_intro_segments)

    display_header("3.1. Fundamental Concepts", "٣.١. المفاهيم الأساسية", level=2)
    english_concepts_md = r"""
At its heart, Bayesian inference combines prior knowledge with observed data to arrive at an updated understanding, known as the posterior distribution.

* **Prior Distribution ($P(\theta)$):** This represents our initial belief about a parameter $\theta$ (e.g., the proportion of satisfied pilgrims) *before* observing new data. It can be based on historical data, expert opinion, or be deliberately "uninformative" if we want the data to speak for itself.
* **Likelihood ($P(D|\theta)$):** This quantifies how probable the observed data ($D$) is, given a particular value of the parameter $\theta$.
* **Posterior Distribution ($P(\theta|D)$):** This is our updated belief about $\theta$ *after* observing the data. It is calculated using Bayes' Theorem:
    $$ P(\theta|D) = \frac{P(D|\theta) \times P(\theta)}{P(D)} $$
    Where $P(D)$ is the marginal likelihood. Often, we use:
    $$ P(\theta|D) \propto P(D|\theta) \times P(\theta) $$
* **Credible Interval:** A range of values containing $\theta$ with a certain probability.
    """
    arabic_concepts_segments = [
        {"type": "html", "content": "<p>في جوهره، يجمع استدلال Bayesian بين المعرفة المسبقة والبيانات المرصودة للوصول إلى فهم محدث، يُعرف بالتوزيع اللاحق.</p>"},
        {"type": "html", "content": "<ul>"},
        {"type": "html", "content": "<li><b>التوزيع المسبق ($P(\\theta)$):</b> يمثل هذا اعتقادنا الأولي حول معلمة &theta; (ثيتا).</li>"},
        {"type": "html", "content": "<li><b>دالة الإمكان ($P(D|\\theta)$):</b> تحدد مدى احتمالية البيانات $D$.</li>"},
        {"type": "html", "content": "<li><b>التوزيع اللاحق ($P(\\theta|D)$):</b> اعتقادنا المحدث، يحسب بنظرية Bayes:</li>"},
        {"type": "latex", "content": r"P(\theta|D) = \frac{P(D|\theta) P(\theta)}{P(D)}"},
        {"type": "html", "content": "<li>حيث $P(D)$ هو الإمكان الهامشي. غالبًا ما نستخدم:</li>"},
        {"type": "latex", "content": r"P(\theta|D) \propto P(D|\theta) P(\theta)"},
        {"type": "html", "content": "<li><b>فترة الموثوقية:</b> نطاق قيم يحتوي على &theta; باحتمال معين.</li>"},
        {"type": "html", "content": "</ul>"}
    ]
    display_text_content(english_concepts_md, arabic_concepts_segments)

    # ... Continue this pattern for all subsections 3.2, 3.3, 3.4, 3.5
    # Remember to remove the st.image call or replace it with a local one.
    # For 3.3, ensure the formula for posterior mean is included.

def implementation_roadmap_content():
    display_header("4. Implementation Roadmap (Conceptual)", "٤. خارطة طريق التنفيذ (مفاهيمية)", level=1)
    english_roadmap_intro = "Implementing the Bayesian adaptive estimation framework involves several key stages:"
    arabic_roadmap_intro_segments = [{"type": "html", "content": "<p>يتضمن تنفيذ إطار تقدير Bayesian التكيفي عدة مراحل رئيسية:</p>"}]
    display_text_content(english_roadmap_intro, arabic_roadmap_intro_segments)

    # English DataFrame Data (from your original app)
    data_en = {
        "Phase": ["Phase 1: Foundation & Pilot", "Phase 1: Foundation & Pilot", "Phase 2: Iterative Development & Testing", "Phase 2: Iterative Development & Testing", "Phase 3: Full-Scale Deployment & Refinement", "Phase 3: Full-Scale Deployment & Refinement"],
        "Step": [
            "1. Define Key Metrics & Precision Targets",
            "2. System Setup & Prior Elicitation",
            "3. Model Development & Initial Batching Logic",
            "4. Dashboard Development & Pilot Testing",
            "5. Scaled Rollout & Heterogeneity Modeling",
            "6. Continuous Monitoring & Improvement"
        ],
        "Description": [
            "Identify critical satisfaction indicators and service aspects. For each, define the desired level of precision (e.g., 95% credible interval width of ±3%).",
            "Establish data collection pathways. For each metric, determine initial priors (e.g., $Beta(1,1)$ for uninformative, or derive from historical averages if stable and relevant).",
            "Develop the Bayesian models (e.g., Beta-Binomial) for core metrics. Implement the logic for posterior updates and initial rules for determining subsequent sample batch sizes.",
            "Create a dashboard to visualize posterior distributions, credible intervals, precision achieved vs. target, and sampling progress. Conduct a pilot study on a limited scale to test the workflow, model performance, and adaptive logic.",
            "Gradually roll out the adaptive system across more survey areas/service providers. Implement or refine mechanisms for handling data heterogeneity over time (e.g., discount factors, change-point monitoring).",
            "Continuously monitor the system's performance, resource efficiency, and the quality of estimates. Refine models, priors, and adaptive rules based on ongoing learning and feedback."
        ]
    }
    # Arabic DataFrame Data (You'll need to provide the translations)
    data_ar = {
        "المرحلة": ["المرحلة ١: التأسيس والتجربة", "المرحلة ١: التأسيس والتجربة", "المرحلة ٢: التطوير التكراري", "المرحلة ٢: التطوير التكراري", "المرحلة ٣: النشر الكامل", "المرحلة ٣: النشر الكامل"],
        "الخطوة": [
            "١. تحديد المقاييس الرئيسية", "٢. إعداد النظام والتوزيعات المسبقة",
            "٣. تطوير النموذج ومنطق الدفعات", "٤. تطوير لوحة التحكم والاختبار",
            "٥. التوسع ونمذجة التباين", "٦. المراقبة والتحسين المستمر"
        ],
        "الوصف": [
            "تحديد مؤشرات الرضا الهامة. تحديد الدقة المطلوبة (مثال: فترة موثوقية ٩٥٪ بعرض ±٣٪).",
            "إنشاء مسارات جمع البيانات. تحديد التوزيعات المسبقة الأولية (مثال: $Beta(1,1)$).", # KaTeX will render in English version
            "تطوير نماذج Bayesian. تنفيذ منطق التحديثات اللاحقة وقواعد حجم الدفعات.",
            "إنشاء لوحة تحكم. إجراء دراسة تجريبية لاختبار سير العمل.",
            "توسيع النظام تدريجيًا. تنفيذ آليات للتعامل مع تباين البيانات.",
            "مراقبة أداء النظام باستمرار. تحسين النماذج بناءً على التعلم."
        ]
    }
    df_display = pd.DataFrame(data_ar if current_lang == "ar" else data_en)
    
    # For English, st.dataframe is fine and will render KaTeX in cells.
    # For Arabic, if KaTeX in cells is problematic with to_html, use simple text or HTML entities.
    if current_lang == 'ar':
        # For Arabic, we might need to be careful if KaTeX is in data_ar and rendered via to_html
        # Simplest is to ensure data_ar uses HTML entities for simple math if needed.
        st.markdown(f"<div dir='rtl'>{df_display.to_html(escape=False, index=False, classes='dataframe')}</div>", unsafe_allow_html=True)
    else:
        st.dataframe(df_display, hide_index=True, use_container_width=True)


def note_to_practitioners_content():
    display_header("5. Note to Practitioners", "٥. ملاحظة للممارسين", level=1)
    # ... Populate with your full English markdown and corresponding Arabic segments ...

def interactive_illustration_content():
    display_header("6. Interactive Illustration: Beta-Binomial Model", "٦. توضيح تفاعلي: نموذج Beta-Binomial", level=1)
    
    english_interactive_intro = """
This section provides a simple interactive illustration of how a Beta prior is updated to a Beta posterior with new data (Binomial likelihood). This is the core of estimating a proportion (e.g., satisfaction rate) in a Bayesian way.
    """
    arabic_interactive_intro_segments = [
        {"type": "html", "content": "<p>يقدم هذا القسم توضيحًا تفاعليًا بسيطًا لكيفية تحديث توزيع Beta المسبق إلى توزيع Beta لاحق ببيانات جديدة (دالة الإمكان Binomial). هذا هو جوهر تقدير نسبة (على سبيل المثال، معدل الرضا) بطريقة Bayesian.</p>"}
    ]
    display_text_content(english_interactive_intro, arabic_interactive_intro_segments)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        display_header("Prior Beliefs", "المعتقدات المسبقة", level=2)
        english_prior_desc = "The Beta distribution $Beta(\\alpha, \\beta)$ is a common prior for proportions. $\\alpha$ can be thought of as prior 'successes' and $\\beta$ as prior 'failures'. $Beta(1,1)$ is a uniform (uninformative) prior."
        arabic_prior_desc_segments = [
            {"type": "html", "content":"<p>توزيع Beta $Beta(&alpha;, &beta;)$ هو توزيع مسبق شائع للنسب. يمكن اعتبار &alpha; بمثابة 'نجاحات' مسبقة و &beta; بمثابة 'إخفاقات' مسبقة. $Beta(1,1)$ هو توزيع منتظم (غير إعلامي) مسبق.</p>"} # Using HTML entities
        ]
        display_text_content(english_prior_desc, arabic_prior_desc_segments)
        
        prior_alpha_label_en = "Prior Alpha (α₀)"
        prior_alpha_label_ar = "ألفا المسبقة (α₀)"
        prior_alpha = st.slider(prior_alpha_label_ar if current_lang == "ar" else prior_alpha_label_en, 0.1, 50.0, 1.0, 0.1, key="ia_prior_a_final")

        prior_beta_label_en = "Prior Beta (β₀)"
        prior_beta_label_ar = "بيتا المسبقة (β₀)"
        prior_beta = st.slider(prior_beta_label_ar if current_lang == "ar" else prior_beta_label_en, 0.1, 50.0, 1.0, 0.1, key="ia_prior_b_final")
        
        prior_mean = prior_alpha / (prior_alpha + prior_beta) if (prior_alpha + prior_beta) > 0 else 0
        st.write(f"{'المتوسط المسبق' if current_lang == 'ar' else 'Prior Mean'}: {prior_mean:.3f}")
        
        prior_ci = get_credible_interval(prior_alpha, prior_beta)
        st.write(f"{'فترة الموثوقية ٩٥٪ (المسبقة)' if current_lang == 'ar' else '95% Credible Interval (Prior)'}: [{prior_ci[0]:.3f}, {prior_ci[1]:.3f}], {'العرض' if current_lang == 'ar' else 'Width'}: {abs(prior_ci[1]-prior_ci[0]):.3f}")

    with col2:
        display_header("New Survey Data (Likelihood)", "بيانات الاستطلاع الجديدة (دالة الإمكان)", level=2)
        display_text_content("Enter the results from a new batch of surveys.", [{"type": "html", "content": "<p>أدخل النتائج من دفعة جديدة من الاستطلاعات.</p>"}])
        
        num_surveys_label_en = "Number of New Surveys (n)"
        num_surveys_label_ar = "عدد الاستطلاعات الجديدة (n)"
        num_surveys = st.slider(num_surveys_label_ar if current_lang == "ar" else num_surveys_label_en, 1, 500, 50, 1, key="ia_n_final")

        num_satisfied_label_en = "Number Satisfied in New Surveys (k)"
        num_satisfied_label_ar = "عدد الراضين في الاستطلاعات الجديدة (k)"
        num_satisfied = st.slider(num_satisfied_label_ar if current_lang == "ar" else num_satisfied_label_en, 0, num_surveys, int(num_surveys/2), 1, key="ia_k_final")
        
        st.write(f"{'الرضا الملاحظ في البيانات الجديدة' if current_lang == 'ar' else 'Observed Satisfaction in New Data'}: {num_satisfied/num_surveys if num_surveys > 0 else 0:.3f}")

    st.markdown("---")
    display_header("Posterior Beliefs (After Update)", "المعتقدات اللاحقة (بعد التحديث)", level=2)
    posterior_alpha, posterior_beta = update_beta_parameters(prior_alpha, prior_beta, num_satisfied, num_surveys - num_satisfied)
    
    # For this formula, English uses markdown, Arabic uses st.latex via segments
    posterior_desc_en = f"The posterior distribution is $Beta(\\alpha_0 + k, \\beta_0 + n - k) = Beta({posterior_alpha:.1f}, {posterior_beta:.1f})$"
    posterior_desc_ar_segments = [
        {"type": "html", "content": f"<p>التوزيع اللاحق هو Beta({posterior_alpha:.1f}, {posterior_beta:.1f}) = Beta(&alpha;<sub>0</sub> + k, &beta;<sub>0</sub> + n - k)</p>"} 
        # The above uses HTML for subscripts. For proper LaTeX rendering:
        # {"type": "latex", "content": rf"Beta(\alpha_0 + k, \beta_0 + n - k) = Beta({posterior_alpha:.1f}, {posterior_beta:.1f})"}
    ]
    display_text_content(posterior_desc_en, posterior_desc_ar_segments) # You might prefer the latex segment for Arabic here too.

    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta) if (posterior_alpha + posterior_beta) > 0 else 0
    st.write(f"{'المتوسط اللاحق' if current_lang == 'ar' else 'Posterior Mean'}: {posterior_mean:.3f}")
    posterior_ci = get_credible_interval(posterior_alpha, posterior_beta)
    st.write(f"{'فترة الموثوقية ٩٥٪ (اللاحقة)' if current_lang == 'ar' else '95% Credible Interval (Posterior)'}: [{posterior_ci[0]:.3f}, {posterior_ci[1]:.3f}], {'العرض' if current_lang == 'ar' else 'Width'}: {abs(posterior_ci[1]-posterior_ci[0]):.3f}")

    target_width_label_en = "Target Credible Interval Width for Stopping"
    target_width_label_ar = "عرض فترة الموثوقية المستهدف للإيقاف"
    target_width = st.number_input(target_width_label_ar if current_lang == "ar" else target_width_label_en, 0.01, 1.0, 0.10, 0.01, key="target_w_final")
    current_width = abs(posterior_ci[1]-posterior_ci[0]) if posterior_ci[0] is not None and posterior_ci[1] is not None else float('inf')

    # ... (rest of interactive illustration, plots, etc. as in your English version)
    # Ensure plot labels and titles are passed to plot_beta_distribution correctly.
    fig, ax = plt.subplots()
    if prior_alpha > 0 and prior_beta > 0: plot_beta_distribution(prior_alpha, prior_beta, "Prior", "المسبق", ax)
    if posterior_alpha > 0 and posterior_beta > 0: plot_beta_distribution(posterior_alpha, posterior_beta, "Posterior", "اللاحق", ax)
    ax.set_title("Prior and Posterior Distributions" if current_lang == "en" else "التوزيعات المسبقة واللاحقة")
    ax.set_xlabel("Satisfaction Rate (θ)" if current_lang == "en" else "معدل الرضا (θ)")
    ax.set_ylabel("Density" if current_lang == "en" else "الكثافة")
    if (prior_alpha > 0 and prior_beta > 0) or (posterior_alpha > 0 and posterior_beta > 0): ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    # ... (Discounting illustration would also need this bilingual approach for its text)


def conclusion_content():
    display_header("7. Conclusion", "٧. الخلاصة", level=1)
    # ... Populate with your full English markdown and corresponding Arabic segments ...


# --- Streamlit App Structure (Sidebar etc.) ---
selected_lang_option = st.sidebar.selectbox(
    label="Select Language / اختر اللغة",
    options=["English", "العربية"],
    index=0
)
current_lang = "ar" if selected_lang_option == "العربية" else "en"

app_title_en = "Proposal: Adaptive Bayesian Estimation for Pilgrim Satisfaction Surveys" # From your original app
app_title_ar = "مقترح: تقدير Bayesian التكيفي لاستطلاعات رضا الحجاج"

if current_lang == "ar":
    st.markdown(f"<h1 style='text-align: center; direction: rtl;'>{app_title_ar}</h1>", unsafe_allow_html=True)
else:
    st.markdown(f"<h1 style='text-align: center;'>{app_title_en}</h1>", unsafe_allow_html=True)


PAGES_SETUP = { # Using simple keys for PAGES_SETUP for easier lookup
    "Intro": {"en": "1. Introduction & Objectives", "ar": "١. مقدمة وأهداف", "func": introduction_objectives_content},
    "Challenges": {"en": "2. Challenges Addressed", "ar": "٢. التحديات التي تعالجها هذه المنهجية", "func": challenges_addressed_content},
    "Methodology": {"en": "3. Bayesian Adaptive Methodology", "ar": "٣. المنهجية الأساسية: تقدير Bayesian التكيفي", "func": bayesian_adaptive_methodology_content},
    "Roadmap": {"en": "4. Implementation Roadmap", "ar": "٤. خارطة طريق التنفيذ", "func": implementation_roadmap_content},
    "Practitioners": {"en": "5. Note to Practitioners", "ar": "٥. ملاحظة للممارسين", "func": note_to_practitioners_content},
    "Illustration": {"en": "6. Interactive Illustration", "ar": "٦. توضيح تفاعلي", "func": interactive_illustration_content},
    "Conclusion": {"en": "7. Conclusion", "ar": "٧. الخلاصة", "func": conclusion_content}
}

sidebar_display_options = [PAGES_SETUP[key][current_lang] for key in PAGES_SETUP]

st.sidebar.title("Proposal Sections" if current_lang == "en" else "أقسام المقترح")
# The radio button options are now directly the translated display names
selected_page_display_name = st.sidebar.radio(
    label="Go to" if current_lang == "en" else "اذهب إلى", # Label for the radio group itself
    options=sidebar_display_options,
    key=f"nav_radio_main_hybrid_{current_lang}" 
)

# Find the function associated with the selected display name
page_func_to_call = None
for key in PAGES_SETUP:
    if PAGES_SETUP[key][current_lang] == selected_page_display_name:
        page_func_to_call = PAGES_SETUP[key]["func"]
        break

if page_func_to_call:
    page_func_to_call()
else:
    st.error("Page loading error." if current_lang == "en" else "<div dir='rtl'>خطأ في تحميل الصفحة.</div>")

st.sidebar.markdown("---")
st.sidebar.info(
    "This app presents a proposal for using Bayesian adaptive estimation "
    "for Hajj pilgrim satisfaction surveys. Developed by Dr. Mohammad Nabhan."
    if current_lang == "en" else
    "يقدم هذا التطبيق مقترحًا لاستخدام تقدير Bayesian التكيفي لاستطلاعات رضا الحجاج. تم التطوير بواسطة د. محمد نبهان."
)
