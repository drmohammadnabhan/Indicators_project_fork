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
def plot_beta_distribution(alpha, beta, label_en, label_ar, ax):
    """Plots a Beta distribution with internationalized label."""
    label_base = label_ar if current_lang == "ar" else label_en
    # The (α=..., β=...) part will be in LTR within the label
    label = f'{label_base} (α={alpha:.2f}, β={beta:.2f})'
    
    x = np.linspace(0.001, 0.999, 500)
    y = stats.beta.pdf(x, alpha, beta)
    ax.plot(x, y, label=label)
    ax.fill_between(x, y, alpha=0.2)
    ax.set_ylim(bottom=0)


def update_beta_parameters(prior_alpha, prior_beta, successes, failures):
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + failures
    return posterior_alpha, posterior_beta

def get_credible_interval(alpha, beta, conf_level=0.95):
    if alpha <= 0 or beta <= 0:
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
        st.markdown(f"<{header_tag}>{english_text}</{header_tag}>", unsafe_allow_html=True) # English header might have markdown

def display_content(english_markdown_text, arabic_segments_list):
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
            # Add more types if needed, e.g., "markdown_ar" if some Arabic parts use simple markdown without KaTeX issues
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(english_markdown_text)

# --- Proposal Content Functions ---

def introduction_objectives_content():
    display_header("1. Introduction & Objectives", "١. مقدمة وأهداف", level=1)

    english_intro_md = """
This proposal outlines an **Adaptive Bayesian Estimation framework** designed to enhance the process of gathering and analyzing survey data for Hajj pilgrim satisfaction and the assessment of services provided by various companies.

The current practice of developing satisfaction metrics month over month faces complexities, such as delays in pilgrim arrivals or non-uniformity across different months, making it difficult to consistently achieve high-confidence and low-error confidence intervals for key indicators. This proposal aims to introduce a more dynamic, efficient, and robust methodology.
    """
    arabic_intro_segments = [
        {"type": "html", "content": "<p>يحدد هذا المقترح إطارًا لـ <b>تقدير Bayesian التكيفي</b> مصممًا لتعزيز عملية جمع وتحليل بيانات الاستطلاع الخاصة برضا حجاج بيت الله الحرام وتقييم الخدمات المقدمة من مختلف الشركات.</p>"},
        {"type": "html", "content": "<p>تواجه الممارسة الحالية المتمثلة في تطوير مقاييس الرضا شهرًا بعد شهر تعقيدات، مثل التأخير في وصول الحجاج أو عدم الانتظام عبر الأشهر المختلفة، مما يجعل من الصعب تحقيق فترات ثقة عالية الدقة ومنخفضة الخطأ للمؤشرات الرئيسية بشكل مستمر. يهدف هذا المقترح إلى تقديم منهجية أكثر ديناميكية وكفاءة وقوة.</p>"}
    ]
    display_content(english_intro_md, arabic_intro_segments)

    display_header("1.1. Primary Objectives", "١.١. الأهداف الأساسية", level=2)
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
    display_content(english_objectives_md, arabic_objectives_segments)


def challenges_addressed_content():
    display_header("2. Challenges Addressed by this Methodology", "٢. التحديات التي تعالجها هذه المنهجية", level=1)
    english_challenges_md = """
The proposed Bayesian adaptive estimation framework directly addresses several key challenges currently faced in the Hajj survey process:

* **Difficulty in Obtaining Stable Confidence Intervals:**
    * **Challenge:** Operational complexities like staggered pilgrim arrivals, varying visa availability periods, and diverse pilgrim schedules lead to non-uniform data collection across time. This makes it hard to achieve consistent and narrow confidence intervals for satisfaction indicators using fixed sampling plans.
    * **Bayesian Solution:** The adaptive nature allows sampling to continue until a desired precision (credible interval width) is met, regardless of initial data flow irregularities. Estimates stabilize as more data is incorporated.

* **Inefficiency of Fixed Sample Size Approaches:**
    * **Challenge:** Predetermined sample sizes often lead to either over-sampling (wasting resources when satisfaction is homogenous or already precisely estimated) or under-sampling (resulting in inconclusive results or wide confidence intervals).
    * **Bayesian Solution:** Sampling effort is guided by the current level of uncertainty. If an estimate is already precise, sampling can be reduced or stopped for that segment. If it's imprecise, targeted additional sampling is guided by the model.

* **Incorporation of Prior Knowledge and Historical Data:**
    * **Challenge:** Valuable insights from past surveys or existing knowledge about certain pilgrim groups or services are often not formally used to inform current survey efforts or baseline estimates.
    * **Bayesian Solution:** Priors provide a natural mechanism to incorporate such information. This can lead to more accurate estimates, especially when current data is sparse, and can make the learning process more efficient.

* **Assessing Service Provider Performance with Evolving Data:**
    * **Challenge:** Evaluating service providers is difficult when their performance might change over time, or when initial data for a new provider is limited. Deciding when enough data has been collected to make a fair assessment is crucial.
    * **Bayesian Solution:** The framework can be designed to track performance iteratively. For new providers, it starts with less informative priors and builds evidence. For existing ones, it can incorporate past performance, potentially with mechanisms to down-weight older data if performance is expected to evolve (see Section 3.5).

* **Balancing Fresh Data with Historical Insights:**
    * **Challenge:** Determining how much weight to give to historical data versus new, incoming data is critical, especially if there's a possibility of changes in pilgrim sentiment or service quality.
    * **Bayesian Solution:** Techniques like using power priors or dynamic models allow for a tunable "forgetting factor" or learning rate, systematically managing the influence of past data on current estimates.

* **Resource Allocation for Data Collection:**
    * **Challenge:** Allocating limited survey resources (personnel, time, budget) effectively across numerous metrics, pilgrim segments, and service providers.
    * **Bayesian Solution:** The adaptive approach helps prioritize data collection where uncertainty is highest and the need for precision is greatest, leading to more optimal resource allocation.
    """
    arabic_challenges_segments = [
        {"type": "html", "content": "<p>يعالج إطار تقدير Bayesian التكيفي المقترح بشكل مباشر العديد من التحديات الرئيسية التي تواجه حاليًا في عملية استطلاع الحج:</p>"},
        {"type": "html", "content": "<ul>"},
        {"type": "html", "content": "<li><b>صعوبة الحصول على فترات ثقة مستقرة:</b><ul><li><b>التحدي:</b> التعقيدات التشغيلية مثل وصول الحجاج المتدرج، وفترات توفر التأشيرات المتفاوتة، وجداول الحجاج المتنوعة تؤدي إلى جمع بيانات غير منتظم عبر الزمن. هذا يجعل من الصعب تحقيق فترات ثقة متسقة وضيقة لمؤشرات الرضا باستخدام خطط أخذ عينات ثابتة.</li><li><b>حل Bayesian:</b> تسمح الطبيعة التكيفية باستمرار أخذ العينات حتى يتم تحقيق الدقة المطلوبة (عرض فترة الموثوقية)، بغض النظر عن عدم انتظام تدفق البيانات الأولي. تستقر التقديرات مع دمج المزيد من البيانات.</li></ul></li>"},
        {"type": "html", "content": "<li><b>عدم كفاءة مناهج حجم العينة الثابت:</b><ul><li><b>التحدي:</b> غالبًا ما تؤدي أحجام العينات المحددة مسبقًا إما إلى الإفراط في أخذ العينات (إهدار الموارد عندما يكون الرضا متجانسًا أو تم تقديره بدقة بالفعل) أو نقص أخذ العينات (مما يؤدي إلى نتائج غير حاسمة أو فترات ثقة واسعة).</li><li><b>حل Bayesian:</b> يتم توجيه جهد أخذ العينات حسب المستوى الحالي من عدم اليقين. إذا كان التقدير دقيقًا بالفعل، فيمكن تقليل أخذ العينات أو إيقافه لتلك الشريحة. إذا كان غير دقيق، يتم توجيه أخذ عينات إضافية مستهدفة بواسطة النموذج.</li></ul></li>"},
        {"type": "html", "content": "<li><b>دمج المعرفة المسبقة والبيانات التاريخية:</b><ul><li><b>التحدي:</b> غالبًا لا يتم استخدام الرؤى القيمة من الاستطلاعات السابقة أو المعرفة الحالية حول مجموعات معينة من الحجاج أو الخدمات رسميًا لإبلاغ جهود الاستطلاع الحالية أو التقديرات الأساسية.</li><li><b>حل Bayesian:</b> توفر التوزيعات المسبقة (Priors) آلية طبيعية لدمج هذه المعلومات. يمكن أن يؤدي ذلك إلى تقديرات أكثر دقة، خاصة عندما تكون البيانات الحالية متفرقة، ويمكن أن يجعل عملية التعلم أكثر كفاءة.</li></ul></li>"},
        {"type": "html", "content": "<li><b>تقييم أداء مقدمي الخدمات مع تطور البيانات:</b><ul><li><b>التحدي:</b> يصعب تقييم مقدمي الخدمات عندما قد يتغير أداؤهم بمرور الوقت، أو عندما تكون البيانات الأولية لمقدم خدمة جديد محدودة. يعد تحديد متى تم جمع بيانات كافية لإجراء تقييم عادل أمرًا بالغ الأهمية.</li><li><b>حل Bayesian:</b> يمكن تصميم الإطار لتتبع الأداء بشكل متكرر. بالنسبة لمقدمي الخدمات الجدد، يبدأ بتوزيعات مسبقة أقل إفادة ويبني الأدلة. بالنسبة للموجودين، يمكنه دمج الأداء السابق، مع آليات محتملة لتقليل وزن البيانات القديمة إذا كان من المتوقع أن يتطور الأداء (انظر القسم ٣.٥).</li></ul></li>"},
        {"type": "html", "content": "<li><b>الموازنة بين البيانات الجديدة والرؤى التاريخية:</b><ul><li><b>التحدي:</b> يعد تحديد مقدار الوزن الذي يجب إعطاؤه للبيانات التاريخية مقابل البيانات الجديدة الواردة أمرًا بالغ الأهمية، خاصةً إذا كان هناك احتمال حدوث تغييرات في شعور الحجاج أو جودة الخدمة.</li><li><b>حل Bayesian:</b> تسمح التقنيات مثل استخدام "power priors" أو النماذج الديناميكية بعامل "نسيان" قابل للضبط أو معدل تعلم، مما يدير بشكل منهجي تأثير البيانات السابقة على التقديرات الحالية.</li></ul></li>"},
        {"type": "html", "content": "<li><b>تخصيص الموارد لجمع البيانات:</b><ul><li><b>التحدي:</b> تخصيص موارد المسح المحدودة (الأفراد، الوقت، الميزانية) بشكل فعال عبر العديد من المقاييس وشرائح الحجاج ومقدمي الخدمات.</li><li><b>حل Bayesian:</b> يساعد النهج التكيفي في تحديد أولويات جمع البيانات حيث يكون عدم اليقين هو الأعلى والحاجة إلى الدقة هي الأكبر، مما يؤدي إلى تخصيص أمثل للموارد.</li></ul></li>"},
        {"type": "html", "content": "</ul>"}
        # You'll need to translate the sub-bullets under "Why Not Just Combine All Data" if you had that section.
    ]
    display_content(english_challenges_md, arabic_challenges_segments)


def bayesian_adaptive_methodology_content():
    display_header("3. Core Methodology: Bayesian Adaptive Estimation", "٣. المنهجية الأساسية: تقدير Bayesian التكيفي", level=1)
    
    english_methodology_intro_md = """
The Bayesian adaptive estimation framework is an iterative process that leverages Bayes' theorem to update our beliefs about pilgrim satisfaction or service performance as new survey data is collected. This allows for dynamic adjustments to the sampling strategy.
    """
    arabic_methodology_intro_segments = [
        {"type": "html", "content": "<p>إطار تقدير Bayesian التكيفي هو عملية تكرارية تستفيد من نظرية Bayes لتحديث معتقداتنا حول رضا الحجاج أو أداء الخدمة عند جمع بيانات استطلاع جديدة. هذا يسمح بإجراء تعديلات ديناميكية على استراتيجية أخذ العينات.</p>"}
    ]
    display_content(english_methodology_intro_md, arabic_methodology_intro_segments)

    display_header("3.1. Fundamental Concepts", "٣.١. المفاهيم الأساسية", level=2)
    english_concepts_md = r"""
At its heart, Bayesian inference combines prior knowledge with observed data to arrive at an updated understanding, known as the posterior distribution.

* **Prior Distribution ($P(\theta)$):** This represents our initial belief about a parameter $\theta$ (e.g., the proportion of satisfied pilgrims) *before* observing new data. It can be based on historical data, expert opinion, or be deliberately "uninformative" if we want the data to speak for itself.
* **Likelihood ($P(D|\theta)$):** This quantifies how probable the observed data ($D$) is, given a particular value of the parameter $\theta$. It is the function that connects the data to the parameter.
* **Posterior Distribution ($P(\theta|D)$):** This is our updated belief about $\theta$ *after* observing the data. It is calculated using Bayes' Theorem:
    $$ P(\theta|D) = \frac{P(D|\theta) \times P(\theta)}{P(D)} $$
    Where $P(D)$ is the marginal likelihood of the data, acting as a normalizing constant. In practice, we often focus on the proportionality:
    $$ P(\theta|D) \propto P(D|\theta) \times P(\theta) $$
* **Credible Interval:** In Bayesian statistics, a credible interval is a range of values that contains the parameter $\theta$ with a certain probability (e.g., 95%). This is a direct probabilistic statement about the parameter, unlike the frequentist confidence interval.
    """
    arabic_concepts_segments = [
        {"type": "html", "content": "<p>في جوهره، يجمع استدلال Bayesian بين المعرفة المسبقة والبيانات المرصودة للوصول إلى فهم محدث، يُعرف بالتوزيع اللاحق.</p>"},
        {"type": "html", "content": "<ul>"},
        {"type": "html", "content": "<li><b>التوزيع المسبق ($P(\\theta)$):</b> يمثل هذا اعتقادنا الأولي حول معلمة &theta; (ثيتا) (على سبيل المثال، نسبة الحجاج الراضين) <i>قبل</i> ملاحظة البيانات الجديدة. يمكن أن يستند إلى البيانات التاريخية، أو رأي الخبراء، أو أن يكون \"غير إعلامي\" بشكل متعمد إذا أردنا أن تتحدث البيانات عن نفسها.</li>"},
        {"type": "html", "content": "<li><b>دالة الإمكان ($P(D|\\theta)$):</b> تحدد هذه الدالة مدى احتمالية البيانات المرصودة ($D$)، بالنظر إلى قيمة معينة للمعلومة &theta;. إنها الدالة التي تربط البيانات بالمعلمة.</li>"},
        {"type": "html", "content": "<li><b>التوزيع اللاحق ($P(\\theta|D)$):</b> هذا هو اعتقادنا المحدث حول &theta; <i>بعد</i> ملاحظة البيانات. يتم حسابه باستخدام نظرية Bayes:</li>"},
        {"type": "latex", "content": r"P(\theta|D) = \frac{P(D|\theta) P(\theta)}{P(D)}"},
        {"type": "html", "content": "<li>حيث $P(D)$ هو الإمكان الهامشي للبيانات، ويعمل كثابت تسوية. في الممارسة العملية، غالبًا ما نركز على التناسب:</li>"},
        {"type": "latex", "content": r"P(\theta|D) \propto P(D|\theta) P(\theta)"},
        {"type": "html", "content": "<li><b>فترة الموثوقية:</b> في إحصاءات Bayesian، فترة الموثوقية هي نطاق من القيم يحتوي على المعلمة &theta; باحتمال معين (على سبيل المثال، ٩٥٪). هذا بيان احتمالي مباشر حول المعلمة، على عكس فترة الثقة الإحصائية التقليدية.</li>"},
        {"type": "html", "content": "</ul>"}
    ]
    display_content(english_concepts_md, arabic_concepts_segments)

    display_header("3.2. The Iterative Process", "٣.٢. العملية التكرارية", level=2)
    english_iterative_md = """
The adaptive methodology follows these steps:
1.  **Initialization:**
    * Define the parameter(s) of interest (e.g., satisfaction with lodging, food, logistics for a specific company).
    * Specify an initial **prior distribution** for each parameter. For satisfaction proportions, a Beta distribution is commonly used.
    * Set a target precision (e.g., a maximum width for the 95% credible interval).

2.  **Initial Data Collection:**
    * Collect an initial batch of survey responses relevant to the parameter(s). The size of this initial batch can be based on practical considerations or a small fixed number.

3.  **Posterior Update:**
    * Use the collected data (likelihood) and the current prior distribution to calculate the **posterior distribution** for each parameter.

4.  **Precision Assessment:**
    * Calculate the credible interval from the posterior distribution.
    * Compare the width of this interval to the target precision.

5.  **Adaptive Decision & Iteration:**
    * **If Target Precision Met:** For the given parameter, the current level of precision is sufficient. Sampling for this specific indicator/segment can be paused or stopped. The current posterior distribution provides the estimate and its uncertainty.
    * **If Target Precision Not Met:** More data is needed.
        * Determine an appropriate additional sample size. This can be guided by projecting how the credible interval width might decrease with more data (based on the current posterior).
        * Collect the additional batch of survey responses.
        * Return to Step 3 (Posterior Update), using the current posterior as the new prior for the next update.

This cycle continues until the desired precision is achieved for all key indicators or available resources for the current wave are exhausted.
    """
    arabic_iterative_segments = [
        {"type": "html", "content": "<p>تتبع المنهجية التكيفية هذه الخطوات:</p><ol>"},
        {"type": "html", "content": "<li><b>التهيئة:</b><ul><li>تحديد المعلمة (المعلمات) ذات الأهمية (على سبيل المثال، الرضا عن السكن، الطعام، الخدمات اللوجستية لشركة معينة).</li><li>تحديد <b>توزيع مسبق</b> أولي لكل معلمة. بالنسبة لنسب الرضا، يشيع استخدام توزيع Beta.</li><li>تحديد دقة مستهدفة (على سبيل المثال، أقصى عرض لفترة موثوقية بنسبة ٩٥٪).</li></ul></li>"},
        {"type": "html", "content": "<li><b>جمع البيانات الأولي:</b><ul><li>جمع دفعة أولية من ردود الاستطلاع ذات الصلة بالمعلمة (المعلمات). يمكن أن يعتمد حجم هذه الدفعة الأولية على اعتبارات عملية أو عدد ثابت صغير.</li></ul></li>"},
        {"type": "html", "content": "<li><b>تحديث التوزيع اللاحق:</b><ul><li>استخدام البيانات المجمعة (دالة الإمكان) والتوزيع المسبق الحالي لحساب <b>التوزيع اللاحق</b> لكل معلمة.</li></ul></li>"},
        {"type": "html", "content": "<li><b>تقييم الدقة:</b><ul><li>حساب فترة الموثوقية من التوزيع اللاحق.</li><li>مقارنة عرض هذه الفترة بالدقة المستهدفة.</li></ul></li>"},
        {"type": "html", "content": "<li><b>القرار التكيفي والتكرار:</b><ul><li><b>إذا تم تحقيق الدقة المستهدفة:</b> بالنسبة للمعلمة المحددة، يكون المستوى الحالي من الدقة كافيًا. يمكن إيقاف أخذ العينات مؤقتًا أو إيقافه لهذا المؤشر/الشريحة المحددة. يوفر التوزيع اللاحق الحالي التقدير وعدم اليقين فيه.</li><li><b>إذا لم يتم تحقيق الدقة المستهدفة:</b> هناك حاجة إلى مزيد من البيانات.<ul><li>تحديد حجم عينة إضافي مناسب. يمكن توجيه ذلك من خلال توقع كيف قد ينخفض عرض فترة الموثوقية مع المزيد من البيانات (بناءً على التوزيع اللاحق الحالي).</li><li>جمع الدفعة الإضافية من ردود الاستطلاع.</li><li>العودة إلى الخطوة ٣ (تحديث التوزيع اللاحق)، باستخدام التوزيع اللاحق الحالي كتوزيع مسبق جديد للتحديث التالي.</li></ul></li></ul></li>"},
        {"type": "html", "content": "</ol><p>تستمر هذه الدورة حتى يتم تحقيق الدقة المطلوبة لجميع المؤشرات الرئيسية أو استنفاد الموارد المتاحة للموجة الحالية.</p>"}
    ]
    display_content(english_iterative_md, arabic_iterative_segments)
    
    # Remove or replace st.image - for now, it's removed.
    # If you have a local image:
    # try:
    #     st.image("path_to_your_local_image.png", caption="Conceptual Flow...")
    # except FileNotFoundError:
    #     st.warning("Local image for Bayesian flow not found.")


    display_header("3.3. Modeling Satisfaction (e.g., using Beta-Binomial Model)", "٣.٣. نمذجة الرضا (على سبيل المثال، باستخدام نموذج Beta-Binomial)", level=2)
    english_modeling_md = r"""
For satisfaction metrics that are proportions (e.g., percentage of pilgrims rating a service as "satisfied" or "highly satisfied"), the Beta-Binomial model is highly suitable and commonly used.

* **Parameter of Interest ($\theta$):** The true underlying proportion of satisfied pilgrims.
* **Prior Distribution (Beta):** We assume the prior belief about $\theta$ follows a Beta distribution, denoted as $Beta(\alpha_0, \beta_0)$.
    * $\alpha_0 > 0$ and $\beta_0 > 0$ are the parameters of the prior.
    * An uninformative prior could be $Beta(1, 1)$, which is equivalent to a Uniform(0,1) distribution.
    * Prior knowledge can be incorporated by setting $\alpha_0$ and $\beta_0$ based on historical data (e.g., $\alpha_0$ = past successes, $\beta_0$ = past failures).
* **Likelihood (Binomial/Bernoulli):** If we collect $n$ new responses, and $k$ of them are "satisfied" (successes), the likelihood of observing $k$ successes in $n$ trials is given by the Binomial distribution:
    $$ P(k, n | \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k} $$
* **Posterior Distribution (Beta):** Due to the conjugacy between the Beta prior and Binomial likelihood, the posterior distribution of $\theta$ is also a Beta distribution:
    $$ \theta | k, n \sim Beta(\alpha_0 + k, \beta_0 + n - k) $$
    So, the updated parameters are $\alpha_{post} = \alpha_0 + k$ and $\beta_{post} = \beta_0 + n - k$.
    The mean of this posterior distribution, often used as the point estimate for satisfaction, is $\frac{\alpha_{post}}{\alpha_{post} + \beta_{post}}$.

This conjugacy simplifies calculations significantly.
    """
    arabic_modeling_segments = [
        {"type": "html", "content": "<p>بالنسبة لمقاييس الرضا التي هي عبارة عن نسب (على سبيل المثال، النسبة المئوية للحجاج الذين يقيمون خدمة بأنها \"مرضية\" أو \"مرضية للغاية\")، فإن نموذج Beta-Binomial مناسب جدًا ويستخدم بشكل شائع.</p>"},
        {"type": "html", "content": "<ul>"},
        {"type": "html", "content": "<li><b>المعلمة ذات الأهمية (&theta;):</b> النسبة الأساسية الحقيقية للحجاج الراضين.</li>"},
        {"type": "html", "content": "<li><b>التوزيع المسبق (Beta):</b> نفترض أن الاعتقاد المسبق حول &theta; يتبع توزيع Beta، يُشار إليه بالرمز $Beta(&alpha;_0, &beta;_0)$.<ul><li>&alpha;<sub>0</sub> > 0 و &beta;<sub>0</sub> > 0 هما معلمات التوزيع المسبق.</li><li>يمكن أن يكون التوزيع المسبق غير الإعلامي $Beta(1, 1)$، وهو ما يعادل توزيع Uniform(0,1).</li><li>يمكن دمج المعرفة المسبقة عن طريق تعيين &alpha;<sub>0</sub> و &beta;<sub>0</sub> بناءً على البيانات التاريخية (على سبيل المثال، &alpha;<sub>0</sub> = النجاحات السابقة، &beta;<sub>0</sub> = الإخفاقات السابقة).</li></ul></li>"},
        {"type": "html", "content": "<li><b>دالة الإمكان (Binomial/Bernoulli):</b> إذا جمعنا $n$ ردودًا جديدة، وكان $k$ منها \"راضين\" (نجاحات)، فإن إمكانية ملاحظة $k$ نجاحات في $n$ محاولات تُعطى بواسطة توزيع Binomial:</li>"},
        {"type": "latex", "content": r"P(k, n | \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k}"},
        {"type": "html", "content": "<li><b>التوزيع اللاحق (Beta):</b> نظرًا للترافق بين توزيع Beta المسبق ودالة الإمكان Binomial، فإن التوزيع اللاحق لـ &theta; هو أيضًا توزيع Beta:</li>"},
        {"type": "latex", "content": r"\theta | k, n \sim Beta(\alpha_0 + k, \beta_0 + n - k)"},
        {"type": "html", "content": "<li>لذا، فإن المعلمات المحدثة هي &alpha;<sub>post</sub> = &alpha;<sub>0</sub> + k و &beta;<sub>post</sub> = &beta;<sub>0</sub> + n - k.</li>"},
        {"type": "html", "content": "<li>متوسط هذا التوزيع اللاحق، الذي غالبًا ما يستخدم كتقدير نقطي للرضا، هو:</li>"},
        {"type": "latex", "content": r"\frac{\alpha_{post}}{\alpha_{post} + \beta_{post}}"},
        {"type": "html", "content": "</ul>"},
        {"type": "html", "content": "<p>هذا الترافق يبسط الحسابات بشكل كبير.</p>"}
    ]
    display_content(english_modeling_md, arabic_modeling_segments)

    display_header("3.4. Adaptive Sampling Logic & Determining Additional Sample Size", "٣.٤. منطق أخذ العينات التكيفي وتحديد حجم العينة الإضافي", level=2)
    english_sampling_md = r"""
The decision to continue sampling is based on whether the current credible interval for $\theta$ meets the desired precision.

* **Stopping Rule:** Stop sampling for a specific metric when (for a $(1-\gamma)\%$ credible interval $[L, U]$):
    $$ U - L \leq \text{Target Width} $$
    And/or when the credible interval lies entirely above/below a certain threshold of practical importance.

* **Estimating Required Additional Sample Size (Conceptual):**
    While exact formulas for sample size ... (rest of the English content from your app)
    """ # Truncated for brevity, use your full English content
    arabic_sampling_segments = [
        {"type": "html", "content": "<p>يعتمد قرار مواصلة أخذ العينات على ما إذا كانت فترة الموثوقية الحالية لـ &theta; تفي بالدقة المطلوبة.</p>"},
        {"type": "html", "content": "<ul><li><b>قاعدة الإيقاف:</b> إيقاف أخذ العينات لمقياس معين عندما (لفترة موثوقية $(1-&gamma;)\%$ $[L, U]$):</li></ul>"},
        {"type": "latex", "content": r"U - L \leq \text{Target Width}"},
        {"type": "html", "content": "<p>و/أو عندما تقع فترة الموثوقية بالكامل فوق/تحت عتبة معينة ذات أهمية عملية.</p>"},
        {"type": "html", "content": "<ul><li><b>تقدير حجم العينة الإضافي المطلوب (مفاهيمي):</b></li></ul>"},
        {"type": "html", "content": "<p>في حين أن الصيغ الدقيقة لحجم العينة... (أكمل الترجمة والنصوص المقابلة)</p>"}
    ]
    display_content(english_sampling_md, arabic_sampling_segments)

    display_header("3.5. Handling Data Heterogeneity Over Time", "٣.٥. التعامل مع عدم تجانس البيانات بمرور الوقت", level=2)
    english_heterogeneity_md = """
A key challenge is that service provider performance or general pilgrim satisfaction might change over time...
$$ \alpha_{prior, t+1} = \delta \times \alpha_t + (1-\delta) \times \alpha_{initial} $$
(rest of the English content from your app)
    """ # Truncated, use your full English content
    arabic_heterogeneity_segments = [
        {"type": "html", "content": "<p>التحدي الرئيسي هو أن أداء مقدمي الخدمات أو رضا الحجاج العام قد يتغير بمرور الوقت...</p>"},
        {"type": "latex", "content": r"\alpha_{prior, t+1} = \delta \times \alpha_t + (1-\delta) \times \alpha_{initial}"},
        {"type": "html", "content": "<p>(أكمل الترجمة والنصوص المقابلة)</p>"}
    ]
    display_content(english_heterogeneity_md, arabic_heterogeneity_segments)


def implementation_roadmap_content():
    display_header("4. Implementation Roadmap (Conceptual)", "٤. خارطة طريق التنفيذ (مفاهيمية)", level=1)
    english_roadmap_intro = "Implementing the Bayesian adaptive estimation framework involves several key stages:"
    arabic_roadmap_intro_segments = [{"type": "html", "content": "<p>يتضمن تنفيذ إطار تقدير Bayesian التكيفي عدة مراحل رئيسية:</p>"}]
    display_content(english_roadmap_intro, arabic_roadmap_intro_segments)

    df_roadmap_en = pd.DataFrame({ # From your English App
        "Phase": ["Phase 1: Foundation & Pilot", "Phase 1: Foundation & Pilot", "Phase 2: Iterative Development & Testing", "Phase 2: Iterative Development & Testing", "Phase 3: Full-Scale Deployment & Refinement", "Phase 3: Full-Scale Deployment & Refinement"],
        "Step": ["1. Define Key Metrics & Precision Targets", "2. System Setup & Prior Elicitation", "3. Model Development & Initial Batching Logic", "4. Dashboard Development & Pilot Testing", "5. Scaled Rollout & Heterogeneity Modeling", "6. Continuous Monitoring & Improvement"],
        "Description": ["Identify critical satisfaction indicators... $Beta(1,1)$ ...", "Establish data collection pathways... $Beta(1,1)$ ...", "Develop the Bayesian models...", "Create a dashboard...", "Gradually roll out the adaptive system...", "Continuously monitor the system's performance..."]
    })
    # Create translated df_roadmap_ar - KaTeX in cells for English df_roadmap_en will render with st.dataframe
    # For Arabic df_roadmap_ar, if using to_html(escape=False), use HTML entities or avoid complex math in table.
    df_roadmap_ar_data = {
        "المرحلة": ["المرحلة ١", "المرحلة ١", "المرحلة ٢", "المرحلة ٢", "المرحلة ٣", "المرحلة ٣"],
        "الخطوة": ["١. تحديد المقاييس", "٢. إعداد النظام", "٣. تطوير النموذج", "٤. تطوير لوحة التحكم", "٥. التوسع", "٦. المراقبة والتحسين"],
        "الوصف": ["الوصف العربي للمرحلة ١، الخطوة ١. (مثال: Beta(1,1) )", "الوصف العربي ٢", "الوصف العربي ٣", "الوصف العربي ٤", "الوصف العربي ٥", "الوصف العربي ٦"]
    }
    df_roadmap_ar = pd.DataFrame(df_roadmap_ar_data)

    if current_lang == "ar":
        st.markdown(f"<div dir='rtl'>{df_roadmap_ar.to_html(escape=False, index=False, classes='dataframe')}</div>", unsafe_allow_html=True)
    else:
        st.dataframe(df_roadmap_en, hide_index=True, use_container_width=True)


def note_to_practitioners_content():
    display_header("5. Note to Practitioners", "٥. ملاحظة للممارسين", level=1)

    display_header("5.1. Benefits of the Bayesian Adaptive Approach", "٥.١. فوائد نهج Bayesian التكيفي", level=2)
    english_benefits_md = """
* **Efficiency:** Targets sampling effort where it's most needed...
* **Adaptability:** Responds to incoming data...
* **Formal Use of Prior Knowledge:** Allows systematic incorporation of historical data...
* **Intuitive Uncertainty Quantification:** Credible intervals offer a direct probabilistic interpretation...
* **Rich Output:** Provides a full posterior distribution...
    """ # Use your full English content
    arabic_benefits_segments = [
        {"type": "html", "content": "<ul><li><b>الكفاءة:</b> يستهدف جهد أخذ العينات حيث تشتد الحاجة إليه...</li><li><b>القدرة على التكيف:</b> يستجيب للبيانات الواردة...</li><li><b>الاستخدام الرسمي للمعرفة المسبقة:</b> يسمح بالدمج المنهجي للبيانات التاريخية...</li><li><b>قياس عدم اليقين بشكل بديهي:</b> توفر فترات الموثوقية تفسيرًا احتماليًا مباشرًا...</li><li><b>مخرجات غنية:</b> يوفر توزيعًا لاحقًا كاملاً...</li></ul>"}
    ]
    display_content(english_benefits_md, arabic_benefits_segments)

    display_header("5.2. Limitations and Considerations", "٥.٢. القيود والاعتبارات", level=2)
    english_limitations_md = """
* **Complexity:** Bayesian methods can be conceptually more demanding...
* **Prior Selection:** The choice of prior distribution can influence posterior results...
* **Computational Cost:** While Beta-Binomial models are computationally simple...
* **Interpretation Differences:** Practitioners familiar with frequentist statistics...
* **"Black Box" Perception:** If not explained clearly...
    """ # Use your full English content
    arabic_limitations_segments = [
        {"type": "html", "content": "<ul><li><b>التعقيد:</b> يمكن أن تكون طرق Bayesian أكثر تطلبًا من الناحية المفاهيمية...</li><li><b>اختيار التوزيع المسبق:</b> يمكن أن يؤثر اختيار التوزيع المسبق على النتائج اللاحقة...</li><li><b>التكلفة الحسابية:</b> بينما تكون نماذج Beta-Binomial بسيطة حسابيًا...</li><li><b>اختلافات التفسير:</b> يحتاج الممارسون المطلعون على الإحصاءات التقليدية...</li><li><b>تصور "الصندوق الأسود":</b> إذا لم يتم شرحها بوضوح...</li></ul>"}
    ]
    display_content(english_limitations_md, arabic_limitations_segments)

    display_header("5.3. Key Assumptions", "٥.٣. الافتراضات الرئيسية", level=2)
    english_assumptions_md = """
* **Representativeness of Samples:** Each batch of collected data is assumed...
* **Model Appropriateness:** The chosen likelihood and prior distributions should...
* **Stability (or Modeled Change):** The underlying parameter being measured...
* **Accurate Data:** Assumes responses are truthful...
    """ # Use your full English content
    arabic_assumptions_segments = [
        {"type": "html", "content": "<ul><li><b>تمثيلية العينات:</b> يُفترض أن كل دفعة من البيانات المجمعة...</li><li><b>ملاءمة النموذج:</b> يجب أن تعكس دالة الإمكان والتوزيعات المسبقة المختارة...</li><li><b>الاستقرار (أو التغيير المنمذج):</b> يُفترض أن المعلمة الأساسية التي يتم قياسها...</li><li><b>بيانات دقيقة:</b> يفترض أن الردود صادقة...</li></ul>"}
    ]
    display_content(english_assumptions_md, arabic_assumptions_segments)

    display_header("5.4. Practical Recommendations", "٥.٤. توصيات عملية", level=2)
    english_recommendations_md = """
* **Start Simple:** Begin with core satisfaction metrics...
* **Invest in Training:** Ensure that the team involved...
* **Transparency is Key:** Document choices for priors...
* **Regular Review and Validation:** Periodically review the performance...
* **Stakeholder Communication:** Develop clear ways to communicate...
* **Pilot Thoroughly:** Before full-scale implementation...
    """ # Use your full English content
    arabic_recommendations_segments = [
        {"type": "html", "content": "<ul><li><b>ابدأ ببساطة:</b> ابدأ بمقاييس الرضا الأساسية...</li><li><b>استثمر في التدريب:</b> تأكد من أن الفريق المشارك...</li><li><b>الشفافية هي المفتاح:</b> قم بتوثيق الخيارات الخاصة بالتوزيعات المسبقة...</li><li><b>المراجعة والتحقق المنتظم:</b> قم بمراجعة أداء النماذج بشكل دوري...</li><li><b>التواصل مع أصحاب المصلحة:</b> طور طرقًا واضحة لتوصيل...</li><li><b>التجربة الأولية الشاملة:</b> قبل التنفيذ على نطاق واسع...</li></ul>"}
    ]
    display_content(english_recommendations_md, arabic_recommendations_segments)


def interactive_illustration_content():
    display_header("6. Interactive Illustration: Beta-Binomial Model", "٦. توضيح تفاعلي: نموذج Beta-Binomial", level=1)
    
    english_interactive_intro = """
This section provides a simple interactive illustration of how a Beta prior is updated to a Beta posterior with new data (Binomial likelihood). This is the core of estimating a proportion (e.g., satisfaction rate) in a Bayesian way.
    """
    arabic_interactive_intro_segments = [
        {"type": "html", "content": "<p>يقدم هذا القسم توضيحًا تفاعليًا بسيطًا لكيفية تحديث توزيع Beta المسبق إلى توزيع Beta لاحق ببيانات جديدة (دالة الإمكان Binomial). هذا هو جوهر تقدير نسبة (على سبيل المثال، معدل الرضا) بطريقة Bayesian.</p>"}
    ]
    display_content(english_interactive_intro, arabic_interactive_intro_segments)

    st.markdown("---")
    col1, col2 = st.columns(2)

    # Create English and Arabic labels for widgets
    prior_alpha_label_en = "Prior Alpha (α₀)"
    prior_alpha_label_ar = "ألفا المسبقة (α₀)"
    prior_beta_label_en = "Prior Beta (β₀)"
    prior_beta_label_ar = "بيتا المسبقة (β₀)"
    # ... (and so on for all widget labels and text used in this section)

    with col1:
        display_header("Prior Beliefs", "المعتقدات المسبقة", level=2)
        english_prior_desc = "The Beta distribution $Beta(\\alpha, \\beta)$ is a common prior for proportions. $\\alpha$ can be thought of as prior 'successes' and $\\beta$ as prior 'failures'. $Beta(1,1)$ is a uniform (uninformative) prior."
        arabic_prior_desc_segments = [
            {"type": "html", "content":"<p>توزيع Beta $Beta(&alpha;, &beta;)$ هو توزيع مسبق شائع للنسب. يمكن اعتبار &alpha; بمثابة 'نجاحات' مسبقة و &beta; بمثابة 'إخفاقات' مسبقة. $Beta(1,1)$ هو توزيع منتظم (غير إعلامي) مسبق.</p>"}
        ]
        display_content(english_prior_desc, arabic_prior_desc_segments)
        
        prior_alpha = st.slider(prior_alpha_label_ar if current_lang == "ar" else prior_alpha_label_en, 0.1, 50.0, 1.0, 0.1, key="ia_prior_a_final_v2")
        prior_beta = st.slider(prior_beta_label_ar if current_lang == "ar" else prior_beta_label_en, 0.1, 50.0, 1.0, 0.1, key="ia_prior_b_final_v2")
        
        prior_mean = prior_alpha / (prior_alpha + prior_beta) if (prior_alpha + prior_beta) > 0 else 0
        st.write(f"{'المتوسط المسبق' if current_lang == 'ar' else 'Prior Mean'}: {prior_mean:.3f}")
        
        prior_ci = get_credible_interval(prior_alpha, prior_beta)
        st.write(f"{'فترة الموثوقية ٩٥٪ (المسبقة)' if current_lang == 'ar' else '95% Credible Interval (Prior)'}: [{prior_ci[0]:.3f}, {prior_ci[1]:.3f}], {'العرض' if current_lang == 'ar' else 'Width'}: {abs(prior_ci[1]-prior_ci[0]):.3f}")

    with col2:
        display_header("New Survey Data (Likelihood)", "بيانات الاستطلاع الجديدة (دالة الإمكان)", level=2)
        display_content("Enter the results from a new batch of surveys.", [{"type": "html", "content": "<p>أدخل النتائج من دفعة جديدة من الاستطلاعات.</p>"}])
        
        num_surveys_label_en = "Number of New Surveys (n)"
        num_surveys_label_ar = "عدد الاستطلاعات الجديدة (n)"
        num_surveys = st.slider(num_surveys_label_ar if current_lang == "ar" else num_surveys_label_en, 1, 500, 50, 1, key="ia_n_final_v2")

        num_satisfied_label_en = "Number Satisfied in New Surveys (k)"
        num_satisfied_label_ar = "عدد الراضين في الاستطلاعات الجديدة (k)"
        num_satisfied = st.slider(num_satisfied_label_ar if current_lang == "ar" else num_satisfied_label_en, 0, num_surveys, int(num_surveys/2), 1, key="ia_k_final_v2")
        
        st.write(f"{'الرضا الملاحظ في البيانات الجديدة' if current_lang == 'ar' else 'Observed Satisfaction in New Data'}: {num_satisfied/num_surveys if num_surveys > 0 else 0:.3f}")

    st.markdown("---")
    display_header("Posterior Beliefs (After Update)", "المعتقدات اللاحقة (بعد التحديث)", level=2)
    posterior_alpha, posterior_beta = update_beta_parameters(prior_alpha, prior_beta, num_satisfied, num_surveys - num_satisfied)
    
    posterior_desc_en = f"The posterior distribution is $Beta(\\alpha_0 + k, \\beta_0 + n - k) = Beta({posterior_alpha:.1f}, {posterior_beta:.1f})$"
    posterior_desc_ar_segments = [
        {"type": "html", "content": "<p>التوزيع اللاحق هو (يمكن استخدام &alpha; و &beta; هنا):</p>"},
        {"type": "latex", "content": rf"Beta(\alpha_0 + k, \beta_0 + n - k) = Beta({posterior_alpha:.1f}, {posterior_beta:.1f})"}
    ]
    display_content(posterior_desc_en, posterior_desc_ar_segments)

    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta) if (posterior_alpha + posterior_beta) > 0 else 0
    st.write(f"{'المتوسط اللاحق' if current_lang == 'ar' else 'Posterior Mean'}: {posterior_mean:.3f}")
    posterior_ci = get_credible_interval(posterior_alpha, posterior_beta)
    st.write(f"{'فترة الموثوقية ٩٥٪ (اللاحقة)' if current_lang == 'ar' else '95% Credible Interval (Posterior)'}: [{posterior_ci[0]:.3f}, {posterior_ci[1]:.3f}], {'العرض' if current_lang == 'ar' else 'Width'}: {abs(posterior_ci[1]-posterior_ci[0]):.3f}")

    target_width_label_en = "Target Credible Interval Width for Stopping"
    target_width_label_ar = "عرض فترة الموثوقية المستهدف للإيقاف"
    target_width = st.number_input(target_width_label_ar if current_lang == "ar" else target_width_label_en, 0.01, 1.0, 0.10, 0.01, key="target_w_final_v2")
    current_width = abs(posterior_ci[1]-posterior_ci[0]) if posterior_ci[0] is not None and posterior_ci[1] is not None else float('inf')

    if current_width <= target_width and (posterior_alpha > 0 or posterior_beta > 0) :
        st.success(f"{'تم تحقيق الدقة المستهدفة!' if current_lang == 'ar' else 'Target precision met!'} {'العرض الحالي' if current_lang == 'ar' else 'Current width'} ({current_width:.3f}) ≤ {'العرض المستهدف' if current_lang == 'ar' else 'Target width'} ({target_width:.3f}).")
    else:
        st.warning(f"{'لم يتم تحقيق الدقة المستهدفة بعد.' if current_lang == 'ar' else 'Target precision not yet met.'} {'العرض الحالي' if current_lang == 'ar' else 'Current width'} ({current_width:.3f}) > {'العرض المستهدف' if current_lang == 'ar' else 'Target width'} ({target_width:.3f}). {'ضع في اعتبارك المزيد من العينات.' if current_lang == 'ar' else 'Consider more samples.'}")

    fig, ax = plt.subplots()
    if prior_alpha > 0 and prior_beta > 0: plot_beta_distribution(prior_alpha, prior_beta, "Prior", "المسبق", ax)
    if posterior_alpha > 0 and posterior_beta > 0: plot_beta_distribution(posterior_alpha, posterior_beta, "Posterior", "اللاحق", ax)
    ax.set_title("Prior and Posterior Distributions of Satisfaction Rate" if current_lang == "en" else "التوزيعات المسبقة واللاحقة لمعدل الرضا")
    ax.set_xlabel("Satisfaction Rate (θ)" if current_lang == "en" else "معدل الرضا (θ)")
    ax.set_ylabel("Density" if current_lang == "en" else "الكثافة")
    if (prior_alpha > 0 and prior_beta > 0) or (posterior_alpha > 0 and posterior_beta > 0): ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("---")
    display_header("Conceptual Illustration: Impact of Discounting Older Data", "توضيح مفاهيمي: تأثير خصم البيانات القديمة", level=2)
    english_discount_desc = """
This illustrates how a discount factor might change the influence of 'old' posterior data when it's used to form a prior for a 'new' period.
Assume the 'Posterior' calculated above is now 'Old Data' from a previous period.
We want to form a new prior for the upcoming period.
An 'Initial Prior' (e.g., $Beta(1,1)$) represents a baseline, less informative belief.
    """
    arabic_discount_desc_segments = [
        {"type": "html", "content": "<p>يوضح هذا كيف يمكن لعامل الخصم أن يغير تأثير بيانات 'البيانات القديمة' اللاحقة عند استخدامها لتشكيل توزيع مسبق لفترة 'جديدة'.</p>"},
        {"type": "html", "content": "<p>افترض أن 'التوزيع اللاحق' المحسوب أعلاه هو الآن 'بيانات قديمة' من فترة سابقة.</p>"},
        {"type": "html", "content": "<p>نريد تشكيل توزيع مسبق جديد للفترة القادمة.</p>"},
        {"type": "html", "content": "<p>يمثل 'التوزيع المسبق الأولي' (على سبيل المثال، $Beta(1,1)$ باستخدام HTML entities مثل &Beta;(1,1) ) اعتقادًا أساسيًا أقل إفادة.</p>"}
    ]
    display_content(english_discount_desc, arabic_discount_desc_segments)
    
    old_posterior_alpha = posterior_alpha
    old_posterior_beta = posterior_beta

    discount_label_en = "Discount Factor (δ) for Old Data"
    discount_label_ar = "عامل الخصم (δ) للبيانات القديمة"
    discount_help_en = "Controls weight of old data. 1.0 = full weight, 0.0 = discard old data, rely only on initial prior."
    discount_help_ar = "يتحكم في وزن البيانات القديمة. ١.٠ = وزن كامل، ٠.٠ = تجاهل البيانات القديمة، والاعتماد فقط على التوزيع المسبق الأولي."
    discount_factor = st.slider(discount_label_ar if current_lang == "ar" else discount_label_en, 0.0, 1.0, 0.8, 0.05,
                                  help=discount_help_ar if current_lang == "ar" else discount_help_en, key="ia_discount_final_v2")

    init_alpha_label_en = "Initial Prior Alpha (for new period if discounting heavily)"
    init_alpha_label_ar = "ألفا المسبقة الأولية (لفترة جديدة إذا كان الخصم كبيرًا)"
    initial_prior_alpha = st.number_input(init_alpha_label_ar if current_lang == "ar" else init_alpha_label_en, 0.1, value=1.0, step=0.1, key="ia_init_alpha_final_v2")

    init_beta_label_en = "Initial Prior Beta (for new period if discounting heavily)"
    init_beta_label_ar = "بيتا المسبقة الأولية (لفترة جديدة إذا كان الخصم كبيرًا)"
    initial_prior_beta = st.number_input(init_beta_label_ar if current_lang == "ar" else init_beta_label_en, 0.1, value=1.0, step=0.1, key="ia_init_beta_final_v2")

    new_prior_alpha = discount_factor * old_posterior_alpha + (1 - discount_factor) * initial_prior_alpha
    new_prior_beta = discount_factor * old_posterior_beta + (1 - discount_factor) * initial_prior_beta

    new_prior_text_en = f"New Prior for Next Period: $Beta({new_prior_alpha:.2f}, {new_prior_beta:.2f})$"
    new_prior_text_ar_segments = [
        {"type": "html", "content": f"<p>التوزيع المسبق الجديد للفترة التالية: Beta({new_prior_alpha:.2f}, {new_prior_beta:.2f}) (يمكن استخدام &Beta;)</p>"}
        # Or for proper LaTeX:
        # {"type": "latex", "content": rf"Beta({new_prior_alpha:.2f}, {new_prior_beta:.2f})"}
    ]
    display_content(new_prior_text_en, new_prior_text_ar_segments)
    
    new_prior_mean = new_prior_alpha / (new_prior_alpha + new_prior_beta) if (new_prior_alpha + new_prior_beta) > 0 else 0
    st.write(f"{'متوسط التوزيع المسبق الجديد' if current_lang == 'ar' else 'Mean of New Prior'}: {new_prior_mean:.3f}")

    fig2, ax2 = plt.subplots()
    if old_posterior_alpha > 0 and old_posterior_beta > 0: plot_beta_distribution(old_posterior_alpha, old_posterior_beta, "Old Posterior (Data from T-1)", "اللاحق القديم (بيانات من T-1)", ax2)
    if initial_prior_alpha > 0 and initial_prior_beta > 0: plot_beta_distribution(initial_prior_alpha, initial_prior_beta, "Fixed Initial Prior", "المسبق الأولي الثابت", ax2)
    # For this specific label, the f-string in plot_beta_distribution handles the delta
    plot_beta_distribution(new_prior_alpha, new_prior_beta, f"New Prior (δ={discount_factor:.1f})", f"المسبق الجديد (δ={discount_factor:.1f})", ax2)

    ax2.set_title("Forming a New Prior with Discounting" if current_lang == "en" else "تشكيل توزيع مسبق جديد مع الخصم")
    ax2.set_xlabel("Satisfaction Rate (θ)" if current_lang == "en" else "معدل الرضا (θ)")
    ax2.set_ylabel("Density" if current_lang == "en" else "الكثافة")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)


def conclusion_content():
    display_header("7. Conclusion", "٧. الخلاصة", level=1)
    english_conclusion_md = """
The proposed Bayesian adaptive estimation framework offers a sophisticated, flexible, and efficient approach to analyzing pilgrim satisfaction surveys. By iteratively updating beliefs and dynamically adjusting sampling efforts, this methodology promises more precise and timely insights, enabling better-informed decision-making for enhancing the Hajj experience.

While it introduces new concepts and requires careful implementation, the long-term benefits—including optimized resource use and a deeper understanding of satisfaction dynamics—are substantial. This proposal advocates for a phased implementation, starting with core metrics and gradually building complexity and scope.

We recommend proceeding with a pilot project to demonstrate the practical benefits and refine the operational aspects of this advanced analytical approach.
    """
    arabic_conclusion_segments = [
        {"type": "html", "content": "<p>يقدم إطار تقدير Bayesian التكيفي المقترح نهجًا متطورًا ومرنًا وفعالًا لتحليل استطلاعات رضا الحجاج. من خلال تحديث المعتقدات بشكل متكرر وتعديل جهود أخذ العينات ديناميكيًا، تعد هذه المنهجية برؤى أكثر دقة وفي الوقت المناسب، مما يتيح اتخاذ قرارات أفضل استنارة لتعزيز تجربة الحج.</p>"},
        {"type": "html", "content": "<p>في حين أنه يقدم مفاهيم جديدة ويتطلب تنفيذًا دقيقًا، فإن الفوائد طويلة الأجل - بما في ذلك الاستخدام الأمثل للموارد والفهم الأعمق لديناميكيات الرضا - كبيرة. يدعو هذا المقترح إلى تنفيذ مرحلي، بدءًا من المقاييس الأساسية وبناء التعقيد والنطاق تدريجيًا.</p>"},
        {"type": "html", "content": "<p>نوصي بالمضي قدمًا في مشروع تجريبي لإثبات الفوائد العملية وتحسين الجوانب التشغيلية لهذا النهج التحليلي المتقدم.</p>"}
    ]
    display_content(english_conclusion_md, arabic_conclusion_segments)

# --- Streamlit App Structure (Sidebar etc.) ---
selected_lang_option = st.sidebar.selectbox(
    label="Select Language / اختر اللغة",
    options=["English", "العربية"],
    index=0,
    key="lang_select_main"
)
current_lang = "ar" if selected_lang_option == "العربية" else "en"

# App Title (from your original app for English)
app_title_en = "Proposal: Adaptive Bayesian Estimation for Pilgrim Satisfaction Surveys"
app_title_ar = "مقترح: تقدير Bayesian التكيفي لاستطلاعات رضا الحجاج"

# Use st.title for the main title, but apply RTL if Arabic
if current_lang == "ar":
    st.markdown(f"<h1 style='text-align: center; direction: rtl;'>{app_title_ar}</h1>", unsafe_allow_html=True)
else:
    st.title(app_title_en) # Using st.title for English as in your original app

# PAGES_SETUP uses keys for internal lookup and then fetches translated display names
PAGES_SETUP = {
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
selected_page_display_name = st.sidebar.radio(
    label="Go to" if current_lang == "en" else "اذهب إلى",
    options=sidebar_display_options,
    key=f"nav_radio_final_{current_lang}"
)

page_func_to_call = None
for key_internal, page_data in PAGES_SETUP.items():
    if page_data[current_lang] == selected_page_display_name:
        page_func_to_call = page_data["func"]
        break

if page_func_to_call:
    page_func_to_call()
else:
    st.error("Page loading error / خطأ في تحميل الصفحة.")


sidebar_info_en = """This app presents a proposal for using Bayesian adaptive estimation 
for Hajj pilgrim satisfaction surveys. Developed by Dr. Mohammad Nabhan."""
sidebar_info_ar = """يقدم هذا التطبيق مقترحًا لاستخدام تقدير Bayesian التكيفي 
لاستطلاعات رضا الحجاج. تم التطوير بواسطة د. محمد نبهان."""

st.sidebar.markdown("---")
st.sidebar.info(sidebar_info_ar if current_lang == "ar" else sidebar_info_en)
