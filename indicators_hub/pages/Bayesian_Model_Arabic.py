import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import os # For image path if you add one locally

# Page Configuration
st.set_page_config(layout="wide", page_title="Adaptive Bayesian Estimation Proposal")

# --- Helper Functions for Interactive Illustrations (from your English app) ---
def plot_beta_distribution(alpha, beta, label, ax):
    """Plots a Beta distribution."""
    if alpha <= 0 or beta <= 0:
        # Optionally, you could display a message on the plot or log this
        # For now, just return to avoid erroring out the plot.
        # ax.text(0.5, 0.5, "Invalid α or β", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return
    x = np.linspace(0.001, 0.999, 500) # Avoid 0 and 1 for pdf if alpha/beta are small
    try:
        y = stats.beta.pdf(x, alpha, beta)
        ax.plot(x, y, label=f'{label} (α={alpha:.2f}, β={beta:.2f})')
        ax.fill_between(x, y, alpha=0.2)
        ax.set_ylim(bottom=0) 
    except ValueError:
        # ax.text(0.5, 0.5, "PDF calculation error", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        pass


def update_beta_parameters(prior_alpha, prior_beta, successes, failures):
    """Updates Beta parameters given new data."""
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + failures
    return posterior_alpha, posterior_beta

def get_credible_interval(alpha, beta, conf_level=0.95):
    """Calculates the credible interval for a Beta distribution."""
    if alpha <= 0 or beta <= 0: 
        return (0.0, 0.0) 
    try:
        interval = stats.beta.interval(conf_level, alpha, beta)
        # Ensure interval bounds are not nan, which can happen with extreme alpha/beta
        lower = interval[0] if not np.isnan(interval[0]) else 0.0
        upper = interval[1] if not np.isnan(interval[1]) else 1.0
        return (lower, upper)
    except ValueError: 
        return (0.0,0.0)

# --- Proposal Content ---

def introduction_objectives():
    st.header("1. Introduction & Objectives")
    st.markdown("""
    This proposal outlines an **Adaptive Bayesian Estimation framework** designed to enhance the process of gathering and analyzing survey data for Hajj pilgrim satisfaction and the assessment of services provided by various companies.

    The current practice of developing satisfaction metrics month over month faces complexities, such as delays in pilgrim arrivals or non-uniformity across different months, making it difficult to consistently achieve high-confidence and low-error confidence intervals for key indicators. This proposal aims to introduce a more dynamic, efficient, and robust methodology.
    """)

    st.subheader("1.1. Primary Objectives")
    st.markdown("""
    The core objectives of implementing this adaptive Bayesian framework are:

    * **Achieve Desired Precision Efficiently:** To obtain satisfaction metrics and service provider assessments with pre-defined levels of precision (i.e., narrow credible intervals at a specific confidence level) using optimized sample sizes.
    * **Dynamic Sampling Adjustments:** To iteratively adjust sampling efforts based on accumulating evidence. This means collecting more data only when and where it's needed to meet precision targets, avoiding over-sampling or under-sampling.
    * **Timely and Reliable Estimates:** To provide decision-makers with more timely and statistically robust estimates, allowing for quicker responses to emerging issues or trends in pilgrim satisfaction.
    * **Incorporate Prior Knowledge:** To formally integrate knowledge from previous survey waves, historical data, or expert opinions into the estimation process, leading to more informed starting points and potentially faster convergence to precise estimates.
    * **Adapt to Changing Conditions:** To develop a system that can adapt to changes in satisfaction levels or service provider performance over time, for instance, by adjusting the influence of older data.
    * **Enhanced Subgroup Analysis:** To facilitate more reliable analysis of specific pilgrim subgroups or service aspects by adaptively ensuring sufficient data is collected for these segments.
    """)

    # --- Arabic Guide ---
    st.markdown("---")
    st.subheader("دليل توضيحي باللغة العربية")
    
    arabic_guide_text_intro = """
    يوضح هذا القسم إطار عمل تقدير Bayesian التكيفي المقترح. الهدف هو تحسين جمع وتحليل بيانات استطلاعات رضا الحجاج وتقييم الخدمات.
    يشرح المنهجية المقترحة لمواجهة التعقيدات الحالية في تطوير مقاييس الرضا.
    """
    st.markdown(f"<div dir='rtl' style='text-align: right;'>{arabic_guide_text_intro}</div>", unsafe_allow_html=True)
    
    arabic_guide_text_objectives = """
    الأهداف الأساسية تشمل:
    - تحقيق الدقة المطلوبة بكفاءة.
    - إجراء تعديلات ديناميكية على حجم العينات.
    - تقديم تقديرات موثوقة وفي الوقت المناسب.
    - دمج المعرفة المسبقة والبيانات التاريخية.
    - التكيف مع الظروف المتغيرة.
    - تحسين تحليل المجموعات الفرعية.
    (ملاحظة: استخدم <b>HTML</b> للتنسيق إذا لزم الأمر بعد التأكد من عمل النص العادي)
    """
    st.markdown(f"<div dir='rtl' style='text-align: right;'>{arabic_guide_text_objectives}</div>", unsafe_allow_html=True)


def challenges_addressed():
    st.header("2. Challenges Addressed by this Methodology")
    st.markdown("""
    The proposed Bayesian adaptive estimation framework directly addresses several key challenges currently faced in the Hajj survey process:

    * **Difficulty in Obtaining Stable Confidence Intervals:**
        * **Challenge:** Operational complexities like staggered pilgrim arrivals, varying visa availability periods, and diverse pilgrim schedules lead to non-uniform data collection across time. This makes it hard to achieve consistent and narrow confidence intervals for satisfaction indicators using fixed sampling plans.
        * **Bayesian Solution:** The adaptive nature allows sampling to continue until a desired precision (credible interval width) is met, regardless of initial data flow irregularities. Estimates stabilize as more data is incorporated.

    * **Inefficiency of Fixed Sample Size Approaches:**
        * **Challenge:** Predetermined sample sizes often lead to either over-sampling (wasting resources when satisfaction is homogenous or already precisely estimated) or under-sampling (resulting in inconclusive results or wide confidence intervals).
        * **Bayesian Solution:** Sampling effort is guided by the current level of uncertainty. If an estimate is already precise, sampling can be reduced or stopped for that segment. If it's imprecise, targeted additional sampling is guided by the model.

    * **Incorporation of Prior Knowledge and Historical Data:** (Content from your English app)
    * **Assessing Service Provider Performance with Evolving Data:** (Content from your English app)
    * **Balancing Fresh Data with Historical Insights:** (Content from your English app)
    * **Resource Allocation for Data Collection:** (Content from your English app)
    """) # Ensure all English content is here

    # --- Arabic Guide ---
    st.markdown("---")
    st.subheader("دليل توضيحي باللغة العربية")
    arabic_guide_text = """
    يشرح هذا القسم كيف يساعد إطار Bayesian التكيفي في التغلب على الصعوبات الحالية في عملية استطلاع الحج، مثل:
    - صعوبة الحصول على فترات ثقة مستقرة.
    - عدم كفاءة أحجام العينات الثابتة.
    (أكمل هذا النص بترجمتك وشرحك الكامل)
    """
    st.markdown(f"<div dir='rtl' style='text-align: right;'>{arabic_guide_text}</div>", unsafe_allow_html=True)

def bayesian_adaptive_methodology():
    st.header("3. Core Methodology: Bayesian Adaptive Estimation")
    st.markdown("""
    The Bayesian adaptive estimation framework is an iterative process that leverages Bayes' theorem to update our beliefs about pilgrim satisfaction or service performance as new survey data is collected. This allows for dynamic adjustments to the sampling strategy.
    """)

    st.subheader("3.1. Fundamental Concepts")
    st.markdown(r"""
    At its heart, Bayesian inference combines prior knowledge with observed data to arrive at an updated understanding, known as the posterior distribution.

    * **Prior Distribution ($P(\theta)$):** This represents our initial belief...
    * **Likelihood ($P(D|\theta)$):** This quantifies how probable...
    * **Posterior Distribution ($P(\theta|D)$):** This is our updated belief...
        $$ P(\theta|D) = \frac{P(D|\theta) \times P(\theta)}{P(D)} $$
        Where $P(D)$ is the marginal likelihood...
        $$ P(\theta|D) \propto P(D|\theta) \times P(\theta) $$
    * **Credible Interval:** In Bayesian statistics, a credible interval...
    """) # Full English content

    st.subheader("3.2. The Iterative Process")
    st.markdown("""
    The adaptive methodology follows these steps:
    1.  **Initialization:** ...
    2.  **Initial Data Collection:** ...
    3.  **Posterior Update:** ...
    4.  **Precision Assessment:** ...
    5.  **Adaptive Decision & Iteration:** ...
    This cycle continues until the desired precision is achieved...
    """) # Full English content
    
    # st.image("https_miro.medium.com_v2_resize_fit_1400_1__f_xL41kP9n2_n3L9yY0gLg.png", caption="Conceptual Flow of Bayesian Updating (Source: Medium - adapted for context)")
    # Commented out - replace with local image if you have one:
    # if os.path.exists("your_bayesian_flow_image.png"):
    # st.image("your_bayesian_flow_image.png", caption="Conceptual Flow of Bayesian Updating")


    st.subheader("3.3. Modeling Satisfaction (e.g., using Beta-Binomial Model)")
    st.markdown(r"""
    For satisfaction metrics that are proportions...
    * **Parameter of Interest ($\theta$):** The true underlying proportion...
    * **Prior Distribution (Beta):** We assume the prior belief about $\theta$ follows... $Beta(\alpha_0, \beta_0)$.
    * **Likelihood (Binomial/Bernoulli):** If we collect $n$ new responses...
        $$ P(k, n | \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k} $$
    * **Posterior Distribution (Beta):** Due to the conjugacy...
        $$ \theta | k, n \sim Beta(\alpha_0 + k, \beta_0 + n - k) $$
        The mean of this posterior distribution... is $\frac{\alpha_{post}}{\alpha_{post} + \beta_{post}}$.
    This conjugacy simplifies calculations significantly.
    """) # Full English content

    st.subheader("3.4. Adaptive Sampling Logic & Determining Additional Sample Size")
    st.markdown(r"""
    The decision to continue sampling is based on whether the current credible interval...
    * **Stopping Rule:** ... $$ U - L \leq \text{Target Width} $$
    * **Estimating Required Additional Sample Size (Conceptual):** ...
    """) # Full English content

    st.subheader("3.5. Handling Data Heterogeneity Over Time")
    st.markdown("""
    A key challenge is that service provider performance...
    * **The "Learning Hyperparameter" (Discount Factor / Power Prior):** ...
        $$ \alpha_{prior, t+1} = \delta \times \alpha_t + (1-\delta) \times \alpha_{initial} $$
        $$ \beta_{prior, t+1} = \delta \times \beta_t + (1-\delta) \times \beta_{initial} $$
    * **Change-Point Detection:** ...
    * **Hierarchical Bayesian Models (Advanced):** ...
    """) # Full English content

    # --- Arabic Guide ---
    st.markdown("---")
    st.subheader("دليل توضيحي باللغة العربية")
    arabic_guide_text_methodology_concepts = """
    <p><b>المفاهيم الأساسية (القسم ٣.١):</b></p>
    <p>يتم تعريف المصطلحات الرئيسية مثل التوزيع المسبق، دالة الإمكان، التوزيع اللاحق، وفترة الموثوقية. نظرية Bayes هي:</p>
    """
    st.markdown(f"<div dir='rtl' style='text-align: right;'>{arabic_guide_text_methodology_concepts}</div>", unsafe_allow_html=True)
    st.latex(r"P(\theta|D) = \frac{P(D|\theta) P(\theta)}{P(D)}")
    
    arabic_guide_text_methodology_iterative = """
    <p><b>العملية التكرارية (القسم ٣.٢):</b></p>
    <p>وصف للخطوات الخمس في العملية التكيفية.</p>
    """
    st.markdown(f"<div dir='rtl' style='text-align: right;'>{arabic_guide_text_methodology_iterative}</div>", unsafe_allow_html=True)
    # Add Arabic explanation for the image if you include one

    arabic_guide_text_methodology_modeling = """
    <p><b>نمذجة الرضا (القسم ٣.٣):</b></p>
    <p>التركيز على نموذج Beta-Binomial. الصيغ الرئيسية هي:</p>
    <p>دالة الإمكان:</p>
    """
    st.markdown(f"<div dir='rtl' style='text-align: right;'>{arabic_guide_text_methodology_modeling}</div>", unsafe_allow_html=True)
    st.latex(r"P(k, n | \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k}")
    st.markdown(f"<div dir='rtl' style='text-align: right;'><p>التوزيع اللاحق:</p></div>", unsafe_allow_html=True)
    st.latex(r"\theta | k, n \sim Beta(\alpha_0 + k, \beta_0 + n - k)")
    st.markdown(f"<div dir='rtl' style='text-align: right;'><p>المتوسط اللاحق (التقدير النقطي):</p></div>", unsafe_allow_html=True)
    st.latex(r"\frac{\alpha_{post}}{\alpha_{post} + \beta_{post}}")

    arabic_guide_text_methodology_sampling = """
    <p><b>منطق أخذ العينات التكيفي (القسم ٣.٤):</b></p>
    <p>قاعدة الإيقاف:</p>
    """
    st.markdown(f"<div dir='rtl' style='text-align: right;'>{arabic_guide_text_methodology_sampling}</div>", unsafe_allow_html=True)
    st.latex(r"U - L \leq \text{Target Width}")
    
    arabic_guide_text_methodology_heterogeneity = """
    <p><b>التعامل مع عدم تجانس البيانات (القسم ٣.٥):</b></p>
    <p>معادلات معامل الخصم:</p>
    """
    st.markdown(f"<div dir='rtl' style='text-align: right;'>{arabic_guide_text_methodology_heterogeneity}</div>", unsafe_allow_html=True)
    st.latex(r"\alpha_{prior, t+1} = \delta \times \alpha_t + (1-\delta) \times \alpha_{initial}")
    st.latex(r"\beta_{prior, t+1} = \delta \times \beta_t + (1-\delta) \times \beta_{initial}")


def implementation_roadmap():
    st.header("4. Implementation Roadmap (Conceptual)")
    st.markdown("""
    Implementing the Bayesian adaptive estimation framework involves several key stages:
    """)
    df_roadmap = pd.DataFrame({
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
    })
    st.dataframe(df_roadmap, hide_index=True, use_container_width=True)

    # --- Arabic Guide ---
    st.markdown("---")
    st.subheader("دليل توضيحي باللغة العربية")
    arabic_guide_text = """
    <p><b>خارطة طريق التنفيذ (القسم ٤):</b></p>
    <p>يعرض هذا القسم المراحل والخطوات الرئيسية المقترحة لتنفيذ إطار تقدير Bayesian التكيفي.</p>
    <p>(يرجى الرجوع إلى الجدول الإنجليزي أعلاه للحصول على التفاصيل الكاملة. ستحتاج إلى ترجمة محتويات الجدول للغة العربية)</p>
    """
    st.markdown(f"<div dir='rtl' style='text-align: right;'>{arabic_guide_text}</div>", unsafe_allow_html=True)

def note_to_practitioners():
    st.header("5. Note to Practitioners")

    st.subheader("5.1. Benefits of the Bayesian Adaptive Approach")
    st.markdown("""
    * **Efficiency:** Targets sampling effort where it's most needed...
    * **Adaptability:** Responds to incoming data...
    (Full English content from your app)
    """)

    st.subheader("5.2. Limitations and Considerations")
    st.markdown("""
    * **Complexity:** Bayesian methods can be conceptually more demanding...
    (Full English content from your app)
    """)

    st.subheader("5.3. Key Assumptions")
    st.markdown("""
    * **Representativeness of Samples:** Each batch of collected data...
    (Full English content from your app)
    """)

    st.subheader("5.4. Practical Recommendations")
    st.markdown("""
    * **Start Simple:** Begin with core satisfaction metrics...
    (Full English content from your app)
    """)

    # --- Arabic Guide ---
    st.markdown("---")
    st.subheader("دليل توضيحي باللغة العربية")
    arabic_guide_text = """
    <p><b>ملاحظة للممارسين (القسم ٥):</b></p>
    <p>هذا القسم موجه للممارسين ويقدم نظرة عامة على جوانب مختلفة من النهج المقترح.</p>
    <p><b>٥.١ فوائد النهج:</b> (اشرح الفوائد مثل الكفاءة، القدرة على التكيف، إلخ)</p>
    <p><b>٥.٢ القيود والاعتبارات:</b> (ناقش التعقيد، اختيار التوزيع المسبق، إلخ)</p>
    <p><b>٥.٣ الافتراضات الرئيسية:</b> (اذكر الافتراضات مثل تمثيلية العينات، ملاءمة النموذج، إلخ)</p>
    <p><b>٥.٤ التوصيات العملية:</b> (قدم نصائح مثل البدء ببساطة، التدريب، الشفافية، إلخ)</p>
    """
    st.markdown(f"<div dir='rtl' style='text-align: right;'>{arabic_guide_text}</div>", unsafe_allow_html=True)


def interactive_illustration():
    st.header("6. Interactive Illustration: Beta-Binomial Model")
    st.markdown("""
    This section provides a simple interactive illustration of how a Beta prior is updated to a Beta posterior with new data (Binomial likelihood). This is the core of estimating a proportion (e.g., satisfaction rate) in a Bayesian way.
    """)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prior Beliefs")
        st.markdown("The Beta distribution $Beta(\\alpha, \\beta)$ is a common prior for proportions. $\\alpha$ can be thought of as prior 'successes' and $\\beta$ as prior 'failures'. $Beta(1,1)$ is a uniform (uninformative) prior.")
        prior_alpha = st.slider("Prior Alpha (α₀)", min_value=0.1, max_value=50.0, value=1.0, step=0.1, key="prior_a_eng_v3")
        prior_beta = st.slider("Prior Beta (β₀)", min_value=0.1, max_value=50.0, value=1.0, step=0.1, key="prior_b_eng_v3")
        
        # Calculate prior_mean ensuring denominator is not zero
        if (prior_alpha + prior_beta) > 0:
            prior_mean = prior_alpha / (prior_alpha + prior_beta)
        else:
            prior_mean = 0 # Or handle as an error/undefined case
        st.write(f"Prior Mean: {prior_mean:.3f}")
        
        prior_ci = get_credible_interval(prior_alpha, prior_beta)
        st.write(f"95% Credible Interval (Prior): [{prior_ci[0]:.3f}, {prior_ci[1]:.3f}], Width: {abs(prior_ci[1]-prior_ci[0]):.3f}")


    with col2:
        st.subheader("New Survey Data (Likelihood)")
        st.markdown("Enter the results from a new batch of surveys.")
        num_surveys = st.slider("Number of New Surveys (n)", min_value=1, max_value=500, value=50, step=1, key="surveys_n_eng_v3")
        num_satisfied = st.slider("Number Satisfied in New Surveys (k)", min_value=0, max_value=num_surveys, value=int(num_surveys/2), step=1, key="surveys_k_eng_v3")
        
        if num_surveys > 0:
            observed_satisfaction = num_satisfied / num_surveys
        else:
            observed_satisfaction = 0 # Or handle as undefined
        st.write(f"Observed Satisfaction in New Data: {observed_satisfaction:.3f}")

    st.markdown("---")
    st.subheader("Posterior Beliefs (After Update)")
    posterior_alpha, posterior_beta = update_beta_parameters(prior_alpha, prior_beta, num_satisfied, num_surveys - num_satisfied)
    st.markdown(f"The posterior distribution is $Beta(\\alpha_0 + k, \\beta_0 + n - k) = Beta({posterior_alpha:.1f}, {posterior_beta:.1f})$")

    if (posterior_alpha + posterior_beta) > 0:
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    else:
        posterior_mean = 0
    st.write(f"Posterior Mean: {posterior_mean:.3f}")
    
    posterior_ci = get_credible_interval(posterior_alpha, posterior_beta)
    st.write(f"95% Credible Interval (Posterior): [{posterior_ci[0]:.3f}, {posterior_ci[1]:.3f}], Width: {abs(posterior_ci[1]-posterior_ci[0]):.3f}")

    target_width = st.number_input("Target Credible Interval Width for Stopping", min_value=0.01, max_value=1.0, value=0.10, step=0.01, key="target_w_eng_v3")
    current_width = abs(posterior_ci[1]-posterior_ci[0]) if posterior_ci[0] is not None and posterior_ci[1] is not None else float('inf')
    
    # Ensure current_width is a finite number before comparison
    if np.isfinite(current_width):
        if current_width <= target_width and (posterior_alpha > 0 or posterior_beta > 0):
            st.success(f"Target precision met! Current width ({current_width:.3f}) ≤ Target width ({target_width:.3f}).")
        elif posterior_alpha == 0 and posterior_beta == 0 and current_width == 0.0 : # Special case for Beta(0,0) interval being (0,0)
            st.warning(f"Posterior parameters are zero. Check inputs.")
        else:
            st.warning(f"Target precision not yet met. Current width ({current_width:.3f}) > Target width ({target_width:.3f}). Consider more samples.")
    else:
        st.warning("Current width is undefined. Check inputs.")


    fig, ax = plt.subplots()
    if prior_alpha > 0 and prior_beta > 0: plot_beta_distribution(prior_alpha, prior_beta, "Prior", "المسبق", ax)
    if posterior_alpha > 0 and posterior_beta > 0: plot_beta_distribution(posterior_alpha, posterior_beta, "Posterior", "اللاحق", ax)
    ax.set_title("Prior and Posterior Distributions of Satisfaction Rate")
    ax.set_xlabel("Satisfaction Rate (θ)")
    ax.set_ylabel("Density")
    if (prior_alpha > 0 and prior_beta > 0) or (posterior_alpha > 0 and posterior_beta > 0): ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Conceptual Illustration: Impact of Discounting Older Data")
    st.markdown("""
    This illustrates how a discount factor might change the influence of 'old' posterior data when it's used to form a prior for a 'new' period.
    Assume the 'Posterior' calculated above is now 'Old Data' from a previous period.
    We want to form a new prior for the upcoming period.
    An 'Initial Prior' (e.g., $Beta(1,1)$) represents a baseline, less informative belief.
    """)
    old_posterior_alpha = posterior_alpha
    old_posterior_beta = posterior_beta

    discount_factor = st.slider("Discount Factor (δ) for Old Data", 0.0, 1.0, 0.8, 0.05,
                                  help="Controls weight of old data. 1.0 = full weight, 0.0 = discard old data, rely only on initial prior.", key="discount_eng_v3")

    initial_prior_alpha = st.number_input("Initial Prior Alpha (for new period if discounting heavily)", 0.1, value=1.0, step=0.1, key="init_prior_a_eng_v3")
    initial_prior_beta = st.number_input("Initial Prior Beta (for new period if discounting heavily)", 0.1, value=1.0, step=0.1, key="init_prior_b_eng_v3")

    new_prior_alpha = discount_factor * old_posterior_alpha + (1 - discount_factor) * initial_prior_alpha
    new_prior_beta = discount_factor * old_posterior_beta + (1 - discount_factor) * initial_prior_beta

    st.write(f"New Prior for Next Period: $Beta({new_prior_alpha:.2f}, {new_prior_beta:.2f})$")
    if (new_prior_alpha + new_prior_beta) > 0:
        new_prior_mean = new_prior_alpha / (new_prior_alpha + new_prior_beta)
    else:
        new_prior_mean = 0
    st.write(f"Mean of New Prior: {new_prior_mean:.3f}")

    fig2, ax2 = plt.subplots()
    if old_posterior_alpha > 0 and old_posterior_beta > 0: plot_beta_distribution(old_posterior_alpha, old_posterior_beta, "Old Posterior (Data from T-1)", "اللاحق القديم (بيانات من T-1)", ax2)
    if initial_prior_alpha > 0 and initial_prior_beta > 0: plot_beta_distribution(initial_prior_alpha, initial_prior_beta, "Fixed Initial Prior", "المسبق الأولي الثابت", ax2)
    
    new_prior_label_en = f"New Prior (δ={discount_factor:.1f})" # For legend
    new_prior_label_ar = f"المسبق الجديد (δ={discount_factor:.1f})"
    if new_prior_alpha > 0 and new_prior_beta > 0: plot_beta_distribution(new_prior_alpha, new_prior_beta, new_prior_label_en, new_prior_label_ar, ax2)
    
    ax2.set_title("Forming a New Prior with Discounting")
    ax2.set_xlabel("Satisfaction Rate (θ)")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # --- Arabic Guide ---
    st.markdown("---")
    st.subheader("دليل توضيحي باللغة العربية")
    arabic_guide_text = """
    <p><b>التوضيح التفاعلي (القسم ٦):</b></p>
    <p>يوفر هذا القسم أدوات تفاعلية لفهم كيفية تحديث توزيع Beta المسبق إلى توزيع لاحق مع بيانات جديدة.</p>
    <p>الصيغة الأساسية للتوزيع اللاحق هي:</p>
    """
    st.markdown(f"<div dir='rtl' style='text-align: right;'>{arabic_guide_text}</div>", unsafe_allow_html=True)
    st.latex(r"Beta(\alpha_0 + k, \beta_0 + n - k)")
    arabic_guide_text_cont = """
    <p>حيث &alpha;<sub>0</sub> و &beta;<sub>0</sub> هي معلمات التوزيع المسبق، k عدد الراضين، و n إجمالي عدد الاستطلاعات الجديدة.</p>
    <p>(أكمل هذا النص بترجمتك وشرحك الكامل للأجزاء التفاعلية.)</p>
    """
    st.markdown(f"<div dir='rtl' style='text-align: right;'>{arabic_guide_text_cont}</div>", unsafe_allow_html=True)


def conclusion_content():
    st.header("7. Conclusion")
    st.markdown("""
    The proposed Bayesian adaptive estimation framework offers a sophisticated, flexible, and efficient approach to analyzing pilgrim satisfaction surveys. By iteratively updating beliefs and dynamically adjusting sampling efforts, this methodology promises more precise and timely insights, enabling better-informed decision-making for enhancing the Hajj experience.

    While it introduces new concepts and requires careful implementation, the long-term benefits—including optimized resource use and a deeper understanding of satisfaction dynamics—are substantial. This proposal advocates for a phased implementation, starting with core metrics and gradually building complexity and scope.

    We recommend proceeding with a pilot project to demonstrate the practical benefits and refine the operational aspects of this advanced analytical approach.
    """)

    # --- Arabic Guide ---
    st.markdown("---")
    st.subheader("دليل توضيحي باللغة العربية")
    arabic_guide_text = """
    <p><b>الخلاصة (القسم ٧):</b></p>
    <p>يلخص هذا القسم الفوائد الرئيسية لإطار تقدير Bayesian التكيفي المقترح، ويوصي بالبدء بمشروع تجريبي.</p>
    <p>(أكمل هذا النص بترجمتك الكاملة.)</p>
    """
    st.markdown(f"<div dir='rtl' style='text-align: right;'>{arabic_guide_text}</div>", unsafe_allow_html=True)


# --- Streamlit App Structure ---
st.title("Proposal: Adaptive Bayesian Estimation for Pilgrim Satisfaction Surveys")

PAGES = {
    "1. Introduction & Objectives": introduction_objectives_content,
    "2. Challenges Addressed": challenges_addressed_content,
    "3. Bayesian Adaptive Methodology": bayesian_adaptive_methodology_content,
    "4. Implementation Roadmap": implementation_roadmap_content,
    "5. Note to Practitioners": note_to_practitioners_content,
    "6. Interactive Illustration": interactive_illustration_content,
    "7. Conclusion": conclusion_content
}

st.sidebar.title("Proposal Sections")
selection = st.sidebar.radio("Go to", list(PAGES.keys()), key="main_nav_guide_v1")

page_function = PAGES[selection]
page_function()

st.sidebar.markdown("---")
st.sidebar.info(
    "This app presents a proposal for using Bayesian adaptive estimation "
    "for Hajj pilgrim satisfaction surveys. Developed by Dr. Mohammad Nabhan."
)
