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
    # Ensure alpha and beta are valid for pdf calculation
    if alpha <= 0 or beta <= 0:
        # ax.text(0.5, 0.5, "Invalid α or β for PDF", ha='center', va='center')
        return
    x = np.linspace(0.001, 0.999, 500) # Avoid 0 and 1 for pdf if alpha/beta are small
    try:
        y = stats.beta.pdf(x, alpha, beta)
        ax.plot(x, y, label=f'{label} (α={alpha:.2f}, β={beta:.2f})')
        ax.fill_between(x, y, alpha=0.2)
        ax.set_ylim(bottom=0) # Ensure y-axis starts at 0
    except ValueError:
        # ax.text(0.5, 0.5, "Error in PDF calculation", ha='center', va='center')
        pass


def update_beta_parameters(prior_alpha, prior_beta, successes, failures):
    """Updates Beta parameters given new data."""
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + failures
    return posterior_alpha, posterior_beta

def get_credible_interval(alpha, beta, conf_level=0.95):
    """Calculates the credible interval for a Beta distribution."""
    if alpha <= 0 or beta <= 0: # Invalid parameters
        return (0.0, 0.0) # Return float tuple
    try:
        return stats.beta.interval(conf_level, alpha, beta)
    except ValueError: # Catch math domain errors
        return (0.0,0.0)


# --- Proposal Content (Original English + Arabic Guide Sections) ---

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
    st.markdown("""
    <div dir='rtl' style='text-align: right;'>
    <p><b>مقدمة وأهداف المقترح (القسم ١):</b></p>
    <p>يوضح هذا القسم إطار عمل تقدير Bayesian التكيفي المقترح. الهدف هو تحسين جمع وتحليل بيانات استطلاعات رضا الحجاج وتقييم الخدمات.</p>
    <p><b>الأهداف الأساسية (القسم ١.١):</b></p>
    <ul>
        <li>تحقيق الدقة المطلوبة بكفاءة.</li>
        <li>إجراء تعديلات ديناميكية على حجم العينات.</li>
        <li>تقديم تقديرات موثوقة وفي الوقت المناسب.</li>
        <li>دمج المعرفة المسبقة والبيانات التاريخية.</li>
        <li>التكيف مع الظروف المتغيرة.</li>
        <li>تحسين تحليل المجموعات الفرعية.</li>
    </ul>
    <p><i>ملاحظة: هذا النص هو شرح مبسط للنص الإنجليزي أعلاه.</i></p>
    </div>
    """, unsafe_allow_html=True)

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
    """)

    # --- Arabic Guide ---
    st.markdown("---")
    st.subheader("دليل توضيحي باللغة العربية")
    st.markdown("""
    <div dir='rtl' style='text-align: right;'>
    <p><b>التحديات التي تعالجها هذه المنهجية (القسم ٢):</b></p>
    <p>يشرح هذا القسم كيف يساعد إطار Bayesian التكيفي في التغلب على الصعوبات الحالية في عملية استطلاع الحج، مثل:</p>
    <ul>
        <li>صعوبة الحصول على فترات ثقة مستقرة.</li>
        <li>عدم كفاءة أحجام العينات الثابتة.</li>
        <li>الحاجة إلى دمج المعرفة المسبقة.</li>
        <li>تقييم أداء مقدمي الخدمات مع البيانات المتطورة.</li>
        <li>موازنة البيانات الجديدة مع الرؤى التاريخية.</li>
        <li>تخصيص موارد جمع البيانات بكفاءة.</li>
    </ul>
    <p>لكل تحد، يتم توضيح كيف يقدم النهج البايزي التكيفي حلاً.</p>
    </div>
    """, unsafe_allow_html=True)

def bayesian_adaptive_methodology():
    st.header("3. Core Methodology: Bayesian Adaptive Estimation")
    st.markdown("""
    The Bayesian adaptive estimation framework is an iterative process that leverages Bayes' theorem to update our beliefs about pilgrim satisfaction or service performance as new survey data is collected. This allows for dynamic adjustments to the sampling strategy.
    """)

    st.subheader("3.1. Fundamental Concepts")
    st.markdown(r"""
    At its heart, Bayesian inference combines prior knowledge with observed data to arrive at an updated understanding, known as the posterior distribution.

    * **Prior Distribution ($P(\theta)$):** This represents our initial belief about a parameter $\theta$ (e.g., the proportion of satisfied pilgrims) *before* observing new data. It can be based on historical data, expert opinion, or be deliberately "uninformative" if we want the data to speak for itself.
    * **Likelihood ($P(D|\theta)$):** This quantifies how probable the observed data ($D$) is, given a particular value of the parameter $\theta$. It is the function that connects the data to the parameter.
    * **Posterior Distribution ($P(\theta|D)$):** This is our updated belief about $\theta$ *after* observing the data. It is calculated using Bayes' Theorem:
        $$ P(\theta|D) = \frac{P(D|\theta) \times P(\theta)}{P(D)} $$
        Where $P(D)$ is the marginal likelihood of the data, acting as a normalizing constant. In practice, we often focus on the proportionality:
        $$ P(\theta|D) \propto P(D|\theta) \times P(\theta) $$
    * **Credible Interval:** In Bayesian statistics, a credible interval is a range of values that contains the parameter $\theta$ with a certain probability (e.g., 95%). This is a direct probabilistic statement about the parameter, unlike the frequentist confidence interval.
    """)

    st.subheader("3.2. The Iterative Process")
    st.markdown("""
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
    """)
    # st.image("https_miro.medium.com_v2_resize_fit_1400_1__f_xL41kP9n2_n3L9yY0gLg.png", caption="Conceptual Flow of Bayesian Updating (Source: Medium - adapted for context)")
    # Replace with a local image if available or remove. For now, commented out.
    # Example: if os.path.exists("your_image.png"): st.image("your_image.png")


    st.subheader("3.3. Modeling Satisfaction (e.g., using Beta-Binomial Model)")
    st.markdown(r"""
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
    """)

    st.subheader("3.4. Adaptive Sampling Logic & Determining Additional Sample Size")
    st.markdown(r"""
    The decision to continue sampling is based on whether the current credible interval for $\theta$ meets the desired precision.

    * **Stopping Rule:** Stop sampling for a specific metric when (for a $(1-\gamma)\%$ credible interval $[L, U]$):
        $$ U - L \leq \text{Target Width} $$
        And/or when the credible interval lies entirely above/below a certain threshold of practical importance.

    * **Estimating Required Additional Sample Size (Conceptual):**
        While exact formulas for sample size to guarantee a future credible interval width are complex because the width itself is a random variable, several approaches can guide this:
        1.  **Simulation:** Based on the current posterior $Beta(\alpha_{post}, \beta_{post})$, simulate drawing additional samples of various sizes. For each simulated sample size, calculate the resulting posterior and its credible interval width. This can give a distribution of expected widths for different additional $n$.
        2.  **Approximation Formulas:** Some researchers have developed approximations... (rest of your English content)
        3.  **Bayesian Sequential Analysis:** More formal methods...
        4.  **Pragmatic Batching:** Collect data in smaller, manageable batches...

    The tool should aim to provide guidance on a reasonable next batch size based on the current uncertainty and the distance to the target precision.
    """)

    st.subheader("3.5. Handling Data Heterogeneity Over Time")
    st.markdown("""
    A key challenge is that service provider performance or general pilgrim satisfaction might change over time. Using historical data uncritically as a prior might be misleading if changes have occurred.

    * **The "Learning Hyperparameter" (Discount Factor / Power Prior):**
        One way to address this is to down-weight older data...
        $$ \alpha_{prior, t+1} = \delta \times \alpha_t + (1-\delta) \times \alpha_{initial} $$
        $$ \beta_{prior, t+1} = \delta \times \beta_t + (1-\delta) \times \beta_{initial} $$
        Where $(\alpha_{initial}, \beta_{initial})$ could be parameters of a generic, uninformative prior... (rest of your English content)

    * **Change-Point Detection:** Periodically, statistical tests can be run...
    * **Hierarchical Bayesian Models (Advanced):** These models can explicitly model variation...

    The choice of method depends on the complexity deemed appropriate and the available data. Starting with a discount factor is often a pragmatic first step.
    """)

    # --- Arabic Guide ---
    st.markdown("---")
    st.subheader("دليل توضيحي باللغة العربية")
    st.markdown("""
    <div dir='rtl' style='text-align: right;'>
    <p><b>المنهجية الأساسية: تقدير Bayesian التكيفي (القسم ٣):</b></p>
    <p>يشرح هذا القسم جوهر المنهجية البايزية التكيفية المقترحة.</p>
    
    <p><b>المفاهيم الأساسية (القسم ٣.١):</b></p>
    <p>يتم تعريف المصطلحات الرئيسية مثل التوزيع المسبق (Prior)، دالة الإمكان (Likelihood)، التوزيع اللاحق (Posterior)، وفترة الموثوقية (Credible Interval). يتم عرض نظرية Bayes:</p>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"P(\theta|D) = \frac{P(D|\theta) P(\theta)}{P(D)}") # Display math separately
    st.markdown("""
    <div dir='rtl' style='text-align: right;'>
    <p><b>العملية التكرارية (القسم ٣.٢):</b></p>
    <p>وصف للخطوات الخمس في العملية التكيفية: التهيئة، جمع البيانات الأولي، تحديث التوزيع اللاحق، تقييم الدقة، والقرار التكيفي.</p>
    
    <p><b>نمذجة الرضا (القسم ٣.٣):</b></p>
    <p>يتم التركيز على نموذج Beta-Binomial لتقدير نسب الرضا. يتم عرض معادلات التوزيع المسبق، دالة الإمكان، والتوزيع اللاحق، بالإضافة إلى معادلة تقدير المتوسط اللاحق:</p>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"\frac{\alpha_{post}}{\alpha_{post} + \beta_{post}}") # Display math separately
    st.markdown("""
    <div dir='rtl' style='text-align: right;'>
    <p><b>منطق أخذ العينات التكيفي (القسم ٣.٤):</b></p>
    <p>شرح لقاعدة الإيقاف وكيفية تقدير حجم العينة الإضافي المطلوب.</p>
    
    <p><b>التعامل مع عدم تجانس البيانات بمرور الوقت (القسم ٣.٥):</b></p>
    <p>مناقشة لتحدي تغير الأداء بمرور الوقت، وطرق مثل "معامل الخصم" لمعالجة ذلك، مع عرض المعادلات ذات الصلة:</p>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"\alpha_{prior, t+1} = \delta \times \alpha_t + (1-\delta) \times \alpha_{initial}")
    st.latex(r"\beta_{prior, t+1} = \delta \times \beta_t + (1-\delta) \times \beta_{initial}")
    st.markdown("""
    <div dir='rtl' style='text-align: right;'>
    <p>بالإضافة إلى طرق أخرى مثل كشف نقاط التغيير ونماذج Bayesian الهرمية.</p>
    </div>
    """, unsafe_allow_html=True)

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
    st.markdown("""
    <div dir='rtl' style='text-align: right;'>
    <p><b>خارطة طريق التنفيذ (القسم ٤):</b></p>
    <p>يعرض هذا القسم المراحل والخطوات الرئيسية المقترحة لتنفيذ إطار تقدير Bayesian التكيفي. يتم تقديمها في جدول يوضح كل مرحلة، والخطوات ضمنها، ووصف لكل خطوة.</p>
    <p>الجدول أعلاه باللغة الإنجليزية، ويمكن ترجمة محتواه ليتضمن:</p>
    <ul>
        <li><b>المرحلة ١: التأسيس والتجربة الأولية</b> (خطوات مثل تحديد المقاييس، إعداد النظام)</li>
        <li><b>المرحلة ٢: التطوير التكراري والاختبار</b> (خطوات مثل تطوير النموذج، تطوير لوحة التحكم)</li>
        <li><b>المرحلة ٣: النشر على نطاق واسع والتحسين</b> (خطوات مثل التوسع التدريجي، المراقبة المستمرة)</li>
    </ul>
    <p>ملاحظة: إذا كانت هناك رموز رياضية مثل $Beta(1,1)$ في الجدول، فيمكن عرضها بشكل منفصل باستخدام `st.latex()` إذا لم تظهر بشكل صحيح في الترجمة المباشرة للجدول.</p>
    </div>
    """, unsafe_allow_html=True)


def note_to_practitioners():
    st.header("5. Note to Practitioners")

    st.subheader("5.1. Benefits of the Bayesian Adaptive Approach")
    st.markdown("""
    * **Efficiency:** Targets sampling effort where it's most needed, potentially reducing overall sample sizes compared to fixed methods while achieving desired precision.
    * **Adaptability:** Responds to incoming data, making it suitable for dynamic environments where satisfaction might fluctuate or where initial knowledge is low.
    * **Formal Use of Prior Knowledge:** Allows systematic incorporation of historical data or expert insights, which can be particularly useful with sparse initial data for new services or specific subgroups.
    * **Intuitive Uncertainty Quantification:** Credible intervals offer a direct probabilistic interpretation of the parameter's range, which can be easier for stakeholders to understand than frequentist confidence intervals.
    * **Rich Output:** Provides a full posterior distribution for each parameter, offering more insight than just a point estimate and an interval.
    """)

    st.subheader("5.2. Limitations and Considerations")
    st.markdown("""
    * **Complexity:** Bayesian methods can be conceptually more demanding than traditional frequentist approaches. Implementation requires specialized knowledge.
    * **Prior Selection:** The choice of prior distribution can influence posterior results, especially with small sample sizes. This requires careful justification and transparency. While "uninformative" priors aim to minimize this influence, truly uninformative priors are not always straightforward.
    * **Computational Cost:** While Beta-Binomial models are computationally simple, more complex Bayesian models (e.g., hierarchical models, models requiring MCMC simulation) can be computationally intensive.
    * **Interpretation Differences:** Practitioners familiar with frequentist statistics need to understand the different interpretations of Bayesian outputs (e.g., credible intervals vs. confidence intervals).
    * **"Black Box" Perception:** If not explained clearly, the adaptive nature and Bayesian calculations might be perceived as a "black box" by those unfamiliar with the methods. Clear communication is key.
    """)

    st.subheader("5.3. Key Assumptions")
    st.markdown("""
    * **Representativeness of Samples:** Each batch of collected data is assumed to be representative of the (sub)population of interest *at that point in time*. Sampling biases will affect the validity of estimates.
    * **Model Appropriateness:** The chosen likelihood and prior distributions should reasonably reflect the data-generating process and existing knowledge. For satisfaction proportions, the Beta-Binomial model is often robust.
    * **Stability (or Modeled Change):** The underlying parameter being measured (e.g., satisfaction rate) is assumed to be relatively stable between iterative updates within a survey wave, OR changes are explicitly modeled (e.g., via discount factors or dynamic models). Rapid, unmodeled fluctuations can be challenging.
    * **Accurate Data:** Assumes responses are truthful and accurately recorded.
    """)

    st.subheader("5.4. Practical Recommendations")
    st.markdown("""
    * **Start Simple:** Begin with core satisfaction metrics and simple models (like Beta-Binomial). Complexity can be added iteratively as experience is gained.
    * **Invest in Training:** Ensure that the team involved in implementing and interpreting the results has adequate training in Bayesian statistics.
    * **Transparency is Key:** Document choices for priors, models, and adaptive rules. Perform sensitivity analyses to understand the impact of different prior choices, especially in early stages or with limited data.
    * **Regular Review and Validation:** Periodically review the performance of the models. Compare Bayesian estimates with those from traditional methods if possible, especially during a transition period. Validate assumptions.
    * **Stakeholder Communication:** Develop clear ways to communicate the methodology, its benefits, and the interpretation of results to stakeholders who may not be statisticians.
    * **Pilot Thoroughly:** Before full-scale implementation, conduct thorough pilot studies to refine the process, test the technology, and identify unforeseen challenges.
    """)

    # --- Arabic Guide ---
    st.markdown("---")
    st.subheader("دليل توضيحي باللغة العربية")
    st.markdown("""
    <div dir='rtl' style='text-align: right;'>
    <p><b>ملاحظة للممارسين (القسم ٥):</b></p>
    <p>هذا القسم موجه للممارسين ويقدم نظرة عامة على جوانب مختلفة من النهج المقترح.</p>
    
    <p><b>فوائد نهج Bayesian التكيفي (القسم ٥.١):</b></p>
    <p>يتم تفصيل المزايا مثل الكفاءة، القدرة على التكيف، الاستخدام الرسمي للمعرفة المسبقة، قياس عدم اليقين البديهي، والمخرجات الغنية.</p>
    
    <p><b>القيود والاعتبارات (القسم ٥.٢):</b></p>
    <p>تناقش التحديات المحتملة مثل التعقيد، أهمية اختيار التوزيع المسبق، التكلفة الحسابية، اختلافات التفسير، وتصور "الصندوق الأسود".</p>
    
    <p><b>الافتراضات الرئيسية (القسم ٥.٣):</b></p>
    <p>تُسرد الافتراضات الأساسية التي تعتمد عليها المنهجية، مثل تمثيلية العينات، ملاءمة النموذج، استقرار المعلمة (أو نمذجة التغيير)، ودقة البيانات.</p>
    
    <p><b>التوصيات العملية (القسم ٥.٤):</b></p>
    <p>تُقدم نصائح عملية للتنفيذ الناجح، بما في ذلك البدء ببساطة، الاستثمار في التدريب، أهمية الشفافية، المراجعة والتحقق المنتظم، التواصل مع أصحاب المصلحة، وإجراء تجربة أولية شاملة.</p>
    </div>
    """, unsafe_allow_html=True)

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
        prior_alpha = st.slider("Prior Alpha (α₀)", min_value=0.1, max_value=50.0, value=1.0, step=0.1, key="prior_a_eng")
        prior_beta = st.slider("Prior Beta (β₀)", min_value=0.1, max_value=50.0, value=1.0, step=0.1, key="prior_b_eng")
        prior_mean = prior_alpha / (prior_alpha + prior_beta) if (prior_alpha + prior_beta) > 0 else 0
        st.write(f"Prior Mean: {prior_mean:.3f}")
        prior_ci = get_credible_interval(prior_alpha, prior_beta)
        st.write(f"95% Credible Interval (Prior): [{prior_ci[0]:.3f}, {prior_ci[1]:.3f}], Width: {abs(prior_ci[1]-prior_ci[0]):.3f}")


    with col2:
        st.subheader("New Survey Data (Likelihood)")
        st.markdown("Enter the results from a new batch of surveys.")
        num_surveys = st.slider("Number of New Surveys (n)", min_value=1, max_value=500, value=50, step=1, key="surveys_n_eng")
        num_satisfied = st.slider("Number Satisfied in New Surveys (k)", min_value=0, max_value=num_surveys, value=int(num_surveys/2), step=1, key="surveys_k_eng")
        st.write(f"Observed Satisfaction in New Data: {num_satisfied/num_surveys if num_surveys > 0 else 0:.3f}")

    st.markdown("---")
    st.subheader("Posterior Beliefs (After Update)")
    posterior_alpha, posterior_beta = update_beta_parameters(prior_alpha, prior_beta, num_satisfied, num_surveys - num_satisfied)
    st.markdown(f"The posterior distribution is $Beta(\\alpha_0 + k, \\beta_0 + n - k) = Beta({posterior_alpha:.1f}, {posterior_beta:.1f})$")

    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta) if (posterior_alpha + posterior_beta) > 0 else 0
    st.write(f"Posterior Mean: {posterior_mean:.3f}")
    posterior_ci = get_credible_interval(posterior_alpha, posterior_beta)
    st.write(f"95% Credible Interval (Posterior): [{posterior_ci[0]:.3f}, {posterior_ci[1]:.3f}], Width: {abs(posterior_ci[1]-posterior_ci[0]):.3f}")

    target_width = st.number_input("Target Credible Interval Width for Stopping", min_value=0.01, max_value=1.0, value=0.10, step=0.01, key="target_w_eng")
    current_width = abs(posterior_ci[1]-posterior_ci[0]) if posterior_ci[0] is not None and posterior_ci[1] is not None else float('inf')
    
    if current_width <= target_width and (posterior_alpha > 0 or posterior_beta > 0):
        st.success(f"Target precision met! Current width ({current_width:.3f}) ≤ Target width ({target_width:.3f}).")
    else:
        st.warning(f"Target precision not yet met. Current width ({current_width:.3f}) > Target width ({target_width:.3f}). Consider more samples.")

    fig, ax = plt.subplots()
    if prior_alpha > 0 and prior_beta > 0: plot_beta_distribution(prior_alpha, prior_beta, "Prior", "المسبق", ax) # Pass both labels
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
                                  help="Controls weight of old data. 1.0 = full weight, 0.0 = discard old data, rely only on initial prior.", key="discount_eng")

    initial_prior_alpha = st.number_input("Initial Prior Alpha (for new period if discounting heavily)", 0.1, value=1.0, step=0.1, key="init_prior_a_eng")
    initial_prior_beta = st.number_input("Initial Prior Beta (for new period if discounting heavily)", 0.1, value=1.0, step=0.1, key="init_prior_b_eng")

    new_prior_alpha = discount_factor * old_posterior_alpha + (1 - discount_factor) * initial_prior_alpha
    new_prior_beta = discount_factor * old_posterior_beta + (1 - discount_factor) * initial_prior_beta

    st.write(f"New Prior for Next Period: $Beta({new_prior_alpha:.2f}, {new_prior_beta:.2f})$")
    new_prior_mean = new_prior_alpha / (new_prior_alpha + new_prior_beta) if (new_prior_alpha + new_prior_beta) > 0 else 0
    st.write(f"Mean of New Prior: {new_prior_mean:.3f}")

    fig2, ax2 = plt.subplots()
    if old_posterior_alpha > 0 and old_posterior_beta > 0: plot_beta_distribution(old_posterior_alpha, old_posterior_beta, "Old Posterior (Data from T-1)", "اللاحق القديم (بيانات من T-1)", ax2)
    if initial_prior_alpha > 0 and initial_prior_beta > 0: plot_beta_distribution(initial_prior_alpha, initial_prior_beta, "Fixed Initial Prior", "المسبق الأولي الثابت", ax2)
    
    new_prior_label_en = f"New Prior (δ={discount_factor:.1f})"
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
    st.markdown("""
    <div dir='rtl' style='text-align: right;'>
    <p><b>التوضيح التفاعلي (القسم ٦):</b></p>
    <p>يوفر هذا القسم أدوات تفاعلية لفهم كيفية تحديث توزيع Beta المسبق إلى توزيع لاحق مع بيانات جديدة. يمكنك تعديل معلمات التوزيع المسبق (alpha و beta)، وإدخال نتائج مسح جديدة (عدد الاستطلاعات وعدد الراضين)، ورؤية كيف يتغير التوزيع اللاحق والمتوسط وفترة الموثوقية.</p>
    <p>الجزء الثاني يوضح تأثير "عامل الخصم" (Discount Factor &delta;) على البيانات القديمة عند تكوين توزيع مسبق لفترة جديدة.</p>
    <p>يتم عرض الصيغ الرياضية المستخدمة في النص الإنجليزي أعلاه. على سبيل المثال، التوزيع اللاحق يتبع:</p>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"Beta(\alpha_0 + k, \beta_0 + n - k)")
    st.markdown("""
    <div dir='rtl' style='text-align: right;'>
    <p>حيث &alpha;<sub>0</sub> و &beta;<sub>0</sub> هي معلمات التوزيع المسبق، k عدد الراضين، و n إجمالي عدد الاستطلاعات الجديدة.</p>
    </div>
    """, unsafe_allow_html=True)


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
    st.markdown("""
    <div dir='rtl' style='text-align: right;'>
    <p><b>الخلاصة (القسم ٧):</b></p>
    <p>يلخص هذا القسم الفوائد الرئيسية لإطار تقدير Bayesian التكيفي المقترح، مع التأكيد على قدرته على توفير رؤى أكثر دقة وفي الوقت المناسب لتحسين تجربة الحج. يقر المقترح بأن المنهجية تتطلب تنفيذًا دقيقًا، لكنه يسلط الضوء على الفوائد طويلة الأجل. يوصي بالبدء بمشروع تجريبي.</p>
    </div>
    """, unsafe_allow_html=True)


# --- Streamlit App Structure ---
# Language selection can be added here if desired, but for simplicity, we'll keep it English first
# and add Arabic explanations below each English section as requested.

# For now, we remove the language selection and current_lang logic
# to strictly follow the "English app + Arabic guide below" structure.

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
# Use the English keys for the radio button options
selection = st.sidebar.radio("Go to", list(PAGES.keys()), key="main_nav_eng_only")

page_function = PAGES[selection]
page_function()

st.sidebar.markdown("---")
# Defaulting to English sidebar info as per "original English app"
st.sidebar.info(
    "This app presents a proposal for using Bayesian adaptive estimation "
    "for Hajj pilgrim satisfaction surveys. Developed by Dr. Mohammad Nabhan."
)
