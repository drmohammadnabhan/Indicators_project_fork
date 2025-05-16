import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

# Page Configuration
st.set_page_config(layout="wide", page_title="Adaptive Bayesian Estimation Proposal")

# --- Helper Functions for Interactive Illustrations ---
def plot_beta_distribution(alpha, beta, label, ax):
    """Plots a Beta distribution."""
    x = np.linspace(0, 1, 500)
    y = stats.beta.pdf(x, alpha, beta)
    ax.plot(x, y, label=f'{label} (α={alpha:.2f}, β={beta:.2f})')
    ax.fill_between(x, y, alpha=0.2)

def update_beta_parameters(prior_alpha, prior_beta, successes, failures):
    """Updates Beta parameters given new data."""
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + failures
    return posterior_alpha, posterior_beta

def get_credible_interval(alpha, beta, conf_level=0.95):
    """Calculates the credible interval for a Beta distribution."""
    if alpha <= 0 or beta <= 0: # Invalid parameters
        return (0,0)
    return stats.beta.interval(conf_level, alpha, beta)

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
    st.image("https_miro.medium.com_v2_resize_fit_1400_1__f_xL41kP9n2_n3L9yY0gLg.png", caption="Conceptual Flow of Bayesian Updating (Source: Medium - adapted for context)")


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
        2.  **Approximation Formulas:** Some researchers have developed approximations. For instance, one common approach for proportions aims for a certain margin of error (half-width) $E_{target}$ in the credible interval. If the current variance of the posterior is $Var(\theta | D_{current})$, and we approximate the variance of the posterior after $n_{add}$ additional samples as roughly $\frac{Var(\theta | D_{current}) \times N_0}{N_0 + n_{add}}$ (where $N_0 = \alpha_{post} + \beta_{post}$ is the "effective prior sample size"), one can solve for $n_{add}$ that makes the future standard deviation (and thus interval width) small enough.
        3.  **Bayesian Sequential Analysis:** More formal methods from Bayesian sequential analysis (e.g., Bayesian sequential probability ratio tests - BSPRTs) can be adapted, though they might be more complex to implement initially.
        4.  **Pragmatic Batching:** Collect data in smaller, manageable batches (e.g., 30-50 responses). After each batch, reassess precision. This is often a practical starting point.

    The tool should aim to provide guidance on a reasonable next batch size based on the current uncertainty and the distance to the target precision.
    """)

    st.subheader("3.5. Handling Data Heterogeneity Over Time")
    st.markdown("""
    A key challenge is that service provider performance or general pilgrim satisfaction might change over time. Using historical data uncritically as a prior might be misleading if changes have occurred.

    * **The "Learning Hyperparameter" (Discount Factor / Power Prior):**
        One way to address this is to down-weight older data. If we have a series of data batches $D_1, D_2, \dots, D_t$ (from oldest to newest), when forming a prior for the current period $t+1$ based on data up to $t$, we can use a "power prior" approach or a simpler discount factor.
        For example, if using the posterior from period $t$ (with parameters $\alpha_t, \beta_t$) as a prior for period $t+1$, we might introduce a discount factor $\delta \in [0, 1]$:
        $$ \alpha_{prior, t+1} = \delta \times \alpha_t + (1-\delta) \times \alpha_{initial} $$
        $$ \beta_{prior, t+1} = \delta \times \beta_t + (1-\delta) \times \beta_{initial} $$
        Where $(\alpha_{initial}, \beta_{initial})$ could be parameters of a generic, uninformative prior.
        * If $\delta = 1$, all past information is carried forward fully.
        * If $\delta = 0$, all past information is discarded, and we restart with the initial prior (re-estimation).
        * Values between 0 and 1 provide a trade-off. The "learning hyperparameter" $\delta$ can be fixed, tuned, or even learned from the data if a more complex model is used. A simpler approach is to use a fixed $\delta$, e.g., $\delta=0.8$ or $\delta=0.9$, reflecting a belief that recent data is more relevant.

    * **Change-Point Detection:**
        Periodically, statistical tests can be run to detect if there has been a significant change in the underlying satisfaction or performance metric. If a change point is detected (e.g., using CUSUM charts on posterior means, or more formal Bayesian change-point models), the prior for subsequent estimations might be reset to be less informative, or data before the change point heavily discounted or discarded.

    * **Hierarchical Bayesian Models (Advanced):**
        These models can explicitly model variation over time or across different service providers simultaneously, allowing "borrowing strength" across units while also estimating individual trajectories. This is a more sophisticated approach suitable for later phases.

    The choice of method depends on the complexity deemed appropriate and the available data. Starting with a discount factor is often a pragmatic first step.
    """)

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
        prior_alpha = st.slider("Prior Alpha (α₀)", min_value=0.1, max_value=50.0, value=1.0, step=0.1, key="prior_a")
        prior_beta = st.slider("Prior Beta (β₀)", min_value=0.1, max_value=50.0, value=1.0, step=0.1, key="prior_b")
        prior_mean = prior_alpha / (prior_alpha + prior_beta)
        st.write(f"Prior Mean: {prior_mean:.3f}")
        prior_ci = get_credible_interval(prior_alpha, prior_beta)
        st.write(f"95% Credible Interval (Prior): [{prior_ci[0]:.3f}, {prior_ci[1]:.3f}], Width: {prior_ci[1]-prior_ci[0]:.3f}")


    with col2:
        st.subheader("New Survey Data (Likelihood)")
        st.markdown("Enter the results from a new batch of surveys.")
        num_surveys = st.slider("Number of New Surveys (n)", min_value=1, max_value=500, value=50, step=1, key="surveys_n")
        num_satisfied = st.slider("Number Satisfied in New Surveys (k)", min_value=0, max_value=num_surveys, value=int(num_surveys/2), step=1, key="surveys_k")
        num_not_satisfied = num_surveys - num_satisfied
        st.write(f"Observed Satisfaction in New Data: {num_satisfied/num_surveys if num_surveys > 0 else 0:.3f}")

    st.markdown("---")
    st.subheader("Posterior Beliefs (After Update)")
    posterior_alpha, posterior_beta = update_beta_parameters(prior_alpha, prior_beta, num_satisfied, num_not_satisfied)
    st.markdown(f"The posterior distribution is $Beta(\\alpha_0 + k, \\beta_0 + n - k) = Beta({posterior_alpha:.1f}, {posterior_beta:.1f})$")

    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    st.write(f"Posterior Mean: {posterior_mean:.3f}")
    posterior_ci = get_credible_interval(posterior_alpha, posterior_beta)
    st.write(f"95% Credible Interval (Posterior): [{posterior_ci[0]:.3f}, {posterior_ci[1]:.3f}], Width: {posterior_ci[1]-posterior_ci[0]:.3f}")

    target_width = st.number_input("Target Credible Interval Width for Stopping", min_value=0.01, max_value=1.0, value=0.10, step=0.01)
    current_width = posterior_ci[1] - posterior_ci[0]
    if current_width <= target_width :
        if current_width > 0: # Avoid issues if CI is (0,0) due to bad alpha/beta
             st.success(f"Target precision met! Current width ({current_width:.3f}) ≤ Target width ({target_width:.3f}).")
        elif prior_alpha > 0 and prior_beta >0 : # only show if prior was valid
             st.success(f"Target precision met! Current width ({current_width:.3f}) ≤ Target width ({target_width:.3f}).")
    else:
        st.warning(f"Target precision not yet met. Current width ({current_width:.3f}) > Target width ({target_width:.3f}). Consider more samples.")


    fig, ax = plt.subplots()
    plot_beta_distribution(prior_alpha, prior_beta, "Prior", ax)
    plot_beta_distribution(posterior_alpha, posterior_beta, "Posterior", ax)
    ax.set_title("Prior and Posterior Distributions of Satisfaction Rate")
    ax.set_xlabel("Satisfaction Rate (θ)")
    ax.set_ylabel("Density")
    ax.legend()
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

    discount_factor = st.slider("Discount Factor (δ) for Old Data", min_value=0.0, max_value=1.0, value=0.8, step=0.05,
                                 help="Controls weight of old data. 1.0 = full weight, 0.0 = discard old data, rely only on initial prior.")

    initial_prior_alpha = st.number_input("Initial Prior Alpha (for new period if discounting heavily)", min_value=0.1, value=1.0, step=0.1, key="init_prior_a")
    initial_prior_beta = st.number_input("Initial Prior Beta (for new period if discounting heavily)", min_value=0.1, value=1.0, step=0.1, key="init_prior_b")

    new_prior_alpha = discount_factor * old_posterior_alpha + (1 - discount_factor) * initial_prior_alpha
    new_prior_beta = discount_factor * old_posterior_beta + (1 - discount_factor) * initial_prior_beta

    st.write(f"New Prior for Next Period: $Beta({new_prior_alpha:.2f}, {new_prior_beta:.2f})$")
    new_prior_mean = new_prior_alpha / (new_prior_alpha + new_prior_beta)
    st.write(f"Mean of New Prior: {new_prior_mean:.3f}")

    fig2, ax2 = plt.subplots()
    plot_beta_distribution(old_posterior_alpha, old_posterior_beta, "Old Posterior (Data from T-1)", ax2)
    plot_beta_distribution(initial_prior_alpha, initial_prior_beta, "Fixed Initial Prior", ax2)
    plot_beta_distribution(new_prior_alpha, new_prior_beta, f"New Prior (δ={discount_factor})", ax2)
    ax2.set_title("Forming a New Prior with Discounting")
    ax2.set_xlabel("Satisfaction Rate (θ)")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)


def conclusion():
    st.header("7. Conclusion")
    st.markdown("""
    The proposed Bayesian adaptive estimation framework offers a sophisticated, flexible, and efficient approach to analyzing pilgrim satisfaction surveys. By iteratively updating beliefs and dynamically adjusting sampling efforts, this methodology promises more precise and timely insights, enabling better-informed decision-making for enhancing the Hajj experience.

    While it introduces new concepts and requires careful implementation, the long-term benefits—including optimized resource use and a deeper understanding of satisfaction dynamics—are substantial. This proposal advocates for a phased implementation, starting with core metrics and gradually building complexity and scope.

    We recommend proceeding with a pilot project to demonstrate the practical benefits and refine the operational aspects of this advanced analytical approach.
    """)

# --- Streamlit App Structure ---
st.title("Proposal: Adaptive Bayesian Estimation for Pilgrim Satisfaction Surveys")

PAGES = {
    "1. Introduction & Objectives": introduction_objectives,
    "2. Challenges Addressed": challenges_addressed,
    "3. Bayesian Adaptive Methodology": bayesian_adaptive_methodology,
    "4. Implementation Roadmap": implementation_roadmap,
    "5. Note to Practitioners": note_to_practitioners,
    "6. Interactive Illustration": interactive_illustration,
    "7. Conclusion": conclusion
}

st.sidebar.title("Proposal Sections")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page_function = PAGES[selection]
page_function()

st.sidebar.markdown("---")
st.sidebar.info(
    "This app presents a proposal for using Bayesian adaptive estimation "
    "for Hajj pilgrim satisfaction surveys. Developed by your AI consultant."
)
