import streamlit as st
import numpy as np
from scipy.stats import norm
import math

# --- Page Configuration ---
st.set_page_config(
    page_title="A/B Testing Guide & Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Placeholder for Future Content (Backburner List) ---
FUTURE_FEATURES = {
    "Advanced Factor Modeling (in 'Analyze Results')": "Implement logistic regression/ANOVA for analyzing simultaneous factor impacts and interactions.",
    "Estimated Test Duration (in 'Sample Size Calculator')": "Calculate how long a test might need to run based on traffic.",
    "Relative MDE Input (in 'Sample Size Calculator')": "Provide relative MDE as an alternative input for determining sample size.",
    "Option B for Segmentation Display (Cross-Tabulation)": "Automatically create segments for all combinations of multiple selected factors.",
    "Arabic Language Support": "Add Arabic localization to the application.",
    "Expected Loss/Uplift (Advanced Bayesian Metrics)": "Include more detailed Bayesian decision theory metrics.",
    "Support for Continuous Outcomes (Bayesian - Advanced Models)": "Explore more complex Bayesian models or MCMC for continuous data if simpler models are insufficient.",
    "Support for Ratio Metrics": "Enable analysis for metrics that are ratios of two continuous variables (e.g., revenue per transaction)."
}

# --- Helper Function for Sample Size Calculation (Binary) ---
def calculate_binary_sample_size(baseline_cr, mde_abs, power, alpha, num_variations):
    """
    Calculates the required sample size per variation for a binary outcome A/B/n test.
    Assumes pairwise comparisons against a single control group.
    MDE is absolute.
    Alpha is for a two-sided test.
    """
    if baseline_cr <= 0 or baseline_cr >= 1:
        return None, "Baseline Conversion Rate (BCR) must be between 0 and 1 (exclusive)."
    if mde_abs <= 0:
        return None, "Minimum Detectable Effect (MDE) must be positive."
    if power <= 0 or power >= 1:
        return None, "Statistical Power must be between 0 and 1 (exclusive)."
    if alpha <= 0 or alpha >= 1:
        return None, "Significance Level (Alpha) must be between 0 and 1 (exclusive)."
    if num_variations < 2:
        return None, "Number of variations (including control) must be at least 2."

    p1 = baseline_cr
    p2 = baseline_cr + mde_abs

    if p2 >= 1 or p2 <=0:
        return None, f"MDE ({mde_abs*100:.2f}%) results in an invalid target conversion rate ({p2*100:.2f}%). Adjust BCR or MDE."

    z_alpha_half = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    
    variance_p1 = p1 * (1 - p1)
    variance_p2 = p2 * (1 - p2)
    
    numerator = (z_alpha_half + z_beta)**2 * (variance_p1 + variance_p2)
    denominator = mde_abs**2
    
    if denominator == 0: 
        return None, "MDE cannot be zero."

    n_per_variation = numerator / denominator
    
    return math.ceil(n_per_variation), None


# --- Page Functions ---
def show_introduction_page():
    st.header("Introduction to A/B Testing 🧪")
    st.markdown("This tool is designed to guide users in understanding and effectively conducting A/B tests.")
    st.markdown("---")

    st.subheader("What is A/B Testing?")
    st.markdown("""
    A/B testing (also known as split testing or bucket testing) is a method of comparing two or more versions of something—like a webpage, app feature, email headline, or call-to-action button—to determine which one performs better in achieving a specific goal.
    The core idea is to make **data-driven decisions** rather than relying on gut feelings or opinions. You show one version (the 'control' or 'A') to one group of users, and another version (the 'variation' or 'B') to a different group of users, simultaneously. Then, you measure how each version performs based on your key metric (e.g., conversion rate).
    """)
    st.markdown("""
    *Analogy:* Imagine you're a chef with two different recipes for a cake (Recipe A and Recipe B). You want to know which one your customers like more. You bake both cakes and offer a slice of Recipe A to one group of customers and a slice of Recipe B to another. Then, you ask them which one they preferred or count how many slices of each were eaten. That's essentially what A/B testing does for digital experiences!
    """)
    st.markdown("---")

    st.subheader("Why Use A/B Testing? (The Benefits)")
    st.markdown("""
    A/B testing is a powerful tool because it can help you:
    * ✅ **Improve Key Metrics:** Increase conversion rates, boost engagement, drive sales, or improve any other metric you care about.
    * ρί **Reduce Risk:** Test changes on a smaller scale before rolling them out to your entire user base, minimizing the impact of potentially negative changes.
    * 💡 **Gain Insights:** Understand your users' behavior, preferences, and motivations better. Even a "failed" test can provide valuable learnings.
    * ✨ **Optimize User Experience:** Make your website, app, or product more user-friendly and effective.
    * 🔄 **Foster Iterative Improvement:** A/B testing supports a cycle of continuous learning and optimization.
    """)
    st.markdown("---")

    st.subheader("Basic A/B Testing Terminology")
    st.markdown("Here are a few key terms you'll encounter frequently. More detailed explanations are available and will appear in context throughout the app.")
    
    basic_terms = {
        "Control (Version A)": "The existing, unchanged version that you're comparing against. It acts as a baseline.",
        "Variation (Version B, C, etc.)": "A modified version that you're testing to see if it performs differently than the control.",
        "Conversion / Goal / Metric": "The specific action or outcome you are measuring to determine success (e.g., a sign-up, a purchase, a click).",
        "Conversion Rate (CR)": "The percentage of users who complete the desired goal, out of the total number of users in that group."
    }

    for term, definition in basic_terms.items():
        st.markdown(f"**{term}:** {definition}")

    with st.expander("📖 Learn more about other common A/B testing terms... (Placeholder - Full list coming in a future cycle)"):
        st.markdown("""
        * **Lift / Uplift:** The percentage increase (or decrease) in performance of a variation compared to the control.
        * **Statistical Significance (p-value, Alpha):** A measure of whether an observed difference is likely due to a real effect or just random chance.
        * **Confidence Interval:** A range of values that likely contains the true value of a metric (like the true difference in conversion rates).
        * **Statistical Power (Beta):** The probability that your test will detect a real difference if one actually exists.
        * **Minimum Detectable Effect (MDE):** The smallest change you want your test to be able to reliably detect.
        * *(More terms will be added and explained in context throughout the app's sections.)*
        """)
    st.markdown("---")

    st.subheader("The A/B Testing Process at a Glance")
    st.markdown("""
    A typical A/B testing process involves several key steps. This app is designed to help you with some of these:
    1.  🤔 **Define Your Goal & Formulate a Hypothesis:** What do you want to improve, and what change do you believe will achieve it?
    2.  📐 **Design Your Test & Calculate Sample Size:** Determine how many users you need for a reliable test. (➡️ *The "Designing Your A/B Test" section will help here!*)
    3.  🚀 **Run Your Test & Collect Data:** Implement the test and gather data on how each variation performs. (This step happens on your platform/website.)
    4.  📊 **Analyze Your Results:** Process the collected data to compare the performance of your variations. (➡️ *The "Analyze Results" section is built for this!*)
    5.  🧐 **Interpret Results & Make a Decision:** Understand what the results mean and decide on the next steps. (➡️ *The "Interpreting Results & Detailed Decision Guidance" section will guide you.*)
    """)
    st.markdown("---")
    
    st.subheader("Where This App Fits In")
    st.markdown("""
    This application aims to be your companion for the critical stages of A/B testing:
    * Helping you **design robust tests** by calculating the necessary sample size.
    * Enabling you to **analyze the data** you've collected using both Frequentist and Bayesian statistical approaches.
    * Guiding you in **interpreting those results** to make informed, data-driven decisions.
    * Providing **educational content** (like common pitfalls and FAQs) to improve your A/B testing knowledge.
    """)

# Ensure these imports are at the top of your script
import streamlit as st
import numpy as np
from scipy.stats import norm
import math

# ... (Keep other parts of your script like Page Configuration, FUTURE_FEATURES, helper functions, other page functions)

def show_design_test_page():
    st.header("Designing Your A/B Test 📐")
    st.markdown("A crucial step in designing an A/B test is determining the appropriate sample size. This calculator will help you estimate the number of users needed per variation for tests with **binary outcomes** (e.g., conversion rates, click-through rates).")
    st.markdown("---")
    
    st.subheader("Sample Size Calculator (for Binary Outcomes)")

    st.markdown("**Calculator Inputs:**")
    
    cols = st.columns(2)
    with cols[0]:
        baseline_cr_percent = st.number_input(
            label="Baseline Conversion Rate (BCR) (%)",
            min_value=0.1, max_value=99.9, value=5.0, step=0.1, format="%.1f",
            help="The current conversion rate of your control group (Version A). For example, if 5 out of 100 users convert, your BCR is 5%."
        )
    with cols[1]:
        mde_abs_percent = st.number_input(
            label="Minimum Detectable Effect (MDE) - Absolute (%)",
            min_value=0.1, max_value=50.0, value=1.0, step=0.1, format="%.1f",
            help="The smallest *absolute* improvement you want to detect (e.g., a 1% absolute increase from BCR). A smaller MDE requires a larger sample size."
        )
    
    cols2 = st.columns(2)
    with cols2[0]:
        power_percent = st.slider(
            # Using spelled out "Beta" in label for clarity as LaTeX in labels is not directly rendered
            label="Statistical Power (1 - Beta) (%)", 
            min_value=50, max_value=99, value=80, step=1, format="%d%%",
            help="The probability of detecting an effect if there is one (typically 80-90%). Higher power reduces the chance of a false negative but requires more samples."
        )
    with cols2[1]:
        alpha_percent = st.slider(
            # Using spelled out "Alpha" in label
            label="Significance Level (Alpha) (%) - Two-sided", 
            min_value=1, max_value=20, value=5, step=1, format="%d%%",
            help="The probability of detecting an effect when there isn't one (typically 1-5%). This is your risk tolerance for a false positive. A two-sided test is assumed."
        )
        
    num_variations = st.number_input(
        label="Number of Variations (including Control)",
        min_value=2, value=2, step=1,
        help="Total number of versions you are testing (e.g., Control + 1 Variation = 2; Control + 2 Variations = 3)."
    )

    st.markdown(
        "<p style='font-size: smaller; font-style: italic;'>Note: This calculator determines sample size based on pairwise comparisons against the control group, each at the specified significance level (α).</p>", 
        unsafe_allow_html=True
    )

    if st.button("Calculate Sample Size"):
        baseline_cr = baseline_cr_percent / 100.0
        mde_abs = mde_abs_percent / 100.0
        power = power_percent / 100.0
        alpha = alpha_percent / 100.0

        sample_size_per_variation, error_message = calculate_binary_sample_size(
            baseline_cr, mde_abs, power, alpha, num_variations
        )

        if error_message:
            st.error(error_message)
        elif sample_size_per_variation is not None:
            st.success(f"Calculation Successful!")
            target_cr_percent = (baseline_cr + mde_abs) * 100
            
            res_cols = st.columns(2)
            with res_cols[0]:
                st.metric(label="Required Sample Size PER Variation", value=f"{sample_size_per_variation:,}")
            with res_cols[1]:
                st.metric(label="Total Required Sample Size", value=f"{(sample_size_per_variation * num_variations):,}")
            
            st.markdown(f"""
            **Summary of Inputs Used:**
            - Baseline Conversion Rate (BCR): `{baseline_cr_percent:.1f}%`
            - Absolute MDE: `{mde_abs_percent:.1f}%` (Targeting a CR of at least `{target_cr_percent:.2f}%` for variations)
            - Statistical Power: `{power_percent}%` (1 - $\beta$)
            - Significance Level (α): `{alpha_percent}%` (two-sided)
            - Number of Variations: `{num_variations}`
            
            This means you'll need approximately **{sample_size_per_variation:,} users/observations for your control group** and **{sample_size_per_variation:,} users/observations for each of your other test variations** to confidently detect the specified MDE.
            """)
            
            with st.expander("Show Formula Used for Sample Size Calculation"):
                st.markdown("The sample size ($n$) per variation for comparing two proportions is commonly calculated using:")
                st.latex(r'''
                n = \frac{(Z_{\alpha/2} + Z_{\beta})^2 \cdot (p_1(1-p_1) + p_2(1-p_2))}{(p_2 - p_1)^2}
                ''')
                st.markdown("**Where:**") 
                st.markdown(r"""
                - $n$ = Sample size per variation
                - $p_1$ = Baseline Conversion Rate (BCR) of the control group (as a proportion, e.g., 0.05 for 5%)
                - $p_2$ = Expected conversion rate of the variation group ($p_1 + \text{MDE}$, as a proportion)
                - $Z_{\alpha/2}$ = Z-score corresponding to the chosen significance level $\alpha$ for a two-sided test (e.g., 1.96 for $\alpha=0.05$)
                - $Z_{\beta}$ = Z-score corresponding to the chosen statistical power (1 - $\beta$) (e.g., 0.84 for 80% power)
                - MDE (Minimum Detectable Effect) = $p_2 - p_1$ (absolute difference, as a proportion)
                """)
        else:
            st.error("An unexpected error occurred during calculation.")

    st.markdown("---")
    with st.expander("💡 Understanding Input Impacts on Sample Size"):
        st.markdown(r"""
        Adjusting the input parameters for the sample size calculator has direct consequences on the number of users you'll need. Understanding these trade-offs is key for planning your A/B tests effectively:

        * **Baseline Conversion Rate (BCR):**
            * *Impact:* The required sample size tends to be largest when BCR is close to 50% (for a given MDE). It decreases as BCR moves towards 0% or 100%.
            * *Trade-off:* This is usually an existing fact about your current performance. While you don't typically 'trade it off', knowing this helps understand why tests for metrics around 50% CR might require more users than metrics with very low or very high CRs.

        * **Minimum Detectable Effect (MDE):**
            * *Impact:* This is one of the most influential factors.
                * *Decreasing* MDE (wanting to detect smaller improvements) **significantly increases** the required sample size.
                * *Increasing* MDE (being okay with only detecting larger improvements) **decreases** the sample size.
            * *Trade-off:* A smaller MDE allows you to find more subtle, incremental wins, but at the cost of needing more users and potentially longer test durations. A larger MDE is cheaper/faster but you risk missing smaller, yet potentially valuable, effects. Consider the business value of the smallest change you'd care to implement.

        * **Statistical Power (1 - $\beta$):**
            * *Impact:* *Increasing* power **increases** the required sample size.
            * *Trade-off:* Higher power (e.g., 90% vs. 80%) reduces your risk of a Type II error (a "false negative" – failing to detect a real improvement when one exists). This increased confidence comes at the cost of more samples. Lowering power makes tests cheaper but increases the risk of missing out on actual winning variations. 80% is a common standard.

        * **Significance Level ($\alpha$):**
            * *Impact:* *Decreasing* $\alpha$ (e.g., from 5% to 1%) **increases** the required sample size. (A lower $\alpha$ means you're being more stringent).
            * *Trade-off:* A lower $\alpha$ reduces your risk of a Type I error (a "false positive" – concluding there's an improvement when there isn't one). This means you'll have more confidence in any "winning" result you declare. However, this greater certainty requires more samples. Increasing $\alpha$ (e.g., to 10%) reduces sample size but increases the risk of implementing a change that isn't truly better. 5% is a common standard.
        
        * **Number of Variations:**
            * *Impact:* The sample size *per variation* (as calculated by the formula above) remains the same. However, the **total sample size** for the entire experiment increases proportionally with the number of variations.
            * *Trade-off:* Testing more variations allows you to explore more ideas simultaneously. However, it requires more overall traffic/time and can increase the complexity of analysis and decision-making. Each additional variation needs to "earn its keep" by representing a distinct, valuable hypothesis.
        
        Balancing these factors is key to designing a test that is both statistically sound and practically feasible for your resources and timelines.
        """) # Added r""" for raw string to ensure LaTeX renders correctly

    st.markdown("---")
    st.info("Coming in future cycles: Sample Size Calculator for Continuous Outcomes, 'Common Pitfalls' content.")

# ... (The rest of your app code: show_introduction_page, show_analyze_results_page, etc. and the main navigation logic)
# Ensure you replace the old show_design_test_page with this new one in your main script.


def show_analyze_results_page():
    st.header("Analyze Results 📊")
    st.write("Functionality for uploading data and analyzing A/B test results will be implemented starting in Cycle 4.")
    st.info("Coming soon: Data upload, Frequentist and Bayesian analysis, segmentation!")

def show_interpret_results_page():
    st.header("Interpreting Results & Detailed Decision Guidance 🧐")
    st.write("Guidance on how to interpret your A/B test results and make decisions will be implemented in Cycle 9.")
    st.info("Coming soon: Understanding statistical vs. practical significance, next steps!")

def show_faq_page():
    st.header("FAQ on Common Misinterpretations ❓")
    st.write("Answers to frequently asked questions about common misinterpretations of A/B test results will be implemented in Cycle 3.")
    st.info("Coming soon!")

def show_roadmap_page():
    st.header("Roadmap / Possible Future Features 🚀")
    st.markdown("This application has several potential features planned for future development:")
    
    if FUTURE_FEATURES:
        for feature, description in FUTURE_FEATURES.items():
            st.markdown(f"- **{feature}:** {description}")
    else:
        st.write("No future features currently listed.")
    st.markdown("---")
    st.markdown("Feedback on feature prioritization is welcome.")


# --- Main App Navigation ---
st.sidebar.title("Navigation")
PAGES = {
    "Introduction to A/B Testing": show_introduction_page,
    "Designing Your A/B Test": show_design_test_page,
    "Analyze Results": show_analyze_results_page,
    "Interpreting Results & Detailed Decision Guidance": show_interpret_results_page,
    "FAQ on Misinterpretations": show_faq_page,
    "Roadmap / Possible Future Features": show_roadmap_page
}

selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page_function = PAGES[selection]
page_function()

st.sidebar.markdown("---")
st.sidebar.info("A/B Testing Guide & Analyzer | V0.2.1 (Cycle 2 - Enhanced)")
