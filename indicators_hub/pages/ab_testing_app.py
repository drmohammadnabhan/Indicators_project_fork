import streamlit as st
import numpy as np
from scipy.stats import norm, beta as beta_dist
import math
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
import matplotlib.pyplot as plt

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
        if p2 >=1: return None, f"MDE ({mde_abs*100:.2f}%) results in a target conversion rate of {p2*100:.2f}% or more, which is not possible. Adjust BCR or MDE."
        if p2 <=0: return None, f"MDE ({mde_abs*100:.2f}%) results in a target conversion rate of {p2*100:.2f}% or less, which is not possible. Adjust BCR or MDE."
    z_alpha_half = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    variance_p1 = p1 * (1 - p1)
    variance_p2 = p2 * (1 - p2)
    numerator = (z_alpha_half + z_beta)**2 * (variance_p1 + variance_p2)
    denominator = mde_abs**2
    if denominator == 0: return None, "MDE cannot be zero (already checked by mde_abs > 0)."
    n_per_variation = numerator / denominator
    return math.ceil(n_per_variation), None

# --- Helper Function for Bayesian Analysis (Binary) ---
def run_bayesian_binary_analysis(summary_stats, control_group_name, prior_alpha=1, prior_beta=1, n_samples=10000, ci_level=0.95):
    results = {}
    # Ensure summary_stats has a 'Variation' column correctly named for processing
    if 'Variation' not in summary_stats.columns:
        # Attempt to find it if it was renamed (e.g., from the original variation_col)
        # This part might need adjustment if the original column name is stored differently
        original_var_col_name = summary_stats.columns[0] # Fallback: assume it's the first column if not 'Variation'
        if original_var_col_name != 'Variation':
            summary_stats = summary_stats.rename(columns={original_var_col_name: 'Variation'})
            
    posterior_params = {}
    
    for index, row in summary_stats.iterrows():
        var_name = row['Variation'] 
        users = int(row['Users'])
        conversions = int(row['Conversions'])
        
        alpha_post = prior_alpha + conversions
        beta_post = prior_beta + (users - conversions)
        posterior_params[var_name] = {'alpha': alpha_post, 'beta': beta_post}
        
        samples = beta_dist.rvs(alpha_post, beta_post, size=n_samples)
        results[var_name] = {
            'samples': samples,
            'mean_cr': np.mean(samples),
            'median_cr': np.median(samples),
            'cr_ci_low': beta_dist.ppf((1-ci_level)/2, alpha_post, beta_post),
            'cr_ci_high': beta_dist.ppf(1-(1-ci_level)/2, alpha_post, beta_post),
            'alpha_post': alpha_post,
            'beta_post': beta_post,
            'diff_samples_vs_control': None # Initialize
        }

    if control_group_name not in results:
        return None, f"Control group '{control_group_name}' not found in Bayesian results data. Available groups: {list(results.keys())}"

    control_samples = results[control_group_name]['samples']
    
    for var_name, data in results.items():
        if var_name == control_group_name:
            data['prob_better_than_control'] = None 
            data['uplift_ci_low'] = None
            data['uplift_ci_high'] = None
            data['expected_uplift_abs'] = None
            continue

        var_samples = data['samples']
        diff_samples = var_samples - control_samples
        data['diff_samples_vs_control'] = diff_samples # Store for plotting
        
        data['prob_better_than_control'] = np.mean(diff_samples > 0)
        data['uplift_ci_low'] = np.percentile(diff_samples, (1-ci_level)/2 * 100)
        data['uplift_ci_high'] = np.percentile(diff_samples, (1-(1-ci_level)/2) * 100)
        data['expected_uplift_abs'] = np.mean(diff_samples)

    all_var_names = summary_stats['Variation'].tolist()
    all_samples_matrix = np.array([results[var]['samples'] for var in all_var_names if var in results]) # Ensure var exists
    
    best_variation_counts = np.zeros(len(all_var_names))
    
    if all_samples_matrix.ndim == 2 and all_samples_matrix.shape[0] > 0 and all_samples_matrix.shape[1] == n_samples:
        for i in range(n_samples):
            # Get samples for this iteration for all variations that are in results
            current_iter_samples = all_samples_matrix[:, i]
            best_idx_in_matrix = np.argmax(current_iter_samples)
            # Map this index back to the original all_var_names list
            # This assumes the order in all_samples_matrix matches all_var_names for vars in results
            # A safer way is to map by name directly if orders can mismatch
            original_idx_map = {name: i for i, name in enumerate(all_var_names)}
            # Find which name corresponds to best_idx_in_matrix based on the order of all_samples_matrix
            # This part is tricky if all_var_names has items not in results.keys()
            # For simplicity, let's assume all_var_names are keys in results for now
            # A more robust way would be to iterate through results.keys() for the matrix rows.
            
            # Let's rebuild all_samples_matrix using a defined order of variations present in results
            ordered_var_names_in_results = [name for name in all_var_names if name in results]
            if not ordered_var_names_in_results: # Should not happen if control is found
                 return results, "No variations found in results for P(Best) calculation."

            temp_matrix = np.array([results[var]['samples'] for var in ordered_var_names_in_results])
            if temp_matrix.ndim == 2 and temp_matrix.shape[0] > 0:
                best_idx_in_temp_matrix = np.argmax(temp_matrix[:, i])
                best_var_name_this_iter = ordered_var_names_in_results[best_idx_in_temp_matrix]
                original_idx_for_counts = all_var_names.index(best_var_name_this_iter)
                best_variation_counts[original_idx_for_counts] += 1
            
        prob_best = best_variation_counts / n_samples
        
        for i, var_name in enumerate(all_var_names):
            if var_name in results: # Ensure var_name is in results before assigning prob_best
                 results[var_name]['prob_best'] = prob_best[i]
            # If var_name was skipped in matrix (e.g. bad data), it won't have prob_best
    else: 
        for var_name in all_var_names:
            if var_name in results:
                results[var_name]['prob_best'] = 1.0 if len(all_var_names) == 1 and var_name in results else 0.0
    return results, None

# --- Page Functions (Introduction, Design Test - as in V0.4/Cycle 3) ---
def show_introduction_page():
    # ... (Full content from Cycle 1 - V0.2.2)
    st.header("Introduction to A/B Testing 🧪")
    st.markdown("This tool is designed to guide users in understanding and effectively conducting A/B tests.") 
    st.markdown("---")
    st.subheader("What is A/B Testing?") 
    st.markdown("A/B testing (also known as split testing or bucket testing) is a method of comparing two or more versions of something—like a webpage, app feature, email headline, or call-to-action button—to determine which one performs better in achieving a specific goal. The core idea is to make **data-driven decisions** rather than relying on gut feelings or opinions. You show one version (the 'control' or 'A') to one group of users, and another version (the 'variation' or 'B') to a different group of users, simultaneously. Then, you measure how each version performs based on your key metric (e.g., conversion rate).")
    st.markdown("*Analogy:* Imagine you're a chef with two different recipes for a cake (Recipe A and Recipe B). You want to know which one your customers like more. You bake both cakes and offer a slice of Recipe A to one group of customers and a slice of Recipe B to another. Then, you ask them which one they preferred or count how many slices of each were eaten. That's essentially what A/B testing does for digital experiences!")
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
    for term, definition in basic_terms.items(): st.markdown(f"**{term}:** {definition}")
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
    st.markdown("This application aims to be your companion for the critical stages of A/B testing: * Helping you **design robust tests** by calculating the necessary sample size. * Enabling you to **analyze the data** you've collected using both Frequentist and Bayesian statistical approaches. * Guiding you in **interpreting those results** to make informed, data-driven decisions. * Providing **educational content** (like common pitfalls and FAQs) to improve your A/B testing knowledge.")

def show_design_test_page():
    # ... (Full content from Cycle 3 / V0.3 - including Sample Size Calculator, Formula, Impacts, and Pitfalls expanders)
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
            help="The current conversion rate of your control group (Version A). For example, if 5 out of 100 users convert, your BCR is 5%." , key="ss_bcr"
        )
    with cols[1]:
        mde_abs_percent = st.number_input(
            label="Minimum Detectable Effect (MDE) - Absolute (%)",
            min_value=0.1, max_value=50.0, value=1.0, step=0.1, format="%.1f",
            help="The smallest *absolute* improvement you want to detect (e.g., a 1% absolute increase from BCR). A smaller MDE requires a larger sample size.", key="ss_mde"
        )
    cols2 = st.columns(2)
    with cols2[0]:
        power_percent = st.slider(
            label="Statistical Power (1 - \u03B2) (%)", 
            min_value=50, max_value=99, value=80, step=1, format="%d%%",
            help="The probability of detecting an effect if there is one (typically 80-90%). Higher power reduces the chance of a false negative but requires more samples.", key="ss_power"
        )
    with cols2[1]:
        alpha_percent = st.slider(
            label="Significance Level (\u03B1) (%) - Two-sided",
            min_value=1, max_value=20, value=5, step=1, format="%d%%",
            help="The probability of detecting an effect when there isn't one (typically 1-5%). This is your risk tolerance for a false positive. A two-sided test is assumed.", key="ss_alpha"
        )
    num_variations_ss = st.number_input( 
        label="Number of Variations (including Control)",
        min_value=2, value=2, step=1,
        help="Total number of versions you are testing (e.g., Control + 1 Variation = 2; Control + 2 Variations = 3).", key="ss_num_var"
    )
    st.markdown("<p style='font-size: smaller; font-style: italic;'>Note: This calculator determines sample size based on pairwise comparisons against the control group, each at the specified significance level (\u03B1).</p>", unsafe_allow_html=True)
    if st.button("Calculate Sample Size", key="ss_calc_button"):
        baseline_cr = baseline_cr_percent / 100.0
        mde_abs = mde_abs_percent / 100.0
        power = power_percent / 100.0
        alpha = alpha_percent / 100.0
        sample_size_per_variation, error_message = calculate_binary_sample_size(
            baseline_cr, mde_abs, power, alpha, num_variations_ss
        )
        if error_message: st.error(error_message)
        elif sample_size_per_variation is not None:
            st.success(f"Calculation Successful!")
            target_cr_percent = (baseline_cr + mde_abs) * 100
            res_cols = st.columns(2)
            with res_cols[0]: st.metric(label="Required Sample Size PER Variation", value=f"{sample_size_per_variation:,}")
            with res_cols[1]: st.metric(label="Total Required Sample Size", value=f"{(sample_size_per_variation * num_variations_ss):,}")
            st.markdown(f"**Summary of Inputs Used:**\n- Baseline Conversion Rate (BCR): `{baseline_cr_percent:.1f}%`\n- Absolute MDE: `{mde_abs_percent:.1f}%` (Targeting a CR of at least `{target_cr_percent:.2f}%` for variations)\n- Statistical Power: `{power_percent}%` (1 - \u03B2)\n- Significance Level (\u03B1): `{alpha_percent}%` (two-sided)\n- Number of Variations: `{num_variations_ss}`\n\nThis means you'll need approximately **{sample_size_per_variation:,} users/observations for your control group** and **{sample_size_per_variation:,} users/observations for each of your other test variations** to confidently detect the specified MDE.")
            with st.expander("Show Formula Used for Sample Size Calculation"):
                st.markdown("The sample size ($n$) per variation for comparing two proportions is commonly calculated using:")
                st.latex(r'''n = \frac{(Z_{\alpha/2} + Z_{\beta})^2 \cdot (p_1(1-p_1) + p_2(1-p_2))}{(p_2 - p_1)^2}''')
                st.markdown("**Where:**"); st.markdown(r"""
                - $n$ = Sample size per variation
                - $p_1$ = Baseline Conversion Rate (BCR) of the control group (as a proportion, e.g., 0.05 for 5%)
                - $p_2$ = Expected conversion rate of the variation group ($p_1 + \text{MDE}$, as a proportion)
                - $Z_{\alpha/2}$ = Z-score corresponding to the chosen significance level $\alpha$ for a two-sided test (e.g., 1.96 for $\alpha=0.05$)
                - $Z_{\beta}$ = Z-score corresponding to the chosen statistical power (1 - $\beta$) (e.g., 0.84 for 80% power)
                - MDE (Minimum Detectable Effect) = $p_2 - p_1$ (absolute difference, as a proportion)""")
        else: st.error("An unexpected error occurred during calculation.")
    st.markdown("---")
    with st.expander("💡 Understanding Input Impacts on Sample Size"):
        st.markdown(r"""Adjusting the input parameters for the sample size calculator has direct consequences on the number of users you'll need. Understanding these trade-offs is key for planning your A/B tests effectively:

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
        """)
    st.markdown("---")
    st.subheader("Common Pitfalls in A/B Test Design & Execution")
    st.markdown("Avoiding these common mistakes can significantly improve the quality and reliability of your A/B tests.")
    pitfalls = {
        "Too Short Test Duration / Insufficient Sample Size": {"what": "Ending a test before collecting enough data or running it for an arbitrarily short period.", "problem": "Results may be statistically underpowered or due to random noise.", "howto": "Calculate sample size beforehand and run tests for at least one full business cycle.", "analogy": "Judging a marathon by the first 100 meters."},
        "Ignoring Statistical Significance / Power": {"what": "Making decisions without considering if differences are statistically significant or if the test had enough power.", "problem": "You might implement changes with no real impact or discard truly better variations.", "howto": "Always check p-values/CIs (Frequentist) or probabilities/CrIs (Bayesian). Ensure adequate power.", "analogy": "Getting 2 heads in 3 coin flips doesn't mean a biased coin without more data."},
        "Testing Too Many Things at Once (in a Single Variation)": {"what": "Changing multiple elements in one variation.", "problem": "Impossible to isolate which change caused the performance difference.", "howto": "Test one significant change at a time to understand its specific impact.", "analogy": "Changing multiple ingredients in a recipe; if it's bad, which ingredient was it?"},
        "External Factors Affecting the Test": {"what": "Events outside your test (holidays, campaigns, outages) influencing behavior.", "problem": "Can skew results, making variations appear different due to external factors.", "howto": "Be aware of the test environment. Document external events. Consider restarting if data is heavily skewed.", "analogy": "Measuring plant growth with different fertilizers during a surprise heatwave."},
        "Regression to the Mean": {"what": "Initial extreme results tending to become closer to the average over time.", "problem": "Stopping tests early based on extreme initial results can be misleading.", "howto": "Run tests for their planned duration and sample size.", "analogy": "A basketball player's hot streak eventually cools down to their average performance."},
        "Not Segmenting Results (When Appropriate)": {"what": "Only looking at overall average results, not how different user segments reacted.", "problem": "Overall flat results might hide significant wins in one segment and losses in another.", "howto": "Analyze performance for important, predefined user segments. (This app's 'Analyze Results' section will help with this in a later cycle!). Ensure segments are large enough for meaningful analysis.", "analogy": "A new song might have mixed reviews overall, but specific age groups might love/hate it."},
        "Peeking at Results Too Often (and Stopping Early)": {"what": "Constantly monitoring results and stopping when significance is (randomly) hit with frequentist methods.", "problem": "Dramatically increases Type I error (false positive). This is 'p-hacking'.", "howto": "Determine sample size/duration in advance. Avoid decisions on interim results unless using specific sequential testing methods. Bayesian methods are more robust to peeking.", "analogy": "Stopping coin flips the moment you get 3 heads in a row and concluding bias."},
        "Simpson's Paradox": {"what": "A trend appears in groups but reverses/disappears when groups are combined.", "problem": "Aggregated data can lead to incorrect conclusions.", "howto": "Be aware of confounding variables; analyze important segments.", "analogy": "Hospital A has better overall survival but takes less severe cases. Hospital B might be better for *each* severity level."}
    }
    for pitfall, details in pitfalls.items():
        with st.expander(f"⚠️ {pitfall}"):
            st.markdown(f"**What it is:** {details['what']}")
            st.markdown(f"**Why it's a problem:** {details['problem']}")
            st.markdown(f"**How to avoid it / What to do:** {details['howto']}")
            st.markdown(f"**Analogy / Example:** {details['analogy']}")
    st.markdown("---")
    st.info("Coming in future cycles: Sample Size Calculator for Continuous Outcomes.")

def show_analyze_results_page():
    st.header("Analyze Your A/B Test Results 📊")
    st.markdown("Upload your A/B test data (as a CSV file) to perform an analysis for **binary outcomes**.")
    st.markdown("---")

    # Initialize session state variables
    if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
    if 'df_analysis' not in st.session_state: st.session_state.df_analysis = None # Renamed to avoid conflict
    if 'variation_col_analysis' not in st.session_state: st.session_state.variation_col_analysis = None
    if 'outcome_col_analysis' not in st.session_state: st.session_state.outcome_col_analysis = None
    if 'success_value_analysis' not in st.session_state: st.session_state.success_value_analysis = None
    if 'control_group_name_analysis' not in st.session_state: st.session_state.control_group_name_analysis = None
    if 'alpha_for_analysis' not in st.session_state: st.session_state.alpha_for_analysis = 0.05
    if 'freq_summary_stats' not in st.session_state: st.session_state.freq_summary_stats = None
    if 'bayesian_results' not in st.session_state: st.session_state.bayesian_results = None


    uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"], key="file_uploader_cycle5")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df_analysis = df 
            st.success("File Uploaded Successfully!")
            st.markdown("**Data Preview (first 5 rows):**")
            st.dataframe(df.head())
            st.markdown("---")

            st.subheader("1. Map Your Data Columns")
            columns = df.columns.tolist()
            
            col1_map, col2_map = st.columns(2)
            with col1_map:
                st.session_state.variation_col_analysis = st.selectbox("Select 'Variation ID' column:", options=columns, index=0, key="var_col_c5")
            with col2_map:
                st.session_state.outcome_col_analysis = st.selectbox("Select 'Outcome' column (Binary):", options=columns, index=len(columns)-1 if len(columns)>1 else 0, key="out_col_c5")

            success_value_options = []
            if st.session_state.outcome_col_analysis:
                unique_outcomes = df[st.session_state.outcome_col_analysis].unique()
                if len(unique_outcomes) == 1: st.warning(f"Outcome column '{st.session_state.outcome_col_analysis}' has only one value: `{unique_outcomes[0]}`.")
                elif len(unique_outcomes) > 2: st.warning(f"Outcome column '{st.session_state.outcome_col_analysis}' has >2 unique values: `{unique_outcomes}`. Select the success value.")
                success_value_options = unique_outcomes
                
                if len(success_value_options) > 0:
                    success_value_str = st.selectbox(f"Which value in '{st.session_state.outcome_col_analysis}' is 'Conversion' (Success)?", options=[str(val) for val in success_value_options], index=0, key="succ_val_c5")
                    original_dtype = df[st.session_state.outcome_col_analysis].dtype
                    if success_value_str.lower() == 'nan' and any(pd.isna(val) for val in success_value_options): st.session_state.success_value_analysis = np.nan
                    elif pd.api.types.is_numeric_dtype(original_dtype) and not pd.api.types.is_bool_dtype(original_dtype):
                        try: st.session_state.success_value_analysis = original_dtype.type(success_value_str)
                        except ValueError: st.session_state.success_value_analysis = success_value_str 
                    elif pd.api.types.is_bool_dtype(original_dtype): st.session_state.success_value_analysis = (success_value_str.lower() == 'true') 
                    else: st.session_state.success_value_analysis = success_value_str
                else: st.warning(f"Could not determine distinct values in outcome column '{st.session_state.outcome_col_analysis}'.")
            
            st.markdown("---"); st.subheader("2. Select Your Control Group & Analysis Alpha")
            if st.session_state.variation_col_analysis and st.session_state.df_analysis is not None:
                variation_names = st.session_state.df_analysis[st.session_state.variation_col_analysis].unique().tolist()
                if variation_names:
                    st.session_state.control_group_name_analysis = st.selectbox("Select 'Control Group':", options=variation_names, index=0, key="ctrl_grp_c5")
                else: st.warning(f"No unique variations in '{st.session_state.variation_col_analysis}'.")
            
            st.session_state.alpha_for_analysis = st.slider("Significance Level (\u03B1) for Analysis (%)", 1, 10, 5, 1, key="alpha_analysis_c5_slider") / 100.0 # Changed key slightly
            
            if st.button("🚀 Run Analysis (Frequentist & Bayesian)", key="run_analysis_button_cycle5"):
                st.session_state.analysis_done = False 
                st.session_state.freq_summary_stats = None
                st.session_state.bayesian_results = None

                if not st.session_state.variation_col_analysis or \
                   not st.session_state.outcome_col_analysis or \
                   st.session_state.control_group_name_analysis is None or \
                   st.session_state.success_value_analysis is None:
                    st.error("Please complete all column mapping, success value identification, and control group selections.")
                else:
                    try:
                        current_df = st.session_state.df_analysis.copy()
                        if pd.isna(st.session_state.success_value_analysis):
                            current_df['__converted_binary__'] = current_df[st.session_state.outcome_col_analysis].isna().astype(int)
                        else:
                            current_df['__converted_binary__'] = (current_df[st.session_state.outcome_col_analysis] == st.session_state.success_value_analysis).astype(int)

                        summary_stats = current_df.groupby(st.session_state.variation_col_analysis).agg(
                            Users=('__converted_binary__', 'count'),
                            Conversions=('__converted_binary__', 'sum')
                        ).reset_index()
                        summary_stats.rename(columns={st.session_state.variation_col_analysis: 'Variation'}, inplace=True)

                        if summary_stats['Users'].sum() == 0: st.error("No users found after grouping.")
                        else:
                            summary_stats['Conversion Rate (%)'] = (summary_stats['Conversions'] / summary_stats['Users'].replace(0, np.nan) * 100).round(2)
                            st.session_state.freq_summary_stats = summary_stats
                            
                            bayesian_results, bayesian_error = run_bayesian_binary_analysis(
                                st.session_state.freq_summary_stats, 
                                st.session_state.control_group_name_analysis,
                                ci_level=(1-st.session_state.alpha_for_analysis)
                            )
                            if bayesian_error: st.error(f"Bayesian Analysis Error: {bayesian_error}")
                            else: st.session_state.bayesian_results = bayesian_results
                            st.session_state.analysis_done = True
                    except Exception as e:
                        st.error(f"An error occurred during data processing or initial analysis setup: {e}"); st.exception(e)
        except Exception as e:
            st.error(f"Error reading or processing CSV file: {e}"); st.exception(e)
    else:
        st.info("Upload a CSV file to begin analysis.")

    if st.session_state.analysis_done:
        alpha_display = st.session_state.alpha_for_analysis
        
        st.markdown("---"); st.subheader("Frequentist Analysis Results")
        if st.session_state.freq_summary_stats is not None:
            summary_stats_display = st.session_state.freq_summary_stats.copy()
            st.markdown("##### 📊 Descriptive Statistics"); st.dataframe(summary_stats_display.fillna('N/A (0 Users)'))
            chart_data = summary_stats_display.set_index('Variation')['Conversion Rate (%)'].fillna(0)
            if not chart_data.empty: st.bar_chart(chart_data)
            
            st.markdown(f"##### 📈 Comparison vs. Control ('{st.session_state.control_group_name_analysis}')")
            control_data_rows = summary_stats_display[summary_stats_display['Variation'] == st.session_state.control_group_name_analysis]
            if control_data_rows.empty: st.error(f"Control group '{st.session_state.control_group_name_analysis}' data missing.")
            else:
                control_data = control_data_rows.iloc[0]
                control_users, control_conversions = control_data['Users'], control_data['Conversions']
                control_cr = control_conversions / control_users if control_users > 0 else 0
                comparison_results_freq = []
                for index, row in summary_stats_display.iterrows():
                    var_name, var_users, var_conversions = row['Variation'], row['Users'], row['Conversions']
                    if var_name == st.session_state.control_group_name_analysis: continue
                    var_cr = var_conversions / var_users if var_users > 0 else 0
                    p_val_disp, ci_disp, sig_disp, abs_disp, rel_disp = 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
                    if control_users > 0 and var_users > 0:
                        abs_uplift = var_cr - control_cr; abs_disp = f"{abs_uplift*100:.2f}"
                        rel_disp = f"{(abs_uplift / control_cr) * 100:.2f}%" if control_cr > 0 else "N/A (Control CR is 0)"
                        count, nobs = np.array([var_conversions, control_conversions]), np.array([var_users, control_users])
                        if not (np.any(count < 0) or np.any(nobs <= 0) or np.any(count > nobs)):
                            try:
                                _, p_value = proportions_ztest(count, nobs, alternative='two-sided')
                                p_val_disp = f"{p_value:.4f}"
                                sig_bool = p_value < alpha_display
                                sig_disp = f"Yes (p={p_value:.4f})" if sig_bool else f"No (p={p_value:.4f})"
                                ci_low, ci_high = confint_proportions_2indep(var_conversions, var_users, control_conversions, control_users, method='wald', alpha=alpha_display)
                                ci_disp = f"[{ci_low*100:.2f}, {ci_high*100:.2f}]"
                            except Exception: p_val_disp, ci_disp, sig_disp = 'Error', 'Error', 'Error'
                        else: sig_disp = 'N/A (Invalid counts/nobs)'
                    else: sig_disp = 'N/A (Zero users)'
                    comparison_results_freq.append({"Variation": var_name, "Conversion Rate (%)": f"{var_cr*100:.2f}", "Absolute Uplift (%)": abs_disp, "Relative Uplift (%)": rel_disp, "P-value (vs Control)": p_val_disp, f"CI {100*(1-alpha_display):.0f}% for Diff. (%)": ci_disp, "Statistically Significant?": sig_disp})
                if comparison_results_freq:
                    comparison_df_freq = pd.DataFrame(comparison_results_freq)
                    st.dataframe(comparison_df_freq)
                    for _, row_data in comparison_df_freq.iterrows():
                        if "Yes" in str(row_data["Statistically Significant?"]): st.caption(f"Frequentist: Difference between **{row_data['Variation']}** and control is statistically significant at {alpha_display*100:.0f}% level (P-value: {row_data['P-value (vs Control)']}).")
                        elif "No" in str(row_data["Statistically Significant?"]): st.caption(f"Frequentist: Difference between **{row_data['Variation']}** and control is not statistically significant at {alpha_display*100:.0f}% level (P-value: {row_data['P-value (vs Control)']}).")
        else: st.info("Frequentist analysis could not be completed. Check data and selections.")

        st.markdown("---"); st.subheader("Bayesian Analysis Results")
        if st.session_state.bayesian_results:
            st.markdown(f"Using a Beta(1,1) uninformative prior. Credible Intervals (CrI) at {100*(1-alpha_display):.0f}% level.")
            bayesian_data_to_display = []
            # Ensure control_group_name is from session state for consistency
            control_group_name_for_bayesian = st.session_state.control_group_name_analysis 

            for var_name, b_res in st.session_state.bayesian_results.items():
                prob_better_html = f"<span title=\"Probability that this variation's true conversion rate is higher than the control's. Also consider the Credible Interval for Uplift to understand magnitude and uncertainty.\">{b_res['prob_better_than_control']*100:.2f}%</span>" if b_res.get('prob_better_than_control') is not None else "N/A (Control)"
                cri_uplift_html = f"<span title=\"The range where the true uplift over control likely lies. If this interval includes 0, 'no difference' or a negative effect are plausible.\">[{b_res.get('uplift_ci_low', 0)*100:.2f}, {b_res.get('uplift_ci_high', 0)*100:.2f}]</span>" if b_res.get('uplift_ci_low') is not None else "N/A (Control)"

                bayesian_data_to_display.append({
                    "Variation": var_name,
                    "Posterior Mean CR (%)": f"{b_res['mean_cr']*100:.2f}",
                    f"{100*(1-alpha_display):.0f}% CrI for CR (%)": f"[{b_res['cr_ci_low']*100:.2f}, {b_res['cr_ci_high']*100:.2f}]",
                    "P(Better > Control) (%)": prob_better_html, # Using HTML for tooltip here for table simplicity
                    "Expected Uplift (abs %)": f"{b_res.get('expected_uplift_abs', 0)*100:.2f}" if b_res.get('expected_uplift_abs') is not None else "N/A (Control)",
                    f"{100*(1-alpha_display):.0f}% CrI for Uplift (abs %)": cri_uplift_html,
                    "P(Being Best) (%)": f"{b_res['prob_best']*100:.2f}"
                })
            bayesian_df = pd.DataFrame(bayesian_data_to_display)
            st.markdown(bayesian_df.to_html(escape=False), unsafe_allow_html=True) # Render HTML for tooltips in table
            
            # Plot posterior distributions for CRs
            st.markdown("##### Posterior Distributions for Conversion Rates")
            fig_cr, ax_cr = plt.subplots()
            x_cr = np.linspace(0, 1, 500)
            for var_name, b_res in st.session_state.bayesian_results.items():
                posterior = beta_dist.pdf(x_cr, b_res['alpha_post'], b_res['beta_post'])
                ax_cr.plot(x_cr, posterior, label=f"{var_name} (Post. α={b_res['alpha_post']:.1f}, β={b_res['beta_post']:.1f})")
                ax_cr.fill_between(x_cr, posterior, alpha=0.2)
            ax_cr.set_title("Posterior Distributions of Conversion Rates")
            ax_cr.set_xlabel("Conversion Rate"); ax_cr.set_ylabel("Density"); ax_cr.legend()
            st.pyplot(fig_cr); plt.close(fig_cr)

            # Plot posterior distribution of difference for each variation vs control
            st.markdown("##### Posterior Distribution of Uplift (Variation CR - Control CR)")
            control_cr_mean_post = st.session_state.bayesian_results[control_group_name_for_bayesian]['mean_cr']
            
            num_vars_to_plot = len(st.session_state.bayesian_results) -1 # Exclude control
            if num_vars_to_plot > 0:
                cols_diff_plots = st.columns(min(num_vars_to_plot, 3)) # Max 3 plots per row
                col_idx = 0
                for var_name, b_res in st.session_state.bayesian_results.items():
                    if var_name == control_group_name_for_bayesian: continue
                    if b_res['diff_samples_vs_control'] is not None:
                        with cols_diff_plots[col_idx % min(num_vars_to_plot, 3)]:
                            fig_diff, ax_diff = plt.subplots()
                            ax_diff.hist(b_res['diff_samples_vs_control'], bins=50, density=True, alpha=0.6, label=f"{var_name} - {control_group_name_for_bayesian}")
                            ax_diff.axvline(0, color='grey', linestyle='--')
                            ax_diff.axvline(b_res['expected_uplift_abs'], color='red', linestyle=':', label=f"Mean Diff: {b_res['expected_uplift_abs']*100:.2f}%")
                            ax_diff.set_title(f"Uplift: {var_name} vs {control_group_name_for_bayesian}")
                            ax_diff.set_xlabel("Difference in Conversion Rate"); ax_diff.set_ylabel("Density")
                            ax_diff.legend()
                            st.pyplot(fig_diff); plt.close(fig_diff)
                            col_idx +=1
            
            st.markdown("""
            **Interpreting Bayesian Results (Briefly):**
            - **Posterior Mean CR:** The average conversion rate after observing the data, using the Beta(1,1) prior.
            - **CrI for CR:** We are X% confident that the true conversion rate for this variation lies within this interval.
            - **P(Better > Control):** The probability that this variation's true conversion rate is higher than the control's. _(Tooltip: Also consider the CrI for Uplift for magnitude & uncertainty)._
            - **Expected Uplift:** The average improvement (or decline) you can expect compared to the control.
            - **CrI for Uplift:** We are X% confident that the true uplift over control lies within this interval. _(Tooltip: If this interval includes 0, 'no difference' or negative effect are plausible)._
            - **P(Being Best):** The probability that this variation has the highest true conversion rate among all tested variations.
            (More detailed guidance in the 'Bayesian Analysis Guidelines' section - coming soon!)
            """)
        else: st.info("Bayesian analysis could not be completed. Check data and selections.")
    
    st.markdown("---")
    st.info("Segmentation analysis and support for continuous outcomes coming in future cycles!")

def show_interpret_results_page():
    # ... (Placeholder from Cycle 1)
    st.header("Interpreting Results & Detailed Decision Guidance 🧐")
    st.write("Guidance on how to interpret your A/B test results and make decisions will be implemented in Cycle 9.")
    st.info("Coming soon: Understanding statistical vs. practical significance, next steps!")

def show_faq_page():
    # ... (Content from Cycle 3)
    st.header("FAQ on Common Misinterpretations ❓")
    st.markdown("This section addresses some common questions and misinterpretations that arise when looking at A/B test results.")
    faqs = {
        "Q: My p-value is 0.06 (or just above 0.05). Does this mean my variation *almost* worked or is 'on the verge' of significance?": {
            "answer": "Not exactly. In traditional hypothesis testing, a result is either statistically significant (if p < α, your chosen significance level) or it isn't. A p-value of 0.06 means that *if there were truly no difference between your variations (the null hypothesis is true)*, you'd see data as extreme as yours, or more extreme, about 6% of the time due to random chance alone. It doesn't imply 'almost significant' in a way that suggests a slightly larger sample would guarantee significance. It simply means you didn't meet your pre-defined threshold for rejecting the null hypothesis.",
            "example": "Think of it like a high jump. If the bar is set at 2 meters (your significance level), and you jump 1.98 meters, you didn't clear the bar. You were close, but you didn't clear it. The p-value reflects the evidence against the null hypothesis; a p-value of 0.06 is weaker evidence than a p-value of 0.01."
        },
        "Q: If a test isn't statistically significant, does it mean there's no difference between my variations?": {
            "answer": "No, not necessarily. A non-significant result means your test did not find *sufficient evidence* to conclude that a difference exists (at your chosen significance level and with your current sample size/power). A real difference might still exist, but it could be: \n1. Smaller than the Minimum Detectable Effect (MDE) your test was powered to find. \n2. Your test might have had low statistical power, making it hard to detect a true difference even if it was reasonably large. \n3. There truly is no meaningful difference, or the difference is negligible.",
            "example": "Imagine looking for a specific type of small fish in a large, murky pond with a small net. If you don't catch any, it doesn't mean the fish aren't there. Your net (test power/sample size) might have been too small, or the fish too few or too hard to see (small effect size)."
        },
        "Q: My A/B test showed Variation B was significantly better. Why did my overall conversion rate drop after I implemented it to 100% of users?": {
            "answer": "This can be frustrating and can happen for several reasons: \n1. **Regression to the Mean:** The performance observed during the specific test period might have been an overestimation of the true long-term effect. \n2. **Novelty Effect or Scarcity Effect During Test:** Users might have reacted positively (or negatively) to the change simply because it was new, or if the test implied a limited-time offer. This effect might wear off post-launch. \n3. **Segmentation Issues (Simpson's Paradox):** The variation might have performed well for a large segment during the test, but if the overall traffic mix changes post-launch, or if it performed poorly for other crucial segments, the overall result could differ. \n4. **External Factors:** Were there different market conditions, campaigns, or site issues post-launch compared to the test period? \n5. **Type I Error (False Positive):** Even with a 5% significance level, there's a 1 in 20 chance that a statistically significant result is due to random chance. \n6. **Implementation Issues:** Was the winning variation implemented *exactly* as it was tested? Any small differences in code or UX could alter performance.",
            "example": "A new song might shoot up the charts initially due to hype (novelty effect), but its long-term popularity (post-launch performance) might be lower once the initial excitement fades."
        },
        "Q: Can I combine results from two separate A/B tests (e.g., run on different weeks) to get a larger sample size?": {
            "answer": "Generally, this is not recommended. A/B tests rely on the principle of comparing variations under the *same conditions* at the *same time*. If you run tests at different times, user behavior, traffic sources, seasonality, or other external factors could be different between the two periods, making a direct combination of data statistically invalid and potentially misleading.",
            "example": "Trying to combine data from a lemonade stand's sales on a hot sunny week with sales data from a cold, rainy week. The conditions are too different to fairly compare or combine the results as if they were from one single experiment."
        },
        "Q: Is a 200% lift with a small sample size (e.g., 100 users) more impressive than a 10% lift with a large sample size (e.g., 100,000 users)?": {
            "answer": "Not necessarily. While a 200% lift sounds dramatic, results from very small sample sizes are highly volatile and have wide confidence/credible intervals. This means the 'true' lift could be much lower, much higher, or even negative. A 10% lift observed with a large sample size is likely to be much more stable, reliable, and closer to the true underlying effect. Always look at the confidence/credible intervals and the statistical significance, not just the point estimate of the lift.",
            "example": "If one person buys a $100 item from a new 2-person visitor group, that's a 50% conversion rate and huge revenue per visitor for that tiny sample. If 1,000 people buy a $5 item from a 100,000 visitor group, the overall impact is much larger and the metrics are more reliable, even if the per-item value is smaller."
        },
        "Q: My Bayesian test shows P(B>A) = 92%. Does this mean there's a 92% chance I'll see this exact observed uplift if I roll it out?": {
            "answer": "No. P(B>A) = 92% means there's a 92% probability that the *true underlying parameter* of Variation B (e.g., its true long-term conversion rate) is greater than the true underlying parameter of Variation A. The actual uplift you observe in any given period (during the test or post-rollout) will still have some variability. The P(B>A) gives you confidence in the *direction* of the effect. To understand the *magnitude* of the potential uplift, you should look at the posterior distribution for the difference or the credible interval for the lift.",
            "example": "If a weather forecast says there's a 92% chance of rain, it means it's very likely to rain. It doesn't tell you exactly *how much* it will rain (the magnitude). For that, you'd look at other parts of the forecast, like '0.5 to 1 inch expected'."
        },
        "Q: What if my control group's conversion rate in the test is very different from its historical average?": {
            "answer": "This is a good flag to investigate. Possible reasons include: \n1. **Seasonality/Trends:** User behavior changes over time. \n2. **Different Traffic Mix:** The users in your test period might be different from your historical average (e.g., more mobile users, different marketing channels driving traffic). \n3. **Instrumentation Error:** Double-check your tracking and data collection for the test. \n4. **Actual Change in Baseline:** Something fundamental might have changed on your site or in the market. \nWhile the A/B test still validly compares variations *within the test period*, a significant shift in the baseline might make it harder to extrapolate the observed lift to long-term performance if the conditions causing the shift don't persist.",
            "example": "If your ice cream shop's historical daily average sales are 100 cones, but during a week-long new flavor test it's only 50 cones (perhaps due to cold weather), the *percentage lift* of a new flavor might still be calculable against that week's 50-cone baseline, but predicting future sales based on that lift needs to account for the unusual baseline."
        },
         "Q: The A/B/n test shows Variation C is best overall. Can I just assume it's also significantly better than Variation B without looking at that specific comparison?": {
            "answer": "Not always safely. While C might have the highest overall metric or highest probability of being best, the difference between C and B might be very small and not statistically significant (or the probability C>B might be low). It's good practice to look at key pairwise comparisons, especially between your top-performing variations, to understand the nuances. For example, C might be best, but B might be almost as good and much easier/cheaper to implement.",
            "example": "In a race, even if a runner finishes first, their margin over the second-place runner could be a fraction of a second (not a decisive win) or several seconds (a clear win). You'd look at the gap to understand the true performance difference."
        }
    }
    for question, details in faqs.items():
        with st.expander(question):
            st.markdown(f"**A:** {details['answer']}")
            if "example" in details: st.markdown(f"**Analogy / Example:** {details['example']}")
    st.markdown("---")
    st.info("Content for this section will be reviewed and expanded as needed.")

def show_bayesian_guidelines_page():
    st.header("Bayesian Analysis Guidelines 🧠")
    st.markdown("This section provides a guide to understanding and interpreting Bayesian A/B test results, complementing the direct outputs from the 'Analyze Results' page.")
    st.markdown("""
    The Bayesian approach to A/B testing offers a different perspective compared to traditional frequentist methods. Instead of p-values and fixed confidence intervals, it focuses on probabilities and updating beliefs.
    """)
    st.subheader("Core Concepts")
    st.markdown("""
    * **Prior (e.g., Beta(1,1) for proportions):** Represents your belief about a metric *before* seeing the current test data. A Beta(1,1) prior is 'uninformative', meaning it assumes all conversion rates are equally likely initially.
    * **Likelihood:** How well the observed data from your test supports different values of the metric.
    * **Posterior:** Your updated belief about the metric *after* combining the prior with the observed data. This is what the Bayesian analysis primarily works with. For binary outcomes, if you start with a Beta prior, your posterior will also be a Beta distribution.
    """)
    st.subheader("Interpreting Key Bayesian Outputs")
    st.markdown("""
    * **Posterior Distribution Plot:** Visualizes the range of plausible values for a metric (e.g., conversion rate) after seeing the data. Wider distributions mean more uncertainty.
    * **Posterior Mean CR & Credible Interval (CrI):** The CrI gives a range where the true CR likely lies (e.g., 95% probability). Unlike frequentist CIs, you can say there's an X% probability the true value is in the interval.
    * **P(Variation > Control):** The probability that the variation's true underlying metric is strictly greater than the control's. A high value (e.g., >95%) gives strong confidence the variation is better.
        * *Important Note:* Even if this probability is high, check the **Credible Interval for Uplift**. If that interval is wide or includes zero, the magnitude of the improvement might be small or uncertain.
    * **P(Being Best):** In an A/B/n test, this is the probability that a specific variation has the highest true metric among all tested variations.
    * **Expected Uplift & its CrI:** The average uplift you might expect from choosing a variation over the control, and the range of plausible true uplift values. If the CrI for uplift includes 0, then 'no difference' or even a negative impact are plausible.
    """)
    st.subheader("Advantages of Bayesian A/B Testing")
    st.markdown("""
    * **Intuitive Results:** Probabilities about hypotheses (e.g., "85% chance variation B is better than A") are often easier to understand and communicate than p-values.
    * **Good for Smaller Samples (with caution):** Bayesian methods can provide useful insights even with smaller sample sizes, though results will be more influenced by the prior. Uninformative priors are generally safe.
    * **Direct Probability Statements:** Allows direct statements about the probability of a hypothesis being true, or a parameter being in a certain range.
    * **Decision-Oriented:** Metrics like expected loss/gain (a future feature) can directly feed into decision-making frameworks.
    """)
    st.markdown("---")
    st.info("This section will be further expanded with more examples and detailed explanations on choosing priors (for advanced users) and handling different types of metrics.")

def show_roadmap_page():
    # ... (Content from Cycle 1 - V0.2.2)
    st.header("Roadmap / Possible Future Features 🚀")
    st.markdown("This application has several potential features planned for future development:")
    if FUTURE_FEATURES:
        for feature, description in FUTURE_FEATURES.items(): st.markdown(f"- **{feature}:** {description}")
    else: st.write("No future features currently listed.")
    st.markdown("---")
    st.markdown("Feedback on feature prioritization is welcome.")

# --- Main App Navigation ---
st.sidebar.title("Navigation")
PAGES = {
    "Introduction to A/B Testing": show_introduction_page,
    "Designing Your A/B Test": show_design_test_page,
    "Analyze Results": show_analyze_results_page,
    "Interpreting Results & Detailed Decision Guidance": show_interpret_results_page,
    "Bayesian Analysis Guidelines": show_bayesian_guidelines_page, 
    "FAQ on Misinterpretations": show_faq_page,
    "Roadmap / Possible Future Features": show_roadmap_page
}
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page_function = PAGES[selection]
page_function()

st.sidebar.markdown("---")
st.sidebar.info("A/B Testing Guide & Analyzer | V0.5.1 (Cycle 5 - Enhanced)")
