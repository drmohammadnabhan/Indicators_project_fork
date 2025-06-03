import streamlit as st
import numpy as np
from scipy.stats import norm, ttest_ind, t as t_dist
import math
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from scipy.stats import beta as beta_dist
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
    "Support for Ratio Metrics": "Enable analysis for metrics that are ratios of two continuous variables (e.g., revenue per transaction).",
    "Multiple Comparisons Adjustment (Frequentist)": "Implement Bonferroni correction or other methods when multiple variations are compared."
}

# --- Helper Functions ---
def calculate_binary_sample_size(baseline_cr, mde_abs, power, alpha, num_variations):
    if baseline_cr <= 0 or baseline_cr >= 1: return None, "BCR must be > 0 and < 1."
    if mde_abs <= 0: return None, "MDE must be positive."
    if power <= 0 or power >= 1: return None, "Power must be > 0 and < 1."
    if alpha <= 0 or alpha >= 1: return None, "Alpha must be > 0 and < 1."
    if num_variations < 2: return None, "Num variations must be >= 2."
    p1 = baseline_cr; p2 = baseline_cr + mde_abs
    if p2 >= 1 or p2 <=0:
        if p2 >=1: return None, f"MDE results in target CR >= 100%."
        if p2 <=0: return None, f"MDE results in target CR <= 0%."
    z_alpha_half = norm.ppf(1 - alpha / 2); z_beta = norm.ppf(power)
    var_p1 = p1 * (1 - p1); var_p2 = p2 * (1 - p2)
    num = (z_alpha_half + z_beta)**2 * (var_p1 + var_p2); den = mde_abs**2
    if den == 0: return None, "MDE cannot be zero."
    return math.ceil(num / den), None

def calculate_continuous_sample_size(baseline_mean, std_dev, mde_abs_mean, power, alpha, num_variations):
    if std_dev <= 0: return None, "Standard Deviation must be positive."
    if mde_abs_mean == 0: return None, "MDE for means cannot be zero." 
    if mde_abs_mean < 0: mde_abs_mean = abs(mde_abs_mean) 
    if power <= 0 or power >= 1: return None, "Power must be > 0 and < 1."
    if alpha <= 0 or alpha >= 1: return None, "Alpha must be > 0 and < 1."
    if num_variations < 2: return None, "Num variations must be >= 2."
    z_alpha_half = norm.ppf(1 - alpha / 2); z_beta = norm.ppf(power)
    n_per_variation = (2 * (std_dev**2) * (z_alpha_half + z_beta)**2) / (mde_abs_mean**2)
    return math.ceil(n_per_variation), None

def run_bayesian_binary_analysis(summary_stats, control_group_name, prior_alpha=1, prior_beta=1, n_samples=10000, ci_level=0.95):
    results = {}; posterior_params = {}
    if 'Variation' not in summary_stats.columns:
        original_var_col_name = summary_stats.columns[0] 
        if original_var_col_name != 'Variation': summary_stats = summary_stats.rename(columns={original_var_col_name: 'Variation'})
    for index, row in summary_stats.iterrows():
        var_name = row['Variation']; users = int(row['Users']); conversions = int(row['Conversions'])
        alpha_post = prior_alpha + conversions; beta_post = prior_beta + (users - conversions)
        posterior_params[var_name] = {'alpha': alpha_post, 'beta': beta_post}
        samples = beta_dist.rvs(alpha_post, beta_post, size=n_samples)
        results[var_name] = {'samples': samples, 'mean_cr': np.mean(samples), 'median_cr': np.median(samples), 'cr_ci_low': beta_dist.ppf((1-ci_level)/2, alpha_post, beta_post), 'cr_ci_high': beta_dist.ppf(1-(1-ci_level)/2, alpha_post, beta_post), 'alpha_post': alpha_post, 'beta_post': beta_post, 'diff_samples_vs_control': None}
    if control_group_name not in results: return None, f"Control group '{control_group_name}' not found. Available: {list(results.keys())}"
    control_samples = results[control_group_name]['samples']
    for var_name, data in results.items():
        if var_name == control_group_name: data['prob_better_than_control'] = None; data['uplift_ci_low'] = None; data['uplift_ci_high'] = None; data['expected_uplift_abs'] = None; continue
        var_samples = data['samples']; diff_samples = var_samples - control_samples
        data['diff_samples_vs_control'] = diff_samples; data['prob_better_than_control'] = np.mean(diff_samples > 0)
        data['uplift_ci_low'] = np.percentile(diff_samples, (1-ci_level)/2 * 100); data['uplift_ci_high'] = np.percentile(diff_samples, (1-(1-ci_level)/2) * 100)
        data['expected_uplift_abs'] = np.mean(diff_samples)
    all_var_names = summary_stats['Variation'].tolist()
    ordered_var_names_in_results = [name for name in all_var_names if name in results]
    if not ordered_var_names_in_results: return results, "No variations found for P(Best) calculation."
    all_samples_matrix = np.array([results[var]['samples'] for var in ordered_var_names_in_results])
    best_variation_counts = np.zeros(len(all_var_names))
    if all_samples_matrix.ndim == 2 and all_samples_matrix.shape[0] > 0 and all_samples_matrix.shape[1] == n_samples:
        for i in range(n_samples):
            current_iter_samples = all_samples_matrix[:, i]; best_idx_in_temp_matrix = np.argmax(current_iter_samples)
            best_var_name_this_iter = ordered_var_names_in_results[best_idx_in_temp_matrix]
            if best_var_name_this_iter in all_var_names:
                original_idx_for_counts = all_var_names.index(best_var_name_this_iter)
                best_variation_counts[original_idx_for_counts] += 1
        prob_best = best_variation_counts / n_samples
        for i, var_name in enumerate(all_var_names):
            if var_name in results: results[var_name]['prob_best'] = prob_best[i]
            elif var_name not in results and len(ordered_var_names_in_results) < len(all_var_names): results[var_name] = {'prob_best': 0.0}
    else: 
        for var_name in all_var_names:
            if var_name in results: results[var_name]['prob_best'] = 1.0 if len(ordered_var_names_in_results) == 1 else 0.0
            else: results[var_name] = {'prob_best': 0.0}
    return results, None

# --- Page Functions ---
def show_introduction_page():
    # ... (Content from V0.3)
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
    # ... (Content from V0.6 - Sample Size Calc with Binary/Continuous toggle and Common Pitfalls)
    st.header("Designing Your A/B Test 📐")
    st.markdown("A crucial step in designing an A/B test is determining the appropriate sample size. This calculator will help you estimate the number of users needed per variation.")
    st.markdown("---")
    metric_type_ss = st.radio("Select your primary metric type for sample size calculation:", ('Binary (e.g., Conversion Rate)', 'Continuous (e.g., Average Order Value)'), key="ss_metric_type_radio_c6")
    st.markdown("---")
    if metric_type_ss == 'Binary (e.g., Conversion Rate)':
        st.subheader("Sample Size Calculator (for Binary Outcomes)")
        st.markdown("**Calculator Inputs:**")
        cols_bin = st.columns(2)
        with cols_bin[0]: baseline_cr_percent = st.number_input(label="Baseline Conversion Rate (BCR) (%)", min_value=0.1, max_value=99.9, value=5.0, step=0.1, format="%.1f", help="Current CR of control (e.g., 5% for 5 out of 100).", key="ss_bcr_c6")
        with cols_bin[1]: mde_abs_percent = st.number_input(label="Minimum Detectable Effect (MDE) - Absolute (%)", min_value=0.1, max_value=50.0, value=1.0, step=0.1, format="%.1f", help="Smallest absolute CR increase to detect (e.g., 1% for 5% to 6%).", key="ss_mde_c6")
        cols2_bin = st.columns(2)
        with cols2_bin[0]: power_percent_bin = st.slider(label="Statistical Power (1 - \u03B2) (%)", min_value=50, max_value=99, value=80, step=1, format="%d%%", help="Probability of detecting an effect if one exists (typically 80-90%).", key="ss_power_c6")
        with cols2_bin[1]: alpha_percent_bin = st.slider(label="Significance Level (\u03B1) (%) - Two-sided", min_value=1, max_value=20, value=5, step=1, format="%d%%", help="Risk of false positive (typically 1-5%).", key="ss_alpha_c6")
        num_variations_ss_bin = st.number_input(label="Number of Variations (including Control)", min_value=2, value=2, step=1, help="Total versions (e.g., Control + 1 Var = 2).", key="ss_num_var_c6")
        st.markdown("<p style='font-size: smaller; font-style: italic;'>Note: For binary, based on pairwise comparisons vs. control at specified α.</p>", unsafe_allow_html=True)
        if st.button("Calculate Sample Size (Binary)", key="ss_calc_button_c6_bin"):
            baseline_cr, mde_abs, power, alpha = baseline_cr_percent/100.0, mde_abs_percent/100.0, power_percent_bin/100.0, alpha_percent_bin/100.0
            sample_size, error_msg = calculate_binary_sample_size(baseline_cr, mde_abs, power, alpha, num_variations_ss_bin)
            if error_msg: st.error(error_msg)
            elif sample_size:
                st.success("Calculation Successful!")
                target_cr_pct = (baseline_cr + mde_abs) * 100
                res_cols = st.columns(2); res_cols[0].metric("Required Sample Size PER Variation", f"{sample_size:,}"); res_cols[1].metric("Total Required Sample Size", f"{(sample_size * num_variations_ss_bin):,}")
                st.markdown(f"**Summary of Inputs Used:**\n- Baseline Conversion Rate (BCR): `{baseline_cr_percent:.1f}%`\n- Absolute MDE: `{mde_abs_percent:.1f}%` (Targeting a CR of at least `{target_cr_pct:.2f}%` for variations)\n- Statistical Power: `{power_percent_bin}%` (1 - \u03B2)\n- Significance Level (\u03B1): `{alpha_percent_bin}%` (two-sided)\n- Number of Variations: `{num_variations_ss_bin}`.")
                with st.expander("Show Formula Used (Binary)"):
                    st.markdown("For comparing two proportions ($n$ per group):")
                    st.latex(r'''n = \frac{(Z_{\alpha/2} + Z_{\beta})^2 \cdot (p_1(1-p_1) + p_2(1-p_2))}{(p_2 - p_1)^2}''')
                    st.markdown(r"**Where:** $n$=Sample size per variation, $p_1$=BCR, $p_2=p_1+\text{MDE}$, $Z_{\alpha/2}$=Z-score for $\alpha$, $Z_{\beta}$=Z-score for power.")
    elif metric_type_ss == 'Continuous (e.g., Average Order Value)':
        st.subheader("Sample Size Calculator (for Continuous Outcomes)")
        st.markdown("**Calculator Inputs:**")
        cols_cont = st.columns(2)
        with cols_cont[0]: baseline_mean = st.number_input(label="Baseline Mean (Control Group)", value=100.0, step=1.0, format="%.2f", help="Current average value of your metric for the control group.", key="ss_mean_c6")
        with cols_cont[1]: std_dev = st.number_input(label="Standard Deviation (of the metric)", value=20.0, min_value=0.1, step=0.1, format="%.2f", help="Estimated standard deviation of your continuous metric. Get from historical data if possible.", key="ss_stddev_c6")
        mde_abs_mean = st.number_input(label="Minimum Detectable Effect (MDE) - Absolute Mean Difference", value=5.0,min_value=0.01, step=0.1, format="%.2f", help="Smallest absolute difference in means you want to detect (e.g., $2 increase). Must be > 0.", key="ss_mde_mean_c6")
        cols2_cont = st.columns(2)
        with cols2_cont[0]: power_percent_cont = st.slider(label="Statistical Power (1 - \u03B2) (%)", min_value=50, max_value=99, value=80, step=1, format="%d%%", help="Typically 80-90%.", key="ss_power_cont_c6")
        with cols2_cont[1]: alpha_percent_cont = st.slider(label="Significance Level (\u03B1) (%) - Two-sided", min_value=1, max_value=20, value=5, step=1, format="%d%%", help="Typically 1-5%.", key="ss_alpha_cont_c6")
        num_variations_ss_cont = st.number_input(label="Number of Variations (including Control)", min_value=2, value=2, step=1, help="Total versions.", key="ss_num_var_cont_c6")
        st.markdown("<p style='font-size: smaller; font-style: italic;'>Note: For continuous, based on pairwise comparisons vs. control at specified α, assuming similar standard deviations across groups.</p>", unsafe_allow_html=True)
        if st.button("Calculate Sample Size (Continuous)", key="ss_calc_button_c6_cont"):
            power, alpha = power_percent_cont/100.0, alpha_percent_cont/100.0
            sample_size, error_msg = calculate_continuous_sample_size(baseline_mean, std_dev, mde_abs_mean, power, alpha, num_variations_ss_cont)
            if error_msg: st.error(error_msg)
            elif sample_size:
                st.success("Calculation Successful!")
                target_mean = baseline_mean + mde_abs_mean
                res_cols = st.columns(2); res_cols[0].metric("Required Sample Size PER Variation", f"{sample_size:,}"); res_cols[1].metric("Total Required Sample Size", f"{(sample_size * num_variations_ss_cont):,}")
                st.markdown(f"**Summary of Inputs Used:**\n- Baseline Mean: `{baseline_mean:.2f}`\n- Estimated Standard Deviation: `{std_dev:.2f}`\n- Absolute MDE (Mean Difference): `{mde_abs_mean:.2f}` (Targeting a mean of at least `{target_mean:.2f}` for variations)\n- Statistical Power: `{power_percent_cont}%` (1 - \u03B2)\n- Significance Level (\u03B1): `{alpha_percent_cont}%` (two-sided)\n- Number of Variations: `{num_variations_ss_cont}`.")
                with st.expander("Show Formula Used (Continuous)"):
                    st.markdown("For comparing two means ($n$ per group, assuming similar standard deviations $\sigma$):")
                    st.latex(r'''n = \frac{2 \cdot \sigma^2 \cdot (Z_{\alpha/2} + Z_{\beta})^2}{\text{MDE}^2}''')
                    st.markdown(r"**Where:** $n$=Sample size per variation, $\sigma$=Standard Deviation, MDE=Absolute difference in means, $Z_{\alpha/2}$=Z-score for $\alpha$, $Z_{\beta}$=Z-score for power.")
    st.markdown("---")
    with st.expander("💡 Understanding Input Impacts on Sample Size"):
        st.markdown(r"""Adjusting input parameters affects required sample size. Understanding these trade-offs is key:
        * **Baseline Conversion Rate (BCR) / Baseline Mean:**
        * *Binary:* Sample size largest near BCR 50%.
        * *Continuous:* Baseline mean itself doesn't directly affect the sample size formula as much as Standard Deviation and MDE do, but understanding it sets context for the MDE.
        * **Standard Deviation (Continuous Metrics Only):**
        * *Impact:* *Increasing* Standard Deviation **significantly increases** required sample size.
        * *Trade-off:* Higher variability in your metric naturally requires more data to detect a true signal from noise. Reducing underlying variability (if possible) can make tests more efficient.
        * **Minimum Detectable Effect (MDE):** (Applies to both absolute difference in CRs or Means)
        * *Impact:* *Decreasing* MDE **significantly increases** sample size.
        * *Trade-off:* Detect smaller changes at higher cost (more samples/time). A larger MDE is cheaper/faster but you risk missing smaller, yet potentially valuable, effects.
        * **Statistical Power (1 - $\beta$):**
        * *Impact:* *Increasing* power **increases** sample size.
        * *Trade-off:* Reduce risk of missing real effects (false negatives) at higher cost.
        * **Significance Level ($\alpha$):**
        * *Impact:* *Decreasing* $\alpha$ (more stringent) **increases** sample size.
        * *Trade-off:* Reduce risk of false positives at higher cost.
        * **Number of Variations:** Total sample size increases proportionally with the number of variations (as sample size per variation is calculated pairwise against control).
        Balancing these factors is key for feasible, sound tests.
        """)
    st.markdown("---")
    st.subheader("Common Pitfalls in A/B Test Design & Execution")
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
    st.info("Full support for Continuous Outcomes in Sample Size Calculator is new in this version!")

def show_analyze_results_page():
    st.header("Analyze Your A/B Test Results 📊")
    st.markdown("Upload your A/B test data (as a CSV file) to perform an analysis.")
    st.markdown("---")

    if 'analysis_done_c6' not in st.session_state: st.session_state.analysis_done_c6 = False
    if 'df_analysis_c6' not in st.session_state: st.session_state.df_analysis_c6 = None
    if 'metric_type_analysis_c6' not in st.session_state: st.session_state.metric_type_analysis_c6 = 'Binary'
    if 'variation_col_analysis' not in st.session_state: st.session_state.variation_col_analysis = None
    if 'outcome_col_analysis' not in st.session_state: st.session_state.outcome_col_analysis = None
    if 'success_value_analysis' not in st.session_state: st.session_state.success_value_analysis = None
    if 'control_group_name_analysis' not in st.session_state: st.session_state.control_group_name_analysis = None
    if 'alpha_for_analysis' not in st.session_state: st.session_state.alpha_for_analysis = 0.05
    if 'freq_summary_stats_c6' not in st.session_state: st.session_state.freq_summary_stats_c6 = None
    if 'bayesian_results_c6' not in st.session_state: st.session_state.bayesian_results_c6 = None
    if 'metric_col_name_c6' not in st.session_state: st.session_state.metric_col_name_c6 = None # Initialize this

    uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"], key="file_uploader_cycle6")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df_analysis_c6 = df 
            st.success("File Uploaded Successfully!")
            st.markdown("**Data Preview (first 5 rows):**"); st.dataframe(df.head())
            st.markdown("---")

            st.subheader("1. Map Data & Select Metric Type")
            columns = df.columns.tolist()
            map_col1, map_col2, map_col3 = st.columns(3)
            with map_col1: st.session_state.variation_col_analysis = st.selectbox("Select 'Variation ID' column:", options=columns, index=0, key="var_col_c6")
            with map_col2: st.session_state.outcome_col_analysis = st.selectbox("Select 'Outcome' column:", options=columns, index=len(columns)-1 if len(columns)>1 else 0, key="out_col_c6")
            with map_col3: st.session_state.metric_type_analysis_c6 = st.radio("Select Metric Type for Outcome Column:", ('Binary', 'Continuous'), key="metric_type_analysis_radio_c6", horizontal=True)

            if st.session_state.metric_type_analysis_c6 == 'Binary':
                success_value_options = []
                if st.session_state.outcome_col_analysis:
                    unique_outcomes = df[st.session_state.outcome_col_analysis].unique()
                    if len(unique_outcomes) == 1: st.warning(f"Outcome column '{st.session_state.outcome_col_analysis}' has only one value: `{unique_outcomes[0]}`.")
                    elif len(unique_outcomes) > 2: st.warning(f"Outcome column '{st.session_state.outcome_col_analysis}' has >2 unique values: `{unique_outcomes}`. Select the success value.")
                    success_value_options = unique_outcomes
                    if len(success_value_options) > 0:
                        success_value_str = st.selectbox(f"Which value in '{st.session_state.outcome_col_analysis}' is 'Conversion' (Success)?", options=[str(val) for val in success_value_options], index=0, key="succ_val_c6")
                        original_dtype = df[st.session_state.outcome_col_analysis].dtype
                        if success_value_str.lower() == 'nan' and any(pd.isna(val) for val in success_value_options): st.session_state.success_value_analysis = np.nan
                        elif pd.api.types.is_numeric_dtype(original_dtype) and not pd.api.types.is_bool_dtype(original_dtype):
                            try: st.session_state.success_value_analysis = original_dtype.type(success_value_str)
                            except ValueError: st.session_state.success_value_analysis = success_value_str 
                        elif pd.api.types.is_bool_dtype(original_dtype): st.session_state.success_value_analysis = (success_value_str.lower() == 'true') 
                        else: st.session_state.success_value_analysis = success_value_str
                    else: st.warning(f"Could not determine distinct values in outcome column '{st.session_state.outcome_col_analysis}'.")
            else: 
                st.session_state.success_value_analysis = None 
                if st.session_state.outcome_col_analysis and not pd.api.types.is_numeric_dtype(df[st.session_state.outcome_col_analysis]):
                    st.error(f"For 'Continuous' metric type, the outcome column '{st.session_state.outcome_col_analysis}' must be numeric. Please select a numeric column or check your data.")
            
            st.markdown("---"); st.subheader("2. Select Your Control Group & Analysis Alpha")
            if st.session_state.variation_col_analysis and st.session_state.df_analysis_c6 is not None:
                variation_names = st.session_state.df_analysis_c6[st.session_state.variation_col_analysis].unique().tolist()
                if variation_names: st.session_state.control_group_name_analysis = st.selectbox("Select 'Control Group':", options=variation_names, index=0, key="ctrl_grp_c6")
                else: st.warning(f"No unique variations in '{st.session_state.variation_col_analysis}'.")
            st.session_state.alpha_for_analysis = st.slider("Significance Level (\u03B1) for Analysis (%)", 1, 10, 5, 1, key="alpha_analysis_c6_slider") / 100.0
            
            analysis_button_label = f"🚀 Run Analysis ({st.session_state.metric_type_analysis_c6} Outcome)"
            if st.button(analysis_button_label, key="run_analysis_button_cycle6"):
                st.session_state.analysis_done_c6 = False; st.session_state.freq_summary_stats_c6 = None; st.session_state.bayesian_results_c6 = None
                valid_setup = True
                if not st.session_state.variation_col_analysis or not st.session_state.outcome_col_analysis or st.session_state.control_group_name_analysis is None:
                    st.error("Please complete all column mapping and control group selections."); valid_setup = False
                if st.session_state.metric_type_analysis_c6 == 'Binary' and st.session_state.success_value_analysis is None:
                    st.error("For Binary outcome, please specify the 'Conversion (Success)' value."); valid_setup = False
                if st.session_state.metric_type_analysis_c6 == 'Continuous' and st.session_state.outcome_col_analysis and not pd.api.types.is_numeric_dtype(st.session_state.df_analysis_c6[st.session_state.outcome_col_analysis]):
                    st.error(f"For 'Continuous' metric type, outcome column '{st.session_state.outcome_col_analysis}' must be numeric."); valid_setup = False
                
                if valid_setup:
                    try:
                        current_df = st.session_state.df_analysis_c6.copy(); var_col = st.session_state.variation_col_analysis; out_col = st.session_state.outcome_col_analysis
                        if st.session_state.metric_type_analysis_c6 == 'Binary':
                            if pd.isna(st.session_state.success_value_analysis): current_df['__outcome_processed__'] = current_df[out_col].isna().astype(int)
                            else: current_df['__outcome_processed__'] = (current_df[out_col] == st.session_state.success_value_analysis).astype(int)
                            summary_stats = current_df.groupby(var_col).agg(Users=('__outcome_processed__', 'count'), Converts=('__outcome_processed__', 'sum')).reset_index()
                            summary_stats.rename(columns={var_col: 'Variation', 'Converts': 'Conversions'}, inplace=True)
                            if summary_stats['Users'].sum() == 0: st.error("No users found after grouping."); st.stop()
                            summary_stats['Metric Value (%)'] = (summary_stats['Conversions'] / summary_stats['Users'].replace(0, np.nan) * 100).round(2)
                            st.session_state.metric_col_name_c6 = 'Metric Value (%)'
                            st.session_state.freq_summary_stats_c6 = summary_stats.copy()
                        elif st.session_state.metric_type_analysis_c6 == 'Continuous':
                            current_df['__outcome_processed__'] = pd.to_numeric(current_df[out_col], errors='coerce')
                            summary_stats = current_df.groupby(var_col).agg(Users=(out_col, 'count'), Mean_Value=(out_col, 'mean'), Std_Dev=(out_col, 'std'), Std_Err=(out_col, lambda x: x.std(ddof=1) / np.sqrt(x.count()) if x.count() > 0 else np.nan)).reset_index()
                            summary_stats.rename(columns={var_col: 'Variation'}, inplace=True)
                            if summary_stats['Users'].sum() == 0: st.error("No users found after grouping."); st.stop()
                            summary_stats['Mean_Value'] = summary_stats['Mean_Value'].round(3)
                            summary_stats['Std_Dev'] = summary_stats['Std_Dev'].round(3)
                            summary_stats['Std_Err'] = summary_stats['Std_Err'].round(3)
                            st.session_state.metric_col_name_c6 = 'Mean_Value'
                            st.session_state.freq_summary_stats_c6 = summary_stats.copy()
                        if st.session_state.metric_type_analysis_c6 == 'Binary' and st.session_state.freq_summary_stats_c6 is not None:
                            bayesian_results, bayesian_error = run_bayesian_binary_analysis(st.session_state.freq_summary_stats_c6.copy(), st.session_state.control_group_name_analysis, ci_level=(1-st.session_state.alpha_for_analysis)) # Pass a copy
                            if bayesian_error: st.error(f"Bayesian Analysis Error: {bayesian_error}")
                            else: st.session_state.bayesian_results_c6 = bayesian_results
                        st.session_state.analysis_done_c6 = True
                    except Exception as e: st.error(f"An error during data processing: {e}"); st.exception(e)
        except Exception as e: st.error(f"Error reading/processing CSV: {e}"); st.exception(e)
    else: st.info("Upload a CSV file to begin analysis.")

    if st.session_state.analysis_done_c6:
        alpha_display = st.session_state.alpha_for_analysis; metric_type_display = st.session_state.metric_type_analysis_c6
        summary_stats_display = st.session_state.freq_summary_stats_c6
        metric_col_name_display = st.session_state.metric_col_name_c6

        st.markdown("---"); st.subheader(f"Frequentist Analysis Results ({metric_type_display} Outcome)")
        if summary_stats_display is not None and metric_col_name_display is not None:
            st.markdown("##### 📊 Descriptive Statistics")
            if metric_type_display == 'Binary': st.dataframe(summary_stats_display[['Variation', 'Users', 'Conversions', metric_col_name_display]].fillna('N/A (0 Users)'))
            else: st.dataframe(summary_stats_display[['Variation', 'Users', 'Mean_Value', 'Std_Dev', 'Std_Err']].fillna('N/A'))
            
            # NEW: Warning for Zero Standard Deviation in Continuous Metrics
            if metric_type_display == 'Continuous':
                for index, row in summary_stats_display.iterrows():
                    if row['Std_Dev'] == 0 and row['Users'] > 1 : # Check for Std_Dev being 0 and more than 1 user
                        st.warning(f"⚠️ Variation '{row['Variation']}' has a Standard Deviation of 0 for the outcome '{st.session_state.outcome_col_analysis}'. This means all outcome values for this variation are identical. This can impact the interpretation of statistical tests.")
            
            chart_data = summary_stats_display.set_index('Variation')[metric_col_name_display].fillna(0)
            if not chart_data.empty: st.bar_chart(chart_data, y=metric_col_name_display)
            
            st.markdown(f"##### 📈 Comparison vs. Control ('{st.session_state.control_group_name_analysis}')")
            control_data_rows = summary_stats_display[summary_stats_display['Variation'] == st.session_state.control_group_name_analysis]
            if control_data_rows.empty: st.error(f"Control group data missing.")
            else:
                control_data = control_data_rows.iloc[0]; comparison_results_freq = []
                if metric_type_display == 'Binary':
                    control_users, control_conversions = control_data['Users'], control_data['Conversions']
                    control_metric_val = control_conversions / control_users if control_users > 0 else 0
                    for index, row in summary_stats_display.iterrows():
                        var_name, var_users, var_conversions = row['Variation'], row['Users'], row['Conversions']
                        if var_name == st.session_state.control_group_name_analysis: continue
                        var_metric_val = var_conversions / var_users if var_users > 0 else 0
                        p_val_disp, ci_disp, sig_disp, abs_disp, rel_disp = 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
                        if control_users > 0 and var_users > 0:
                            abs_uplift = var_metric_val - control_metric_val; abs_disp = f"{abs_uplift*100:.2f}"
                            rel_disp = f"{(abs_uplift / control_metric_val) * 100:.2f}%" if control_metric_val > 0 else "N/A"
                            count, nobs = np.array([var_conversions, control_conversions]), np.array([var_users, control_users])
                            if not (np.any(count < 0) or np.any(nobs <= 0) or np.any(count > nobs)):
                                try:
                                    _, p_value = proportions_ztest(count, nobs, alternative='two-sided')
                                    p_val_disp = f"{p_value:.4f}"
                                    sig_bool = p_value < alpha_display; sig_disp = f"Yes (p={p_value:.4f})" if sig_bool else f"No (p={p_value:.4f})"
                                    ci_low, ci_high = confint_proportions_2indep(var_conversions, var_users, control_conversions, control_users, method='wald', alpha=alpha_display)
                                    ci_disp = f"[{ci_low*100:.2f}, {ci_high*100:.2f}]"
                                except Exception: p_val_disp, ci_disp, sig_disp = 'Error', 'Error', 'Error'
                            else: sig_disp = 'N/A (Invalid counts/nobs)'
                        else: sig_disp = 'N/A (Zero users)'
                        comparison_results_freq.append({"Variation": var_name, "CR (%)": f"{var_metric_val*100:.2f}", "Abs. Uplift (%)": abs_disp, "Rel. Uplift (%)": rel_disp, "P-value": p_val_disp, f"CI {100*(1-alpha_display):.0f}% Diff (%)": ci_disp, "Significant?": sig_disp})
                elif metric_type_display == 'Continuous':
                    control_users, control_mean = control_data['Users'], control_data['Mean_Value']
                    control_group_data = st.session_state.df_analysis_c6[st.session_state.df_analysis_c6[st.session_state.variation_col_analysis] == st.session_state.control_group_name_analysis][st.session_state.outcome_col_analysis].dropna()
                    for index, row in summary_stats_display.iterrows():
                        var_name, var_users, var_mean = row['Variation'], row['Users'], row['Mean_Value']
                        if var_name == st.session_state.control_group_name_analysis: continue
                        p_val_disp, ci_disp, sig_disp, abs_disp, rel_disp = 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
                        if control_users > 1 and var_users > 1: # t-test needs at least 2 samples per group
                            abs_diff_means = var_mean - control_mean; abs_disp = f"{abs_diff_means:.3f}"
                            rel_disp = f"{(abs_diff_means / control_mean) * 100:.2f}%" if control_mean != 0 else "N/A"
                            var_group_data = st.session_state.df_analysis_c6[st.session_state.df_analysis_c6[st.session_state.variation_col_analysis] == var_name][st.session_state.outcome_col_analysis].dropna()
                            if len(control_group_data) < 2 or len(var_group_data) < 2: sig_disp = "N/A (Too few data points for t-test in a group)"
                            else:
                                try:
                                    t_stat, p_value = ttest_ind(var_group_data, control_group_data, equal_var=False, nan_policy='omit')
                                    p_val_disp = f"{p_value:.4f}"
                                    sig_bool = p_value < alpha_display; sig_disp = f"Yes (p={p_value:.4f})" if sig_bool else f"No (p={p_value:.4f})"
                                    N1, N2 = len(var_group_data), len(control_group_data)
                                    s1_sq, s2_sq = var_group_data.var(ddof=1), control_group_data.var(ddof=1)
                                    pooled_se_diff = math.sqrt(s1_sq/N1 + s2_sq/N2) if N1 > 0 and N2 > 0 and s1_sq >=0 and s2_sq >=0 else 0
                                    df_t = min(N1-1, N2-1) if N1 > 1 and N2 > 1 else 1 # Simplified df
                                    if df_t > 0 and pooled_se_diff > 0:
                                        t_crit = t_dist.ppf(1 - alpha_display / 2, df=df_t)
                                        ci_low_mean_diff, ci_high_mean_diff = abs_diff_means - t_crit * pooled_se_diff, abs_diff_means + t_crit * pooled_se_diff
                                        ci_disp = f"[{ci_low_mean_diff:.3f}, {ci_high_mean_diff:.3f}]"
                                    else: ci_disp = "N/A (Cannot compute CI)"
                                except Exception as e_ttest: p_val_disp, ci_disp, sig_disp = 'Error', 'Error', 'Error'; st.warning(f"T-test error for {var_name}: {e_ttest}")
                        else: sig_disp = 'N/A (Too few users for reliable test)'
                        comparison_results_freq.append({"Variation": var_name, "Mean Value": f"{var_mean:.3f}", "Abs. Diff.": abs_disp, "Rel. Diff. (%)": rel_disp, "P-value": p_val_disp, f"CI {100*(1-alpha_display):.0f}% Diff.": ci_disp, "Significant?": sig_disp})
                if comparison_results_freq:
                    comparison_df_freq = pd.DataFrame(comparison_results_freq)
                    st.dataframe(comparison_df_freq)
                    for _, row_data in comparison_df_freq.iterrows():
                        if "Yes" in str(row_data["Significant?"]): st.caption(f"Frequentist: Diff between **{row_data['Variation']}** & control is significant at {alpha_display*100:.0f}% level (P-value: {row_data['P-value']}).")
                        elif "No" in str(row_data["Significant?"]): st.caption(f"Frequentist: Diff between **{row_data['Variation']}** & control is not significant at {alpha_display*100:.0f}% level (P-value: {row_data['P-value']}).")
            if metric_type_display == 'Continuous' and st.session_state.df_analysis_c6 is not None:
                st.markdown("##### Distribution of Outcomes by Variation (Box Plots)")
                try:
                    plot_df_cont = st.session_state.df_analysis_c6.copy()
                    plot_df_cont[st.session_state.outcome_col_analysis] = pd.to_numeric(plot_df_cont[st.session_state.outcome_col_analysis], errors='coerce')
                    plot_df_cont.dropna(subset=[st.session_state.outcome_col_analysis], inplace=True)
                    if not plot_df_cont.empty:
                        unique_vars_plot = sorted(plot_df_cont[st.session_state.variation_col_analysis].unique())
                        boxplot_data = [plot_df_cont[plot_df_cont[st.session_state.variation_col_analysis] == var_name][st.session_state.outcome_col_analysis] for var_name in unique_vars_plot if not plot_df_cont[plot_df_cont[st.session_state.variation_col_analysis] == var_name][st.session_state.outcome_col_analysis].empty]
                        valid_labels = [var_name for var_name in unique_vars_plot if not plot_df_cont[plot_df_cont[st.session_state.variation_col_analysis] == var_name][st.session_state.outcome_col_analysis].empty]
                        if not boxplot_data or not valid_labels or len(boxplot_data) != len(valid_labels):
                            st.caption("Not enough data for one or more variations to display box plots after handling non-numeric/empty outcomes.")
                        else:
                            fig_box, ax_box = plt.subplots(); ax_box.boxplot(boxplot_data, labels=valid_labels)
                            ax_box.set_title("Outcome Distributions by Variation"); ax_box.set_ylabel(st.session_state.outcome_col_analysis); ax_box.set_xlabel("Variation")
                            st.pyplot(fig_box); plt.close(fig_box)
                    else: st.caption("Not enough data to display box plots after handling non-numeric outcomes.")
                except Exception as e_plot: st.warning(f"Could not generate box plots: {e_plot}")
        
        if metric_type_display == 'Binary':
            st.markdown("---"); st.subheader("Bayesian Analysis Results (Binary Outcome)")
            if st.session_state.bayesian_results_c6 and summary_stats_display is not None and metric_col_name_display is not None:
                st.markdown(f"Using a Beta(1,1) uninformative prior. Credible Intervals (CrI) at {100*(1-alpha_display):.0f}% level.")
                bayesian_data_to_display = []; control_group_name_for_bayesian = st.session_state.control_group_name_analysis 
                for var_name, b_res in st.session_state.bayesian_results_c6.items():
                    if var_name not in summary_stats_display['Variation'].values: continue
                    prob_better_html = f"<span title=\"Probability that this variation's true conversion rate is higher than the control's. Also consider the Credible Interval for Uplift to understand magnitude and uncertainty.\">{b_res.get('prob_better_than_control',0)*100:.2f}%</span>" if b_res.get('prob_better_than_control') is not None else "N/A (Control)"
                    cri_uplift_html = f"<span title=\"The range where the true uplift over control likely lies. If this interval includes 0, 'no difference' or a negative effect are plausible.\">[{b_res.get('uplift_ci_low', 0)*100:.2f}, {b_res.get('uplift_ci_high', 0)*100:.2f}]</span>" if b_res.get('uplift_ci_low') is not None else "N/A (Control)"
                    bayesian_data_to_display.append({"Variation": var_name, "Posterior Mean CR (%)": f"{b_res.get('mean_cr',0)*100:.2f}", f"{100*(1-alpha_display):.0f}% CrI for CR (%)": f"[{b_res.get('cr_ci_low',0)*100:.2f}, {b_res.get('cr_ci_high',0)*100:.2f}]", "P(Better > Control) (%)": prob_better_html, "Expected Uplift (abs %)": f"{b_res.get('expected_uplift_abs', 0)*100:.2f}" if b_res.get('expected_uplift_abs') is not None else "N/A (Control)", f"{100*(1-alpha_display):.0f}% CrI for Uplift (abs %)": cri_uplift_html, "P(Being Best) (%)": f"{b_res.get('prob_best',0)*100:.2f}"})
                bayesian_df = pd.DataFrame(bayesian_data_to_display); st.markdown(bayesian_df.to_html(escape=False), unsafe_allow_html=True)
                
                st.markdown("##### Posterior Distributions for Conversion Rates"); fig_cr, ax_cr = plt.subplots()
                
                # --- START OF CORRECTED SECTION for Bayesian CR Posterior Plot ---
                observed_max_cr_for_plot = 0.0
                numeric_crs = pd.to_numeric(summary_stats_display[metric_col_name_display], errors='coerce')
                if not numeric_crs.empty and numeric_crs.notna().any():
                    observed_max_cr_for_plot = numeric_crs.max() / 100.0
                else:
                    observed_max_cr_for_plot = 0.1 

                posterior_max_cr_for_plot = 0.0
                if st.session_state.bayesian_results_c6:
                    all_posterior_highs = []
                    for var_name_iter, b_res_iter in st.session_state.bayesian_results_c6.items():
                        if var_name_iter in summary_stats_display['Variation'].tolist() and b_res_iter.get('cr_ci_high') is not None:
                            all_posterior_highs.append(b_res_iter.get('cr_ci_high'))
                    if all_posterior_highs:
                        posterior_max_cr_for_plot = max(all_posterior_highs)

                final_x_limit_candidate = max(observed_max_cr_for_plot, posterior_max_cr_for_plot)
                x_cr_plot_limit = min(1.0, final_x_limit_candidate + 0.05) 
                
                if x_cr_plot_limit <= 0.01: 
                    x_cr_plot_limit = 0.1 

                x_cr_range = np.linspace(0, x_cr_plot_limit, 300)
                
                max_density = 0
                for var_name, b_res in st.session_state.bayesian_results_c6.items():
                    if var_name not in summary_stats_display['Variation'].values: continue
                    alpha_p = b_res.get('alpha_post', 1)
                    beta_p = b_res.get('beta_post', 1)
                    if alpha_p > 0 and beta_p > 0: 
                        posterior = beta_dist.pdf(x_cr_range, alpha_p, beta_p)
                        ax_cr.plot(x_cr_range, posterior, label=f"{var_name} (Post. α={alpha_p:.1f}, β={beta_p:.1f})")
                        ax_cr.fill_between(x_cr_range, posterior, alpha=0.2)
                        if posterior is not None and np.any(np.isfinite(posterior)): 
                            finite_posterior = posterior[np.isfinite(posterior)]
                            if finite_posterior.size > 0:
                                 max_density = max(max_density, np.nanmax(finite_posterior))
                    else:
                        st.caption(f"Skipping plot for {var_name} due to invalid posterior parameters (α={alpha_p}, β={beta_p})")

                if max_density > 0: 
                    ax_cr.set_ylim(0, max_density * 1.1)
                else: 
                    ax_cr.set_ylim(0,1)
                # --- END OF CORRECTED SECTION for Bayesian CR Posterior Plot ---
                
                ax_cr.set_title("Posterior Distributions of CRs"); ax_cr.set_xlabel("Conversion Rate"); ax_cr.set_ylabel("Density"); ax_cr.legend(); st.pyplot(fig_cr); plt.close(fig_cr)
                
                st.markdown("##### Posterior Distribution of Uplift (Variation CR - Control CR)")
                num_vars_to_plot = sum(1 for var_name_uplift in st.session_state.bayesian_results_c6 if var_name_uplift != control_group_name_for_bayesian and st.session_state.bayesian_results_c6[var_name_uplift].get('diff_samples_vs_control') is not None and var_name_uplift in summary_stats_display['Variation'].values)
                if num_vars_to_plot > 0:
                    cols_diff_plots = st.columns(min(num_vars_to_plot, 3)); col_idx = 0
                    for var_name, b_res in st.session_state.bayesian_results_c6.items():
                        if var_name == control_group_name_for_bayesian or b_res.get('diff_samples_vs_control') is None or var_name not in summary_stats_display['Variation'].values: continue
                        with cols_diff_plots[col_idx % min(num_vars_to_plot, 3)]:
                            fig_diff, ax_diff = plt.subplots(); ax_diff.hist(b_res['diff_samples_vs_control'], bins=50, density=True, alpha=0.6, label=f"{var_name} - {control_group_name_for_bayesian}")
                            ax_diff.axvline(0, color='grey', linestyle='--'); ax_diff.axvline(b_res.get('expected_uplift_abs',0), color='red', linestyle=':', label=f"Mean Diff: {b_res.get('expected_uplift_abs',0)*100:.2f}%")
                            ax_diff.set_title(f"Uplift: {var_name} vs {control_group_name_for_bayesian}"); ax_diff.set_xlabel("Difference in CR"); ax_diff.set_ylabel("Density"); ax_diff.legend(); st.pyplot(fig_diff); plt.close(fig_diff)
                            col_idx +=1
                st.markdown("""
                **Interpreting Bayesian Results (Briefly):**
                - **Posterior Mean CR:** The average conversion rate after observing the data, using the Beta(1,1) prior.
                - **CrI for CR:** We are X% confident that the true conversion rate for this variation lies within this interval.
                - **P(Better > Control):** The probability that this variation's true conversion rate is higher than the control's. _(Tooltip: Also consider the CrI for Uplift for magnitude & uncertainty)._
                - **Expected Uplift:** The average improvement (or decline) you can expect compared to the control.
                - **CrI for Uplift:** We are X% confident that the true uplift over control lies within this interval. _(Tooltip: If this interval includes 0, 'no difference' or negative effect are plausible)._
                - **P(Being Best):** The probability that this variation has the highest true conversion rate among all tested variations.
                (More detailed guidance in the 'Bayesian Analysis Guidelines' section.)
                """) 
            else: st.info("Bayesian analysis results are not available or could not be computed for this dataset (ensure binary metric and valid data).")
        elif metric_type_display == 'Continuous' and st.session_state.freq_summary_stats_c6 is not None:
            st.info("Bayesian analysis for continuous outcomes will be added in Cycle 7.")
    st.markdown("---")
    st.info("Segmentation analysis coming in a future cycle!")

def show_interpret_results_page():
    st.header("Interpreting Results & Detailed Decision Guidance 🧐")
    st.write("Guidance on how to interpret your A/B test results and make decisions will be implemented in Cycle 9.")
    st.info("Coming soon: Understanding statistical vs. practical significance, next steps!")

def show_faq_page():
    # ... (Content from Cycle 3)
    st.header("FAQ on Common Misinterpretations ❓")
    st.markdown("This section addresses some common questions and misinterpretations that arise when looking at A/B test results.")
    faqs = {
        "Q: My p-value is 0.06...": {"answer": "Not exactly. ...", "example": "Think of it like a high jump..."},
        "Q: If a test isn't statistically significant...": {"answer": "No, not necessarily. ...", "example": "Imagine looking for a small fish..."},
        "Q: My A/B test showed Variation B was significantly better...": {"answer": "This can be frustrating... \n1. **Regression to the Mean** ... \n2. **Novelty Effect** ... \n3. **Segmentation Issues** ... \n4. **External Factors** ... \n5. **Type I Error** ... \n6. **Implementation Issues** ...", "example": "A new song might shoot up the charts..."},
        "Q: Can I combine results from two separate A/B tests...": {"answer": "Generally, this is not recommended. ...", "example": "Trying to combine lemonade sales data..."},
        "Q: Is a 200% lift with a small sample size...": {"answer": "Not necessarily. ...", "example": "If one person buys a $100 item..."},
        "Q: My Bayesian test shows P(B>A) = 92%...": {"answer": "No. P(B>A) = 92% means there's a 92% probability that the *true underlying parameter*...", "example": "If a weather forecast says there's a 92% chance of rain..."},
        "Q: What if my control group's conversion rate...": {"answer": "This is a good flag to investigate. ... \n1. **Seasonality/Trends** ... \n2. **Different Traffic Mix** ... \n3. **Instrumentation Error** ... \n4. **Actual Change in Baseline** ...", "example": "If your ice cream shop's historical average sales..."},
        "Q: The A/B/n test shows Variation C is best overall...": {"answer": "Not always safely. ...", "example": "In a race, even if a runner finishes first..."}
    }
    for question, details in faqs.items():
        with st.expander(question):
            st.markdown(f"**A:** {details['answer']}")
            if "example" in details: st.markdown(f"**Analogy / Example:** {details['example']}")
    st.markdown("---")
    st.info("Content for this section will be reviewed and expanded as needed.")

def show_bayesian_guidelines_page():
    # ... (Content from Cycle 5 - V0.5.1)
    st.header("Bayesian Analysis Guidelines 🧠")
    st.markdown("This section provides a guide to understanding and interpreting Bayesian A/B test results, complementing the direct outputs from the 'Analyze Results' page.")
    st.markdown("""**Key Concepts to be Covered:** - Priors: What are they? How does the choice of prior (e.g., uninformative Beta(1,1)) affect results? - Likelihood: How your data informs the model. - Posterior: Your updated beliefs after seeing the data. For binary outcomes, if you start with a Beta prior, your posterior will also be a Beta distribution. - Interpreting Probabilities: Deep dive into P(Variation > Control) and P(Variation is Best). - Credible Intervals vs. Confidence Intervals. - Advantages of Bayesian A/B Testing (e.g., intuitive results, good for smaller samples, ability to make probability statements about hypotheses). - Expected Loss (Decision Making).""")
    st.markdown("""**Interpreting Key Bayesian Outputs (Binary Outcomes):**
    * **Posterior Distribution Plot:** Visualizes the range of plausible values for a metric (e.g., conversion rate) after seeing the data. Wider distributions mean more uncertainty.
    * **Posterior Mean CR & Credible Interval (CrI):** The CrI gives a range where the true CR likely lies (e.g., 95% probability). Unlike frequentist CIs, you can say there's an X% probability the true value is in the interval.
    * **P(Variation > Control):** The probability that the variation's true underlying metric is strictly greater than the control's. A high value (e.g., >95%) gives strong confidence the variation is better. _Important Note:_ Even if this probability is high, check the **Credible Interval for Uplift**. If that interval is wide or includes zero, the magnitude of the improvement might be small or uncertain.
    * **P(Being Best):** In an A/B/n test, this is the probability that a specific variation has the highest true metric among all tested variations.
    * **Expected Uplift & its CrI:** The average uplift you might expect from choosing a variation over the control, and the range of plausible true uplift values. If the CrI for uplift includes 0, then 'no difference' or even a negative impact are plausible.
    """)
    st.info("This section will be further expanded with more examples and detailed explanations on choosing priors (for advanced users) and handling different types of metrics, including continuous outcomes.")


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
st.sidebar.info("A/B Testing Guide & Analyzer | V0.6.1 (Cycle 6 - Enhanced)")
