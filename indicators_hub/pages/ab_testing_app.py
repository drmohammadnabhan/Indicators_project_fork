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
    # Calculates sample size for binary outcomes.
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
    # Calculates sample size for continuous outcomes.
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
    # Performs Bayesian analysis for binary outcomes using a Beta-Binomial model.
    results = {}; posterior_params = {}
    if 'Variation' not in summary_stats.columns:
        original_var_col_name = summary_stats.columns[0] 
        if original_var_col_name != 'Variation': summary_stats = summary_stats.rename(columns={original_var_col_name: 'Variation'})
    
    for index, row in summary_stats.iterrows():
        var_name = row['Variation']; users = int(row['Users']); conversions = int(row['Conversions'])
        # Calculate posterior parameters (Beta distribution)
        alpha_post = prior_alpha + conversions; beta_post = prior_beta + (users - conversions)
        posterior_params[var_name] = {'alpha': alpha_post, 'beta': beta_post}
        # Generate samples from the posterior distribution
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
    
    if control_group_name not in results: return None, f"Control group '{control_group_name}' not found. Available: {list(results.keys())}"
    control_samples = results[control_group_name]['samples']
    
    # Calculate comparisons vs. control
    for var_name, data in results.items():
        if var_name == control_group_name: 
            data['prob_better_than_control'] = None
            data['uplift_ci_low'] = None
            data['uplift_ci_high'] = None
            data['expected_uplift_abs'] = None
            continue
        var_samples = data['samples']; diff_samples = var_samples - control_samples
        data['diff_samples_vs_control'] = diff_samples
        data['prob_better_than_control'] = np.mean(diff_samples > 0)
        data['uplift_ci_low'] = np.percentile(diff_samples, (1-ci_level)/2 * 100)
        data['uplift_ci_high'] = np.percentile(diff_samples, (1-(1-ci_level)/2) * 100)
        data['expected_uplift_abs'] = np.mean(diff_samples)
        
    all_var_names = summary_stats['Variation'].tolist()
    ordered_var_names_in_results = [name for name in all_var_names if name in results]
    if not ordered_var_names_in_results: return results, "No variations found for P(Best) calculation."
    
    # Calculate P(Being Best)
    all_samples_matrix = np.array([results[var]['samples'] for var in ordered_var_names_in_results if var in results]) # Ensure var exists
    best_variation_counts = np.zeros(len(all_var_names)) # Use original all_var_names for correct indexing

    if all_samples_matrix.ndim == 2 and all_samples_matrix.shape[0] > 0 and all_samples_matrix.shape[1] == n_samples:
        best_indices = np.argmax(all_samples_matrix, axis=0)
        for i, best_idx_in_temp_matrix in enumerate(best_indices):
            best_var_name_this_iter = ordered_var_names_in_results[best_idx_in_temp_matrix]
            if best_var_name_this_iter in all_var_names:
                original_idx_for_counts = all_var_names.index(best_var_name_this_iter)
                best_variation_counts[original_idx_for_counts] += 1
        
        prob_best = best_variation_counts / n_samples
        for i, var_name in enumerate(all_var_names):
            if var_name in results: 
                results[var_name]['prob_best'] = prob_best[i]
            # If a variation was in summary_stats but somehow not in results (e.g., 0 users), assign 0 prob_best
            elif var_name not in results : 
                 results[var_name] = {'prob_best': 0.0} # Ensure entry exists if it was in summary_stats

    else: # Handle cases with only one variation or issues with sample matrix
        for var_name in all_var_names:
            if var_name in results: 
                results[var_name]['prob_best'] = 1.0 if len(ordered_var_names_in_results) == 1 and var_name == ordered_var_names_in_results[0] else 0.0
            else: 
                results[var_name] = {'prob_best': 0.0}
                
    return results, None

def run_bayesian_continuous_analysis(summary_stats, control_group_name, n_samples=10000, ci_level=0.95):
    # Performs Bayesian analysis for continuous outcomes.
    # Uses a t-distribution approximation for the posterior of the mean for each group.
    # Assumes summary_stats contains 'Variation', 'Users', 'Mean_Value', 'Std_Dev'.
    results = {}
    
    if 'Variation' not in summary_stats.columns:
        return None, "Summary statistics missing 'Variation' column."
    if not {'Users', 'Mean_Value', 'Std_Dev'}.issubset(summary_stats.columns):
        return None, "Summary statistics missing one or more required columns: 'Users', 'Mean_Value', 'Std_Dev'."

    for index, row in summary_stats.iterrows():
        var_name = row['Variation']
        n = int(row['Users'])
        sample_mean = row['Mean_Value']
        sample_std_dev = row['Std_Dev']
        
        current_samples = np.array([])
        posterior_mean_val = np.nan
        ci_low_val = np.nan
        ci_high_val = np.nan

        if n <= 1: # Not enough data for t-distribution or std_dev calculation
            current_samples = np.full(n_samples, sample_mean if n == 1 else np.nan) # If n=1, mean is the only data point
            posterior_mean_val = sample_mean if n == 1 else np.nan
        elif sample_std_dev == 0 and n > 1: # All values are the same
            current_samples = np.full(n_samples, sample_mean)
            posterior_mean_val = sample_mean
            ci_low_val = sample_mean
            ci_high_val = sample_mean
        else: # Standard case for t-distribution
            df = n - 1
            std_err = sample_std_dev / np.sqrt(n)
            # Generate samples from the posterior t-distribution for the mean
            # μ ~ sample_mean + std_err * t(df)
            t_samples = t_dist.rvs(df, size=n_samples)
            current_samples = sample_mean + (std_err * t_samples)
            
            posterior_mean_val = np.mean(current_samples) # Should be very close to sample_mean
            # Calculate credible interval from the samples
            ci_low_val = np.percentile(current_samples, (1 - ci_level) / 2 * 100)
            ci_high_val = np.percentile(current_samples, (1 - (1 - ci_level) / 2) * 100)

        results[var_name] = {
            'samples': current_samples,
            'posterior_mean': posterior_mean_val,
            'mean_ci_low': ci_low_val,
            'mean_ci_high': ci_high_val,
            'diff_samples_vs_control': None # Initialize
        }

    if control_group_name not in results:
        return None, f"Control group '{control_group_name}' not found in results. Available: {list(results.keys())}"
    
    control_samples = results[control_group_name]['samples']
    if np.all(np.isnan(control_samples)): # Check if control samples are all NaN (e.g. N_control <=0)
        # If control samples are unusable, cannot calculate differences or P(Better)
        for var_name in results:
            results[var_name]['prob_better_than_control'] = np.nan
            results[var_name]['diff_ci_low'] = np.nan
            results[var_name]['diff_ci_high'] = np.nan
            results[var_name]['expected_diff_abs'] = np.nan
            results[var_name]['prob_best'] = np.nan # Cannot determine best if control is invalid
        st.warning(f"Control group '{control_group_name}' has insufficient data (N <= 1 or undefined mean/std_dev) for Bayesian comparison.")

    else: # Proceed with comparisons if control samples are valid
        for var_name, data in results.items():
            if var_name == control_group_name:
                data['prob_better_than_control'] = None # Control cannot be "better than control"
                data['diff_ci_low'] = None
                data['diff_ci_high'] = None
                data['expected_diff_abs'] = None
                continue

            var_samples = data['samples']
            if np.all(np.isnan(var_samples)): # If variation samples are all NaN
                data['prob_better_than_control'] = np.nan
                data['diff_ci_low'] = np.nan
                data['diff_ci_high'] = np.nan
                data['expected_diff_abs'] = np.nan
                data['diff_samples_vs_control'] = np.full(n_samples, np.nan)
                continue

            diff_samples = var_samples - control_samples
            data['diff_samples_vs_control'] = diff_samples
            
            if np.all(np.isnan(diff_samples)): # if either var_samples or control_samples were all NaN
                data['prob_better_than_control'] = np.nan
                data['diff_ci_low'] = np.nan
                data['diff_ci_high'] = np.nan
                data['expected_diff_abs'] = np.nan
            else:
                data['prob_better_than_control'] = np.mean(diff_samples[~np.isnan(diff_samples)] > 0) if not np.all(np.isnan(diff_samples)) else np.nan
                data['diff_ci_low'] = np.nanpercentile(diff_samples, (1 - ci_level) / 2 * 100)
                data['diff_ci_high'] = np.nanpercentile(diff_samples, (1 - (1 - ci_level) / 2) * 100)
                data['expected_diff_abs'] = np.nanmean(diff_samples)

    # Calculate P(Being Best) for continuous outcomes
    all_var_names_cont = summary_stats['Variation'].tolist()
    # Filter out variations that had insufficient data for sampling
    valid_var_names_for_pbest = [name for name in all_var_names_cont if name in results and not np.all(np.isnan(results[name]['samples']))]

    if not valid_var_names_for_pbest:
        for var_name in all_var_names_cont:
            if var_name in results: results[var_name]['prob_best'] = np.nan
            else: results[var_name] = {'prob_best': np.nan} # Should not happen if summary_stats is source
        return results, "No variations with valid data found for P(Best) calculation in continuous Bayesian analysis."

    # Create matrix of samples only for valid variations
    all_samples_matrix_cont = np.array([results[var]['samples'] for var in valid_var_names_for_pbest])
    
    best_variation_counts_cont = np.zeros(len(all_var_names_cont))

    if all_samples_matrix_cont.ndim == 2 and all_samples_matrix_cont.shape[0] > 0 and all_samples_matrix_cont.shape[1] == n_samples:
        # Handle NaNs in samples matrix by finding argmax row-wise, ignoring NaNs if possible for each sample iteration
        # This is complex if NaNs are not uniform. A simpler approach for now: if any sample vector is all NaN, it can't be best.
        # The filtering for valid_var_names_for_pbest should prevent all-NaN sample vectors in all_samples_matrix_cont.
        
        best_indices_cont = np.nanargmax(all_samples_matrix_cont, axis=0) # nanargmax returns index of max, ignoring nans
        
        for i, best_idx_in_temp_matrix in enumerate(best_indices_cont):
            # Check if the max value for this iteration was NaN (meaning all values in that column were NaN)
            if np.isnan(all_samples_matrix_cont[best_idx_in_temp_matrix, i]):
                continue # Skip this iteration if no non-NaN max was found

            best_var_name_this_iter = valid_var_names_for_pbest[best_idx_in_temp_matrix]
            original_idx_for_counts = all_var_names_cont.index(best_var_name_this_iter)
            best_variation_counts_cont[original_idx_for_counts] += 1
        
        # Count valid iterations for P(Best)
        # An iteration is valid if not all values in that column of all_samples_matrix_cont were NaN
        valid_iterations_for_pbest = np.sum(~np.all(np.isnan(all_samples_matrix_cont), axis=0))

        if valid_iterations_for_pbest > 0:
            prob_best_cont = best_variation_counts_cont / valid_iterations_for_pbest
        else: # All iterations had all NaNs, so P(Best) is NaN for all
            prob_best_cont = np.full(len(all_var_names_cont), np.nan)

        for i, var_name in enumerate(all_var_names_cont):
            if var_name in results:
                results[var_name]['prob_best'] = prob_best_cont[i]
            else: # Should not be reached if all_var_names_cont is from summary_stats
                results[var_name] = {'prob_best': np.nan}
                
    else: # Handle cases with only one valid variation or issues with sample matrix
        for var_name in all_var_names_cont:
            if var_name in results:
                is_single_valid_var = (len(valid_var_names_for_pbest) == 1 and var_name == valid_var_names_for_pbest[0])
                results[var_name]['prob_best'] = 1.0 if is_single_valid_var else (0.0 if len(valid_var_names_for_pbest) > 1 else np.nan)
            else:
                results[var_name] = {'prob_best': np.nan}
                
    return results, None


# --- Page Functions ---
def show_introduction_page():
    # Displays the introduction to A/B testing.
    st.header("Introduction to A/B Testing �")
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
    * 🛡️ **Reduce Risk:** Test changes on a smaller scale before rolling them out to your entire user base, minimizing the impact of potentially negative changes. (Changed icon for better representation)
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
    # Page for designing A/B tests, including sample size calculators.
    st.header("Designing Your A/B Test 📐")
    st.markdown("A crucial step in designing an A/B test is determining the appropriate sample size. This calculator will help you estimate the number of users needed per variation.")
    st.markdown("---")
    metric_type_ss = st.radio("Select your primary metric type for sample size calculation:", ('Binary (e.g., Conversion Rate)', 'Continuous (e.g., Average Order Value)'), key="ss_metric_type_radio_c7")
    st.markdown("---")
    if metric_type_ss == 'Binary (e.g., Conversion Rate)':
        st.subheader("Sample Size Calculator (for Binary Outcomes)")
        st.markdown("**Calculator Inputs:**")
        cols_bin = st.columns(2)
        with cols_bin[0]: baseline_cr_percent = st.number_input(label="Baseline Conversion Rate (BCR) (%)", min_value=0.1, max_value=99.9, value=5.0, step=0.1, format="%.1f", help="Current CR of control (e.g., 5% for 5 out of 100).", key="ss_bcr_c7")
        with cols_bin[1]: mde_abs_percent = st.number_input(label="Minimum Detectable Effect (MDE) - Absolute (%)", min_value=0.1, max_value=50.0, value=1.0, step=0.1, format="%.1f", help="Smallest absolute CR increase to detect (e.g., 1% for 5% to 6%).", key="ss_mde_c7")
        cols2_bin = st.columns(2)
        with cols2_bin[0]: power_percent_bin = st.slider(label="Statistical Power (1 - \u03B2) (%)", min_value=50, max_value=99, value=80, step=1, format="%d%%", help="Probability of detecting an effect if one exists (typically 80-90%).", key="ss_power_c7")
        with cols2_bin[1]: alpha_percent_bin = st.slider(label="Significance Level (\u03B1) (%) - Two-sided", min_value=1, max_value=20, value=5, step=1, format="%d%%", help="Risk of false positive (typically 1-5%).", key="ss_alpha_c7")
        num_variations_ss_bin = st.number_input(label="Number of Variations (including Control)", min_value=2, value=2, step=1, help="Total versions (e.g., Control + 1 Var = 2).", key="ss_num_var_c7")
        st.markdown("<p style='font-size: smaller; font-style: italic;'>Note: For binary, based on pairwise comparisons vs. control at specified α.</p>", unsafe_allow_html=True)
        if st.button("Calculate Sample Size (Binary)", key="ss_calc_button_c7_bin"):
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
        with cols_cont[0]: baseline_mean = st.number_input(label="Baseline Mean (Control Group)", value=100.0, step=1.0, format="%.2f", help="Current average value of your metric for the control group.", key="ss_mean_c7")
        with cols_cont[1]: std_dev = st.number_input(label="Standard Deviation (of the metric)", value=20.0, min_value=0.01, step=0.1, format="%.2f", help="Estimated standard deviation of your continuous metric. Get from historical data if possible. Must be > 0.", key="ss_stddev_c7") # Min value updated
        mde_abs_mean = st.number_input(label="Minimum Detectable Effect (MDE) - Absolute Mean Difference", value=5.0,min_value=0.01, step=0.1, format="%.2f", help="Smallest absolute difference in means you want to detect (e.g., $2 increase). Must be > 0.", key="ss_mde_mean_c7")
        cols2_cont = st.columns(2)
        with cols2_cont[0]: power_percent_cont = st.slider(label="Statistical Power (1 - \u03B2) (%)", min_value=50, max_value=99, value=80, step=1, format="%d%%", help="Typically 80-90%.", key="ss_power_cont_c7")
        with cols2_cont[1]: alpha_percent_cont = st.slider(label="Significance Level (\u03B1) (%) - Two-sided", min_value=1, max_value=20, value=5, step=1, format="%d%%", help="Typically 1-5%.", key="ss_alpha_cont_c7")
        num_variations_ss_cont = st.number_input(label="Number of Variations (including Control)", min_value=2, value=2, step=1, help="Total versions.", key="ss_num_var_cont_c7")
        st.markdown("<p style='font-size: smaller; font-style: italic;'>Note: For continuous, based on pairwise comparisons vs. control at specified α, assuming similar standard deviations across groups.</p>", unsafe_allow_html=True)
        if st.button("Calculate Sample Size (Continuous)", key="ss_calc_button_c7_cont"):
            power, alpha = power_percent_cont/100.0, alpha_percent_cont/100.0
            sample_size, error_msg = calculate_continuous_sample_size(baseline_mean, std_dev, mde_abs_mean, power, alpha, num_variations_ss_cont)
            if error_msg: st.error(error_msg)
            elif sample_size:
                st.success("Calculation Successful!")
                target_mean = baseline_mean + mde_abs_mean # Assuming positive MDE for target display
                res_cols = st.columns(2); res_cols[0].metric("Required Sample Size PER Variation", f"{sample_size:,}"); res_cols[1].metric("Total Required Sample Size", f"{(sample_size * num_variations_ss_cont):,}")
                st.markdown(f"**Summary of Inputs Used:**\n- Baseline Mean: `{baseline_mean:.2f}`\n- Estimated Standard Deviation: `{std_dev:.2f}`\n- Absolute MDE (Mean Difference): `{mde_abs_mean:.2f}` (Targeting a mean of approx. `{target_mean:.2f}` or `{baseline_mean - mde_abs_mean:.2f}` for variations)\n- Statistical Power: `{power_percent_cont}%` (1 - \u03B2)\n- Significance Level (\u03B1): `{alpha_percent_cont}%` (two-sided)\n- Number of Variations: `{num_variations_ss_cont}`.")
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
    st.info("Sample Size Calculator now supports both Binary and Continuous outcomes!")

def show_analyze_results_page():
    # Page for analyzing A/B test results.
    st.header("Analyze Your A/B Test Results 📊")
    st.markdown("Upload your A/B test data (as a CSV file) to perform an analysis.")
    st.markdown("---")

    # Initialize session state variables if they don't exist
    if 'analysis_done_c7' not in st.session_state: st.session_state.analysis_done_c7 = False
    if 'df_analysis_c7' not in st.session_state: st.session_state.df_analysis_c7 = None
    if 'metric_type_analysis_c7' not in st.session_state: st.session_state.metric_type_analysis_c7 = 'Binary'
    if 'variation_col_analysis_c7' not in st.session_state: st.session_state.variation_col_analysis_c7 = None
    if 'outcome_col_analysis_c7' not in st.session_state: st.session_state.outcome_col_analysis_c7 = None
    if 'success_value_analysis_c7' not in st.session_state: st.session_state.success_value_analysis_c7 = None
    if 'control_group_name_analysis_c7' not in st.session_state: st.session_state.control_group_name_analysis_c7 = None
    if 'alpha_for_analysis_c7' not in st.session_state: st.session_state.alpha_for_analysis_c7 = 0.05
    if 'freq_summary_stats_c7' not in st.session_state: st.session_state.freq_summary_stats_c7 = None
    if 'bayesian_results_binary_c7' not in st.session_state: st.session_state.bayesian_results_binary_c7 = None
    if 'bayesian_results_continuous_c7' not in st.session_state: st.session_state.bayesian_results_continuous_c7 = None # New for continuous
    if 'metric_col_name_c7' not in st.session_state: st.session_state.metric_col_name_c7 = None

    uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"], key="file_uploader_cycle7")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df_analysis_c7 = df 
            st.success("File Uploaded Successfully!")
            st.markdown("**Data Preview (first 5 rows):**"); st.dataframe(df.head())
            st.markdown("---")

            st.subheader("1. Map Data & Select Metric Type")
            columns = df.columns.tolist()
            map_col1, map_col2, map_col3 = st.columns(3)
            with map_col1: st.session_state.variation_col_analysis_c7 = st.selectbox("Select 'Variation ID' column:", options=columns, index=0 if columns else -1, key="var_col_c7")
            with map_col2: st.session_state.outcome_col_analysis_c7 = st.selectbox("Select 'Outcome' column:", options=columns, index=len(columns)-1 if len(columns)>1 else (0 if columns else -1), key="out_col_c7")
            with map_col3: st.session_state.metric_type_analysis_c7 = st.radio("Select Metric Type for Outcome Column:", ('Binary', 'Continuous'), key="metric_type_analysis_radio_c7", horizontal=True)

            if st.session_state.metric_type_analysis_c7 == 'Binary':
                success_value_options = []
                if st.session_state.outcome_col_analysis_c7 and st.session_state.outcome_col_analysis_c7 in df.columns:
                    unique_outcomes = df[st.session_state.outcome_col_analysis_c7].unique()
                    if len(unique_outcomes) == 1: st.warning(f"Outcome column '{st.session_state.outcome_col_analysis_c7}' has only one value: `{unique_outcomes[0]}`.")
                    elif len(unique_outcomes) > 2 and len(unique_outcomes) <=10 : st.warning(f"Outcome column '{st.session_state.outcome_col_analysis_c7}' has >2 unique values: `{unique_outcomes}`. Please select the value representing 'Conversion' or 'Success'.")
                    elif len(unique_outcomes) > 10: st.warning(f"Outcome column '{st.session_state.outcome_col_analysis_c7}' has many unique values ({len(unique_outcomes)}). Ensure this is a binary outcome and select the success value.")

                    success_value_options = unique_outcomes
                    if len(success_value_options) > 0:
                        # Attempt to find a common success indicator like 1 or True if present
                        default_success_idx = 0
                        str_options = [str(val) for val in success_value_options]
                        if '1' in str_options: default_success_idx = str_options.index('1')
                        elif 'True' in str_options: default_success_idx = str_options.index('True')
                        elif 'true' in str_options: default_success_idx = str_options.index('true')
                        
                        success_value_str = st.selectbox(f"Which value in '{st.session_state.outcome_col_analysis_c7}' is 'Conversion' (Success)?", options=str_options, index=default_success_idx, key="succ_val_c7")
                        
                        original_dtype = df[st.session_state.outcome_col_analysis_c7].dtype
                        # Convert selected string back to original type if possible
                        if success_value_str.lower() == 'nan' and any(pd.isna(val) for val in success_value_options): st.session_state.success_value_analysis_c7 = np.nan
                        elif pd.api.types.is_numeric_dtype(original_dtype) and not pd.api.types.is_bool_dtype(original_dtype):
                            try: st.session_state.success_value_analysis_c7 = original_dtype.type(success_value_str)
                            except ValueError: st.session_state.success_value_analysis_c7 = success_value_str # Fallback to string if conversion fails
                        elif pd.api.types.is_bool_dtype(original_dtype): st.session_state.success_value_analysis_c7 = (success_value_str.lower() == 'true') 
                        else: st.session_state.success_value_analysis_c7 = success_value_str
                    else: st.warning(f"Could not determine distinct values in outcome column '{st.session_state.outcome_col_analysis_c7}'.")
                else: st.warning("Please select a valid outcome column.")
            else: # Continuous
                st.session_state.success_value_analysis_c7 = None 
                if st.session_state.outcome_col_analysis_c7 and st.session_state.outcome_col_analysis_c7 in df.columns and not pd.api.types.is_numeric_dtype(df[st.session_state.outcome_col_analysis_c7]):
                    st.error(f"For 'Continuous' metric type, the outcome column '{st.session_state.outcome_col_analysis_c7}' must be numeric. Current type: {df[st.session_state.outcome_col_analysis_c7].dtype}. Please select a numeric column or check your data.")
            
            st.markdown("---"); st.subheader("2. Select Your Control Group & Analysis Alpha")
            if st.session_state.variation_col_analysis_c7 and st.session_state.variation_col_analysis_c7 in df.columns and st.session_state.df_analysis_c7 is not None:
                variation_names = st.session_state.df_analysis_c7[st.session_state.variation_col_analysis_c7].unique().tolist()
                if variation_names: st.session_state.control_group_name_analysis_c7 = st.selectbox("Select 'Control Group':", options=variation_names, index=0, key="ctrl_grp_c7")
                else: st.warning(f"No unique variations found in column '{st.session_state.variation_col_analysis_c7}'.")
            else: st.warning("Please select a valid variation column.")
            st.session_state.alpha_for_analysis_c7 = st.slider("Significance Level (\u03B1) for Analysis (%)", 1, 20, 5, 1, key="alpha_analysis_c7_slider") / 100.0 # Max 20% alpha
            
            analysis_button_label = f"🚀 Run Analysis ({st.session_state.metric_type_analysis_c7} Outcome)"
            if st.button(analysis_button_label, key="run_analysis_button_cycle7"):
                # Reset states before new analysis
                st.session_state.analysis_done_c7 = False
                st.session_state.freq_summary_stats_c7 = None
                st.session_state.bayesian_results_binary_c7 = None
                st.session_state.bayesian_results_continuous_c7 = None

                valid_setup = True
                # Validations
                if not st.session_state.variation_col_analysis_c7 or not st.session_state.outcome_col_analysis_c7 or st.session_state.control_group_name_analysis_c7 is None:
                    st.error("Please complete all column mapping and control group selections."); valid_setup = False
                if st.session_state.metric_type_analysis_c7 == 'Binary' and st.session_state.success_value_analysis_c7 is None:
                    st.error("For Binary outcome, please specify the 'Conversion (Success)' value."); valid_setup = False
                if st.session_state.metric_type_analysis_c7 == 'Continuous':
                    if not st.session_state.outcome_col_analysis_c7 or st.session_state.outcome_col_analysis_c7 not in st.session_state.df_analysis_c7.columns:
                        st.error("Please select a valid outcome column for continuous analysis."); valid_setup = False
                    elif not pd.api.types.is_numeric_dtype(st.session_state.df_analysis_c7[st.session_state.outcome_col_analysis_c7]):
                        st.error(f"For 'Continuous' metric type, outcome column '{st.session_state.outcome_col_analysis_c7}' must be numeric."); valid_setup = False
                
                if valid_setup:
                    try:
                        current_df = st.session_state.df_analysis_c7.copy()
                        var_col = st.session_state.variation_col_analysis_c7
                        out_col = st.session_state.outcome_col_analysis_c7
                        
                        # Frequentist Summary Stats Calculation (common for both Binary and Continuous display)
                        if st.session_state.metric_type_analysis_c7 == 'Binary':
                            if pd.isna(st.session_state.success_value_analysis_c7): 
                                current_df['__outcome_processed__'] = current_df[out_col].isna().astype(int)
                            else: 
                                current_df['__outcome_processed__'] = (current_df[out_col] == st.session_state.success_value_analysis_c7).astype(int)
                            
                            summary_stats = current_df.groupby(var_col).agg(
                                Users=('__outcome_processed__', 'count'), 
                                Converts=('__outcome_processed__', 'sum')
                            ).reset_index()
                            summary_stats.rename(columns={var_col: 'Variation', 'Converts': 'Conversions'}, inplace=True)
                            if summary_stats['Users'].sum() == 0: st.error("No users found after grouping for binary analysis."); st.stop()
                            summary_stats['Metric Value (%)'] = (summary_stats['Conversions'] / summary_stats['Users'].replace(0, np.nan) * 100).round(2)
                            st.session_state.metric_col_name_c7 = 'Metric Value (%)'
                            st.session_state.freq_summary_stats_c7 = summary_stats.copy()

                            # Bayesian Analysis for Binary
                            bayesian_results_bin, bayesian_error_bin = run_bayesian_binary_analysis(
                                st.session_state.freq_summary_stats_c7.copy(), 
                                st.session_state.control_group_name_analysis_c7, 
                                ci_level=(1 - st.session_state.alpha_for_analysis_c7)
                            )
                            if bayesian_error_bin: st.error(f"Bayesian Binary Analysis Error: {bayesian_error_bin}")
                            else: st.session_state.bayesian_results_binary_c7 = bayesian_results_bin

                        elif st.session_state.metric_type_analysis_c7 == 'Continuous':
                            current_df[out_col] = pd.to_numeric(current_df[out_col], errors='coerce') # Ensure numeric, coerce errors to NaN
                            # Filter out rows where the outcome became NaN after coercion, as they can't be used in mean/std calculations
                            current_df_cleaned = current_df.dropna(subset=[out_col])

                            if current_df_cleaned.empty:
                                st.error(f"No valid numeric data in outcome column '{out_col}' after attempting conversion and removing invalid entries."); st.stop()

                            summary_stats = current_df_cleaned.groupby(var_col).agg(
                                Users=(out_col, 'count'), 
                                Mean_Value=(out_col, 'mean'), 
                                Std_Dev=(out_col, 'std'), 
                                Median_Value=(out_col,'median'), # Adding Median
                                Std_Err=(out_col, lambda x: x.std(ddof=1) / np.sqrt(x.count()) if x.count() > 0 else np.nan)
                            ).reset_index()
                            summary_stats.rename(columns={var_col: 'Variation'}, inplace=True)
                            
                            if summary_stats['Users'].sum() == 0: st.error("No users found after grouping for continuous analysis."); st.stop()
                            
                            # Round numeric columns for display
                            for col_to_round in ['Mean_Value', 'Std_Dev', 'Median_Value', 'Std_Err']:
                                if col_to_round in summary_stats.columns:
                                     summary_stats[col_to_round] = summary_stats[col_to_round].round(3)
                            
                            st.session_state.metric_col_name_c7 = 'Mean_Value'
                            st.session_state.freq_summary_stats_c7 = summary_stats.copy()

                            # Bayesian Analysis for Continuous
                            bayesian_results_cont, bayesian_error_cont = run_bayesian_continuous_analysis(
                                st.session_state.freq_summary_stats_c7.copy(), # Pass the summary with N, Mean, StdDev
                                st.session_state.control_group_name_analysis_c7,
                                ci_level=(1 - st.session_state.alpha_for_analysis_c7)
                            )
                            if bayesian_error_cont: st.error(f"Bayesian Continuous Analysis Error: {bayesian_error_cont}")
                            else: st.session_state.bayesian_results_continuous_c7 = bayesian_results_cont
                        
                        st.session_state.analysis_done_c7 = True
                    except Exception as e: st.error(f"An error occurred during data processing: {e}"); st.exception(e)
        except Exception as e: st.error(f"Error reading/processing CSV: {e}"); st.exception(e)
    else: st.info("Upload a CSV file to begin analysis.")

    # --- Display Results ---
    if st.session_state.analysis_done_c7:
        alpha_display = st.session_state.alpha_for_analysis_c7
        metric_type_display = st.session_state.metric_type_analysis_c7
        summary_stats_display = st.session_state.freq_summary_stats_c7
        metric_col_name_display = st.session_state.metric_col_name_c7 # This is 'Metric Value (%)' or 'Mean_Value'

        st.markdown("---"); st.subheader(f"Frequentist Analysis Results ({metric_type_display} Outcome)")
        if summary_stats_display is not None and metric_col_name_display is not None:
            st.markdown("##### 📊 Descriptive Statistics")
            if metric_type_display == 'Binary': 
                st.dataframe(summary_stats_display[['Variation', 'Users', 'Conversions', metric_col_name_display]].fillna('N/A (0 Users)'))
            else: # Continuous
                st.dataframe(summary_stats_display[['Variation', 'Users', 'Mean_Value', 'Median_Value', 'Std_Dev', 'Std_Err']].fillna('N/A'))
            
            if metric_type_display == 'Continuous':
                for index, row in summary_stats_display.iterrows():
                    if row['Std_Dev'] == 0 and row['Users'] > 1 : 
                        st.warning(f"⚠️ Variation '{row['Variation']}' has a Standard Deviation of 0 for the outcome '{st.session_state.outcome_col_analysis_c7}'. This means all outcome values for this variation are identical. This can impact the interpretation of statistical tests like the t-test (may result in NaN p-values or errors if variance is assumed unequal).")
            
            # Bar chart for the primary metric
            if metric_col_name_display in summary_stats_display.columns:
                chart_data = summary_stats_display.set_index('Variation')[metric_col_name_display].fillna(0)
                if not chart_data.empty: 
                    st.bar_chart(chart_data, y=metric_col_name_display) # Use default x-axis (Variation)
            
            st.markdown(f"##### 📈 Comparison vs. Control ('{st.session_state.control_group_name_analysis_c7}')")
            control_data_rows = summary_stats_display[summary_stats_display['Variation'] == st.session_state.control_group_name_analysis_c7]
            
            if control_data_rows.empty: st.error(f"Control group '{st.session_state.control_group_name_analysis_c7}' data missing in summary statistics.")
            else:
                control_data = control_data_rows.iloc[0]; comparison_results_freq = []
                # ... (Frequentist comparison logic - largely unchanged, ensure keys match new session state vars)
                if metric_type_display == 'Binary':
                    control_users, control_conversions = control_data['Users'], control_data['Conversions']
                    control_metric_val = control_conversions / control_users if control_users > 0 else 0
                    for index, row in summary_stats_display.iterrows():
                        var_name, var_users, var_conversions = row['Variation'], row['Users'], row['Conversions']
                        if var_name == st.session_state.control_group_name_analysis_c7: continue
                        var_metric_val = var_conversions / var_users if var_users > 0 else 0
                        p_val_disp, ci_disp, sig_disp, abs_disp, rel_disp = 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
                        if control_users > 0 and var_users > 0:
                            abs_uplift = var_metric_val - control_metric_val; abs_disp = f"{abs_uplift*100:.2f}"
                            rel_disp = f"{(abs_uplift / control_metric_val) * 100:.2f}%" if control_metric_val != 0 else "N/A (Control CR is 0)" # Handle control CR = 0
                            count, nobs = np.array([var_conversions, control_conversions]), np.array([var_users, control_users])
                            if not (np.any(count < 0) or np.any(nobs <= 0) or np.any(count > nobs)): # Basic validation
                                try:
                                    _, p_value = proportions_ztest(count, nobs, alternative='two-sided')
                                    p_val_disp = f"{p_value:.4f}"
                                    sig_bool = p_value < alpha_display; sig_disp = f"Yes (p={p_value:.4f})" if sig_bool else f"No (p={p_value:.4f})"
                                    ci_low, ci_high = confint_proportions_2indep(var_conversions, var_users, control_conversions, control_users, method='wald', alpha=alpha_display)
                                    ci_disp = f"[{ci_low*100:.2f}, {ci_high*100:.2f}]"
                                except Exception as e_prop_z: 
                                    p_val_disp, ci_disp, sig_disp = 'Error', 'Error', 'Error'
                                    st.caption(f"Z-test error for {var_name}: {e_prop_z}")
                            else: sig_disp = 'N/A (Invalid counts/nobs for Z-test)'
                        else: sig_disp = 'N/A (Zero users in control or variation)'
                        comparison_results_freq.append({"Variation": var_name, "CR (%)": f"{var_metric_val*100:.2f}", "Abs. Uplift (%)": abs_disp, "Rel. Uplift (%)": rel_disp, "P-value": p_val_disp, f"CI {100*(1-alpha_display):.0f}% Diff (%)": ci_disp, "Significant?": sig_disp})
                
                elif metric_type_display == 'Continuous':
                    control_mean = control_data['Mean_Value']
                    # Get raw data for t-test
                    control_group_data_raw = st.session_state.df_analysis_c7[st.session_state.df_analysis_c7[st.session_state.variation_col_analysis_c7] == st.session_state.control_group_name_analysis_c7][st.session_state.outcome_col_analysis_c7].dropna()
                    control_users_raw = len(control_group_data_raw)

                    for index, row in summary_stats_display.iterrows():
                        var_name, var_mean = row['Variation'], row['Mean_Value']
                        if var_name == st.session_state.control_group_name_analysis_c7: continue
                        
                        var_group_data_raw = st.session_state.df_analysis_c7[st.session_state.df_analysis_c7[st.session_state.variation_col_analysis_c7] == var_name][st.session_state.outcome_col_analysis_c7].dropna()
                        var_users_raw = len(var_group_data_raw)

                        p_val_disp, ci_disp, sig_disp, abs_disp, rel_disp = 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
                        
                        if control_users_raw > 1 and var_users_raw > 1: # t-test needs at least 2 samples per group
                            abs_diff_means = var_mean - control_mean; abs_disp = f"{abs_diff_means:.3f}"
                            rel_disp = f"{(abs_diff_means / control_mean) * 100:.2f}%" if control_mean != 0 else "N/A (Control Mean is 0)"
                            
                            # Perform t-test only if std devs are not both zero (or if equal_var=False can handle one zero)
                            # Welch's t-test (equal_var=False) is generally robust.
                            # It can handle zero variance in one group if the other is non-zero. If both are zero, it might error or give NaN.
                            if control_group_data_raw.var(ddof=1) == 0 and var_group_data_raw.var(ddof=1) == 0 and control_users_raw > 1 and var_users_raw > 1:
                                # If both groups have zero variance, means are exact. Difference is exact. P-value depends on whether means are different.
                                if control_mean == var_mean:
                                    p_value = 1.0 # No difference
                                else:
                                    p_value = 0.0 # Difference is certain
                                t_stat = np.inf if control_mean != var_mean else 0.0 # Or some indicator
                                p_val_disp = f"{p_value:.4f}"
                                sig_bool = p_value < alpha_display; sig_disp = f"Yes (p={p_value:.4f})" if sig_bool else f"No (p={p_value:.4f})"
                                # CI for difference when variances are zero: it's just the point difference [diff, diff]
                                ci_disp = f"[{abs_diff_means:.3f}, {abs_diff_means:.3f}] (Exact due to zero variance)"

                            else: # At least one group has non-zero variance, or ttest_ind can handle it
                                try:
                                    t_stat, p_value = ttest_ind(var_group_data_raw, control_group_data_raw, equal_var=False, nan_policy='omit') # Welch's t-test
                                    p_val_disp = f"{p_value:.4f}"
                                    sig_bool = p_value < alpha_display; sig_disp = f"Yes (p={p_value:.4f})" if sig_bool else f"No (p={p_value:.4f})"
                                    
                                    # Calculate CI for difference of means (Welch-Satterthwaite for df if needed, or use simpler t_dist with min_df)
                                    N1, N2 = var_users_raw, control_users_raw
                                    s1_sq, s2_sq = var_group_data_raw.var(ddof=1), control_group_data_raw.var(ddof=1)
                                    
                                    # Handle cases where variance might be zero for one group
                                    if N1 > 0 and N2 > 0 : # s1_sq or s2_sq can be 0
                                        pooled_se_diff_sq = (s1_sq / N1) + (s2_sq / N2)
                                        if pooled_se_diff_sq > 0 : # Ensure SE is positive
                                            pooled_se_diff = math.sqrt(pooled_se_diff_sq)
                                            # Degrees of freedom for Welch's t-test (can be complex)
                                            # Using a simpler approximation: min(N1-1, N2-1) for critical t-value
                                            df_t_approx = min(N1 - 1, N2 - 1)
                                            if df_t_approx > 0:
                                                t_crit = t_dist.ppf(1 - alpha_display / 2, df=df_t_approx)
                                                ci_low_mean_diff, ci_high_mean_diff = abs_diff_means - t_crit * pooled_se_diff, abs_diff_means + t_crit * pooled_se_diff
                                                ci_disp = f"[{ci_low_mean_diff:.3f}, {ci_high_mean_diff:.3f}]"
                                            else: ci_disp = "N/A (df for CI <=0)"
                                        else: ci_disp = "N/A (SE for diff is 0)" # Both variances likely 0 and means are same
                                    else: ci_disp = "N/A (N<=0 in a group for CI)"
                                except Exception as e_ttest: 
                                    p_val_disp, ci_disp, sig_disp = 'Error', 'Error', 'Error'
                                    st.caption(f"T-test error for {var_name}: {e_ttest}")
                        else: sig_disp = 'N/A (Too few users for reliable t-test: N <= 1 in a group)'
                        comparison_results_freq.append({"Variation": var_name, "Mean Value": f"{var_mean:.3f}", "Abs. Diff.": abs_disp, "Rel. Diff. (%)": rel_disp, "P-value": p_val_disp, f"CI {100*(1-alpha_display):.0f}% Diff.": ci_disp, "Significant?": sig_disp})

                if comparison_results_freq:
                    comparison_df_freq = pd.DataFrame(comparison_results_freq)
                    st.dataframe(comparison_df_freq)
                    for _, row_data in comparison_df_freq.iterrows():
                        if "Yes" in str(row_data["Significant?"]): st.caption(f"Frequentist: Diff between **{row_data['Variation']}** & control is significant at {alpha_display*100:.0f}% level (P-value: {row_data['P-value']}).")
                        elif "No" in str(row_data["Significant?"]): st.caption(f"Frequentist: Diff between **{row_data['Variation']}** & control is not significant at {alpha_display*100:.0f}% level (P-value: {row_data['P-value']}).")
            
            # Box plots for continuous data
            if metric_type_display == 'Continuous' and st.session_state.df_analysis_c7 is not None:
                st.markdown("##### Distribution of Outcomes by Variation (Box Plots)")
                try:
                    # Use the already cleaned (numeric converted, NaN dropped) df for plotting if available
                    # Or re-process from st.session_state.df_analysis_c7
                    plot_df_cont = st.session_state.df_analysis_c7.copy()
                    plot_df_cont[st.session_state.outcome_col_analysis_c7] = pd.to_numeric(plot_df_cont[st.session_state.outcome_col_analysis_c7], errors='coerce')
                    plot_df_cont.dropna(subset=[st.session_state.outcome_col_analysis_c7], inplace=True)
                    
                    if not plot_df_cont.empty and st.session_state.variation_col_analysis_c7 in plot_df_cont.columns:
                        # Order variations for plotting, ensure control is first if present
                        unique_vars_plot = sorted(plot_df_cont[st.session_state.variation_col_analysis_c7].unique())
                        control_name = st.session_state.control_group_name_analysis_c7
                        if control_name in unique_vars_plot:
                            unique_vars_plot.insert(0, unique_vars_plot.pop(unique_vars_plot.index(control_name)))

                        boxplot_data = []
                        valid_labels = []
                        for var_name_plot in unique_vars_plot:
                            data_series = plot_df_cont[plot_df_cont[st.session_state.variation_col_analysis_c7] == var_name_plot][st.session_state.outcome_col_analysis_c7]
                            if not data_series.empty:
                                boxplot_data.append(data_series)
                                valid_labels.append(var_name_plot)
                        
                        if not boxplot_data or not valid_labels or len(boxplot_data) != len(valid_labels):
                            st.caption("Not enough data for one or more variations to display box plots after handling non-numeric/empty outcomes.")
                        else:
                            fig_box, ax_box = plt.subplots(); 
                            ax_box.boxplot(boxplot_data, labels=valid_labels, patch_artist=True) # patch_artist for colors
                            ax_box.set_title(f"Outcome Distributions: {st.session_state.outcome_col_analysis_c7} by Variation"); 
                            ax_box.set_ylabel(st.session_state.outcome_col_analysis_c7); 
                            ax_box.set_xlabel("Variation")
                            plt.xticks(rotation=45, ha="right") # Rotate labels if many variations
                            plt.tight_layout()
                            st.pyplot(fig_box); plt.close(fig_box)
                    else: st.caption("Not enough data to display box plots after handling non-numeric outcomes or variation column missing.")
                except Exception as e_plot: st.warning(f"Could not generate box plots: {e_plot}")
        
        # --- Bayesian Analysis Display ---
        if metric_type_display == 'Binary':
            st.markdown("---"); st.subheader("Bayesian Analysis Results (Binary Outcome)")
            bayesian_results_to_display = st.session_state.bayesian_results_binary_c7
            if bayesian_results_to_display and summary_stats_display is not None and metric_col_name_display is not None:
                st.markdown(f"Using a Beta(1,1) uninformative prior. Credible Intervals (CrI) at {100*(1-alpha_display):.0f}% level.")
                bayesian_data_disp_bin = []; control_name_bayes_bin = st.session_state.control_group_name_analysis_c7
                
                # Ensure display order matches summary_stats_display if possible
                ordered_vars_for_bayes_display = summary_stats_display['Variation'].tolist()

                for var_name in ordered_vars_for_bayes_display:
                    if var_name not in bayesian_results_to_display: continue # Skip if var not in bayes results
                    b_res = bayesian_results_to_display[var_name]
                    
                    prob_better_html = f"<span title=\"Probability that this variation's true conversion rate is higher than the control's. Also consider the Credible Interval for Uplift to understand magnitude and uncertainty.\">{b_res.get('prob_better_than_control',0)*100:.2f}%</span>" if b_res.get('prob_better_than_control') is not None else "N/A (Control)"
                    cri_uplift_html = f"<span title=\"The range where the true uplift over control likely lies. If this interval includes 0, 'no difference' or a negative effect are plausible.\">[{b_res.get('uplift_ci_low', 0)*100:.2f}, {b_res.get('uplift_ci_high', 0)*100:.2f}]</span>" if b_res.get('uplift_ci_low') is not None else "N/A (Control)"
                    
                    bayesian_data_disp_bin.append({
                        "Variation": var_name, 
                        "Posterior Mean CR (%)": f"{b_res.get('mean_cr',0)*100:.2f}", 
                        f"{100*(1-alpha_display):.0f}% CrI for CR (%)": f"[{b_res.get('cr_ci_low',0)*100:.2f}, {b_res.get('cr_ci_high',0)*100:.2f}]", 
                        "P(Better > Control) (%)": prob_better_html, 
                        "Expected Uplift (abs %)": f"{b_res.get('expected_uplift_abs', 0)*100:.2f}" if b_res.get('expected_uplift_abs') is not None else "N/A (Control)", 
                        f"{100*(1-alpha_display):.0f}% CrI for Uplift (abs %)": cri_uplift_html, 
                        "P(Being Best) (%)": f"{b_res.get('prob_best',0)*100:.2f}"
                    })
                if bayesian_data_disp_bin:
                    bayesian_df_bin = pd.DataFrame(bayesian_data_disp_bin); st.markdown(bayesian_df_bin.to_html(escape=False), unsafe_allow_html=True)
                
                # Plots for Bayesian Binary
                st.markdown("##### Posterior Distributions for Conversion Rates (Binary)"); fig_cr_bin, ax_cr_bin = plt.subplots()
                # ... (Plotting logic from V0.6.1, adapted for c7 state vars) ...
                observed_max_cr_for_plot = 0.0
                numeric_crs_bin = pd.to_numeric(summary_stats_display[metric_col_name_display], errors='coerce') # metric_col_name_display is 'Metric Value (%)'
                if not numeric_crs_bin.empty and numeric_crs_bin.notna().any(): observed_max_cr_for_plot = numeric_crs_bin.max() / 100.0
                else: observed_max_cr_for_plot = 0.1 

                posterior_max_cr_for_plot_bin = 0.0
                all_posterior_highs_bin = [res.get('cr_ci_high') for res in bayesian_results_to_display.values() if res.get('cr_ci_high') is not None]
                if all_posterior_highs_bin: posterior_max_cr_for_plot_bin = max(all_posterior_highs_bin)
                
                final_x_limit_candidate_bin = max(observed_max_cr_for_plot, posterior_max_cr_for_plot_bin)
                x_cr_plot_limit_bin = min(1.0, final_x_limit_candidate_bin + 0.05) 
                if x_cr_plot_limit_bin <= 0.01: x_cr_plot_limit_bin = 0.1 
                x_cr_range_bin = np.linspace(0, x_cr_plot_limit_bin, 300)
                
                max_density_bin = 0
                for var_name in ordered_vars_for_bayes_display: # Use ordered list for consistent plot legend
                    if var_name not in bayesian_results_to_display: continue
                    b_res = bayesian_results_to_display[var_name]
                    alpha_p, beta_p = b_res.get('alpha_post', 1), b_res.get('beta_post', 1)
                    if alpha_p > 0 and beta_p > 0: 
                        posterior_pdf = beta_dist.pdf(x_cr_range_bin, alpha_p, beta_p)
                        ax_cr_bin.plot(x_cr_range_bin, posterior_pdf, label=f"{var_name} (α={alpha_p:.1f},β={beta_p:.1f})")
                        ax_cr_bin.fill_between(x_cr_range_bin, posterior_pdf, alpha=0.2)
                        if posterior_pdf is not None and np.any(np.isfinite(posterior_pdf)): 
                            finite_pdf = posterior_pdf[np.isfinite(posterior_pdf)]
                            if finite_pdf.size > 0: max_density_bin = max(max_density_bin, np.nanmax(finite_pdf))
                
                if max_density_bin > 0: ax_cr_bin.set_ylim(0, max_density_bin * 1.1)
                else: ax_cr_bin.set_ylim(0,1)
                ax_cr_bin.set_title("Posterior Distributions of CRs"); ax_cr_bin.set_xlabel("Conversion Rate"); ax_cr_bin.set_ylabel("Density"); ax_cr_bin.legend(); st.pyplot(fig_cr_bin); plt.close(fig_cr_bin)

                st.markdown("##### Posterior Distribution of Uplift (Variation CR - Control CR)")
                num_vars_to_plot_bin = sum(1 for var_name_uplift in bayesian_results_to_display if var_name_uplift != control_name_bayes_bin and bayesian_results_to_display[var_name_uplift].get('diff_samples_vs_control') is not None and var_name_uplift in summary_stats_display['Variation'].values)
                if num_vars_to_plot_bin > 0:
                    cols_diff_plots_bin = st.columns(min(num_vars_to_plot_bin, 3)); col_idx_bin = 0
                    for var_name in ordered_vars_for_bayes_display: # Use ordered list
                        if var_name == control_name_bayes_bin or var_name not in bayesian_results_to_display: continue
                        b_res = bayesian_results_to_display[var_name]
                        if b_res.get('diff_samples_vs_control') is None: continue
                        
                        with cols_diff_plots_bin[col_idx_bin % min(num_vars_to_plot_bin, 3)]:
                            fig_diff_bin, ax_diff_bin = plt.subplots(); 
                            ax_diff_bin.hist(b_res['diff_samples_vs_control'], bins=50, density=True, alpha=0.6, label=f"{var_name} - {control_name_bayes_bin}")
                            ax_diff_bin.axvline(0, color='grey', linestyle='--'); 
                            ax_diff_bin.axvline(b_res.get('expected_uplift_abs',0), color='red', linestyle=':', label=f"Mean Diff: {b_res.get('expected_uplift_abs',0)*100:.2f}%")
                            ax_diff_bin.set_title(f"Uplift: {var_name} vs {control_name_bayes_bin}"); 
                            ax_diff_bin.set_xlabel("Difference in CR"); ax_diff_bin.set_ylabel("Density"); 
                            ax_diff_bin.legend(); st.pyplot(fig_diff_bin); plt.close(fig_diff_bin)
                            col_idx_bin +=1
                st.markdown("""**Interpreting Bayesian Results (Binary - Briefly):** (Full guidance in 'Bayesian Analysis Guidelines' section)
                - **Posterior Mean CR:** Average CR after data. - **CrI for CR:** Range for true CR. - **P(Better > Control):** Probability variation's true CR is higher. - **Expected Uplift:** Average expected improvement. - **CrI for Uplift:** Range for true uplift. If includes 0, 'no difference' is plausible. - **P(Being Best):** Probability variation has highest true CR.""")
            else: st.info("Bayesian analysis results for binary outcomes are not available or could not be computed.")

        elif metric_type_display == 'Continuous':
            st.markdown("---"); st.subheader("Bayesian Analysis Results (Continuous Outcome)")
            bayesian_results_to_display_cont = st.session_state.bayesian_results_continuous_c7
            if bayesian_results_to_display_cont and summary_stats_display is not None:
                st.markdown(f"Using a t-distribution approximation for posteriors. Credible Intervals (CrI) at {100*(1-alpha_display):.0f}% level.")
                bayesian_data_disp_cont = []; control_name_bayes_cont = st.session_state.control_group_name_analysis_c7
                
                ordered_vars_for_bayes_display_cont = summary_stats_display['Variation'].tolist()

                for var_name in ordered_vars_for_bayes_display_cont:
                    if var_name not in bayesian_results_to_display_cont: continue
                    b_res_cont = bayesian_results_to_display_cont[var_name]
                    
                    # Handle potential NaN values from Bayesian calculation for display
                    post_mean_disp = f"{b_res_cont.get('posterior_mean', np.nan):.3f}" if pd.notna(b_res_cont.get('posterior_mean')) else "N/A"
                    mean_ci_disp = f"[{b_res_cont.get('mean_ci_low', np.nan):.3f}, {b_res_cont.get('mean_ci_high', np.nan):.3f}]" if pd.notna(b_res_cont.get('mean_ci_low')) else "N/A"
                    prob_better_disp = f"{b_res_cont.get('prob_better_than_control', np.nan)*100:.2f}%" if pd.notna(b_res_cont.get('prob_better_than_control')) else ("N/A (Control)" if var_name == control_name_bayes_cont else "N/A")
                    exp_diff_disp = f"{b_res_cont.get('expected_diff_abs', np.nan):.3f}" if pd.notna(b_res_cont.get('expected_diff_abs')) else ("N/A (Control)" if var_name == control_name_bayes_cont else "N/A")
                    diff_ci_disp = f"[{b_res_cont.get('diff_ci_low', np.nan):.3f}, {b_res_cont.get('diff_ci_high', np.nan):.3f}]" if pd.notna(b_res_cont.get('diff_ci_low')) else ("N/A (Control)" if var_name == control_name_bayes_cont else "N/A")
                    prob_best_disp = f"{b_res_cont.get('prob_best', np.nan)*100:.2f}%" if pd.notna(b_res_cont.get('prob_best')) else "N/A"

                    bayesian_data_disp_cont.append({
                        "Variation": var_name,
                        "Posterior Mean": post_mean_disp,
                        f"{100*(1-alpha_display):.0f}% CrI for Mean": mean_ci_disp,
                        "P(Better > Control) (%)": prob_better_disp,
                        "Expected Diff. (abs)": exp_diff_disp,
                        f"{100*(1-alpha_display):.0f}% CrI for Diff. (abs)": diff_ci_disp,
                        "P(Being Best) (%)": prob_best_disp
                    })
                if bayesian_data_disp_cont:
                    bayesian_df_cont = pd.DataFrame(bayesian_data_disp_cont)
                    st.markdown(bayesian_df_cont.to_html(escape=False), unsafe_allow_html=True) # Display table

                # Plots for Bayesian Continuous
                st.markdown("##### Posterior Distributions for Means (Continuous)")
                fig_mean_cont, ax_mean_cont = plt.subplots()
                max_density_mean_cont = 0
                # Determine a suitable x-range for mean plots
                all_mean_samples = np.concatenate([res.get('samples', np.array([])) for res in bayesian_results_to_display_cont.values() if res.get('samples') is not None and res.get('samples').size > 0])
                if all_mean_samples.size > 0:
                    x_min_mean_cont, x_max_mean_cont = np.nanmin(all_mean_samples), np.nanmax(all_mean_samples)
                    padding_mean_cont = (x_max_mean_cont - x_min_mean_cont) * 0.1 if (x_max_mean_cont - x_min_mean_cont) > 0 else 1.0
                    x_range_mean_cont = np.linspace(x_min_mean_cont - padding_mean_cont, x_max_mean_cont + padding_mean_cont, 300)

                    for var_name in ordered_vars_for_bayes_display_cont:
                        if var_name not in bayesian_results_to_display_cont: continue
                        b_res_cont = bayesian_results_to_display_cont[var_name]
                        samples = b_res_cont.get('samples')
                        if samples is not None and samples.size > 0 and not np.all(np.isnan(samples)):
                            # Kernel Density Estimation for smooth plot
                            from scipy.stats import gaussian_kde
                            kde = gaussian_kde(samples[~np.isnan(samples)])
                            pdf_values = kde(x_range_mean_cont)
                            ax_mean_cont.plot(x_range_mean_cont, pdf_values, label=f"{var_name}")
                            ax_mean_cont.fill_between(x_range_mean_cont, pdf_values, alpha=0.2)
                            max_density_mean_cont = max(max_density_mean_cont, np.nanmax(pdf_values))
                    
                    if max_density_mean_cont > 0: ax_mean_cont.set_ylim(0, max_density_mean_cont * 1.1)
                    else: ax_mean_cont.set_ylim(0,1) # Default if no valid density
                    ax_mean_cont.set_title("Posterior Distributions of Means"); ax_mean_cont.set_xlabel("Mean Value"); 
                    ax_mean_cont.set_ylabel("Density"); ax_mean_cont.legend(); st.pyplot(fig_mean_cont); plt.close(fig_mean_cont)
                else:
                    st.caption("Not enough valid sample data to plot posterior distributions of means.")


                st.markdown("##### Posterior Distribution of Difference (Variation Mean - Control Mean)")
                num_vars_to_plot_cont_diff = sum(1 for var_name_diff in bayesian_results_to_display_cont if var_name_diff != control_name_bayes_cont and bayesian_results_to_display_cont[var_name_diff].get('diff_samples_vs_control') is not None and not np.all(np.isnan(bayesian_results_to_display_cont[var_name_diff].get('diff_samples_vs_control'))))

                if num_vars_to_plot_cont_diff > 0:
                    cols_diff_plots_cont = st.columns(min(num_vars_to_plot_cont_diff, 2)) # Max 2 columns for these plots
                    col_idx_cont_diff = 0
                    for var_name in ordered_vars_for_bayes_display_cont:
                        if var_name == control_name_bayes_cont or var_name not in bayesian_results_to_display_cont: continue
                        b_res_cont = bayesian_results_to_display_cont[var_name]
                        diff_samples = b_res_cont.get('diff_samples_vs_control')
                        if diff_samples is None or diff_samples.size == 0 or np.all(np.isnan(diff_samples)): continue
                        
                        with cols_diff_plots_cont[col_idx_cont_diff % min(num_vars_to_plot_cont_diff, 2)]:
                            fig_diff_cont, ax_diff_cont = plt.subplots(); 
                            ax_diff_cont.hist(diff_samples[~np.isnan(diff_samples)], bins=50, density=True, alpha=0.6, label=f"{var_name} - {control_name_bayes_cont}")
                            ax_diff_cont.axvline(0, color='grey', linestyle='--'); 
                            expected_diff_val = b_res_cont.get('expected_diff_abs', np.nan)
                            if pd.notna(expected_diff_val):
                                ax_diff_cont.axvline(expected_diff_val, color='red', linestyle=':', label=f"Mean Diff: {expected_diff_val:.3f}")
                            ax_diff_cont.set_title(f"Diff: {var_name} vs {control_name_bayes_cont}"); 
                            ax_diff_cont.set_xlabel("Difference in Mean"); ax_diff_cont.set_ylabel("Density"); 
                            ax_diff_cont.legend(); st.pyplot(fig_diff_cont); plt.close(fig_diff_cont)
                            col_idx_cont_diff +=1
                st.markdown("""**Interpreting Bayesian Results (Continuous - Briefly):** (Full guidance in 'Bayesian Analysis Guidelines' section)
                - **Posterior Mean:** Average value of the metric after data. - **CrI for Mean:** Range for true mean. - **P(Better > Control):** Probability variation's true mean is higher. - **Expected Difference:** Average expected difference from control. - **CrI for Difference:** Range for true difference. If includes 0, 'no difference' is plausible. - **P(Being Best):** Probability variation has highest true mean.""")

            else: st.info("Bayesian analysis results for continuous outcomes are not available or could not be computed.")
    
    st.markdown("---")
    st.info("Segmentation analysis (Cycle 8) and detailed interpretation guidance (Cycle 9) are planned for future updates!")


def show_interpret_results_page():
    # Page for interpreting results and decision guidance.
    st.header("Interpreting Results & Detailed Decision Guidance 🧐")
    st.write("This section will guide you on how to integrate statistical findings with practical significance and business context to make informed decisions.")
    st.info("Coming in Cycle 9: Understanding statistical vs. practical significance, decision frameworks, and next steps after an A/B test!")

def show_faq_page():
    # Displays FAQ on common misinterpretations.
    st.header("FAQ on Common Misinterpretations ❓")
    st.markdown("This section addresses some common questions and misinterpretations that arise when looking at A/B test results.")
    faqs = {
        "Q: My p-value is 0.06. Is my test almost significant? Should I run it longer?": {
            "answer": "Not exactly. A p-value of 0.06 (if your alpha is 0.05) means that if the null hypothesis (no difference) were true, you'd see a result as extreme as, or more extreme than, what you observed 6% of the time due to random chance. It doesn't mean it's 'almost' significant in a graded sense. \n\n**Regarding running it longer:** If you determined your sample size correctly *before* starting the test to achieve adequate power, extending the test just to 'get' a p-value below 0.05 can inflate your Type I error rate (false positives). This is a form of p-hacking. Decisions to extend should be based on pre-defined rules or sequential testing methodologies, not on peeking at p-values.",
            "example": "Think of it like a high jump. If the bar is at 2.00m (your alpha=0.05 significance level) and you jump 1.98m (p=0.06), you haven't cleared the bar. Simply trying again and again hoping to randomly clear it changes the nature of the competition."
        },
        "Q: If a test isn't statistically significant, does it mean there's no difference between the variations?": {
            "answer": "No, not necessarily. A non-significant result means you **failed to find sufficient evidence to reject the null hypothesis** (the hypothesis of no difference). It doesn't prove the null hypothesis is true. There could be a real difference, but your test might have been underpowered (too small sample size to detect it), or the true difference might be smaller than your MDE.",
            "example": "Imagine looking for a very small, specific type of fish in a large lake with a small net. If you don't catch it, it doesn't mean the fish isn't in the lake; your net might have been too small or you might not have fished long enough in the right spot."
        },
        "Q: My A/B test showed Variation B was significantly better, but when I launched it, performance didn't improve or even got worse. Why?": {
            "answer": "This can be frustrating and can happen for several reasons:\n1.  **Regression to the Mean:** Your test might have caught Variation B on a random 'hot streak'. Over a longer period with more users, its performance might naturally regress towards its true (less impressive) mean.\n2.  **Novelty Effect:** Users might have initially reacted positively to something new and different, but this effect wears off over time.\n3.  **Segmentation Issues:** The overall 'win' might have been driven by a specific segment of users in your test. If the live traffic composition is different, the effect might dilute or reverse.\n4.  **External Factors:** Were there any marketing campaigns, holidays, or site issues during the test that might not be present post-launch (or vice-versa)?\n5.  **Type I Error (False Positive):** Even with an alpha of 5%, there's a 1 in 20 chance that your significant result was purely due to random chance.\n6.  **Implementation Issues:** Was Variation B implemented *exactly* the same way in the live environment as it was in the test? Small differences can have big impacts.",
            "example": "A new song might shoot up the charts due to initial hype (novelty) but then fade as long-term appeal isn't as strong."
        },
        "Q: Can I combine results from two separate A/B tests that tested the same feature but at different times?": {
            "answer": "Generally, this is not recommended and should be approached with extreme caution. User behavior can change over time due to seasonality, different traffic sources, product updates, or other external factors. Combining data from periods with potentially different underlying baselines or user populations can lead to misleading conclusions (Simpson's Paradox is a risk here).",
            "example": "Trying to combine lemonade sales data from a hot summer week with data from a cool autumn week to determine the effectiveness of a new sign. The weather (an external factor) is a major confounder."
        },
        "Q: Is a 200% lift with a small sample size (e.g., 1 conversion vs. 3 conversions) reliable?": {
            "answer": "Not necessarily, and often not. While the percentage lift might look huge, the absolute numbers are tiny. With such small numbers, results are highly susceptible to random chance. A single extra conversion can create a massive percentage lift. Statistical significance tests (and Bayesian credible intervals) will likely show very high uncertainty (e.g., a very wide confidence/credible interval for the difference). Always consider the absolute numbers, sample size, and the uncertainty metrics, not just the headline lift percentage.",
            "example": "If one person buys a $100 item in Control (1 user, 1 conversion) and three people buy in Variation (3 users, 3 conversions), that's a 200% lift in users who converted if you only count those specific users. But it's based on tiny numbers. What if the next user in Control also buys? The lift changes dramatically."
        },
        "Q: My Bayesian test shows P(B>A) = 92%. Does this mean there's a 92% chance my decision to launch B is correct?": {
            "answer": "No, not directly. P(B>A) = 92% (or 'Probability B is Better than A') means there's a 92% probability that the *true underlying parameter* (e.g., true conversion rate) of Variation B is greater than that of Variation A, given your data and your prior beliefs. \n\nWhile this is strong evidence in favor of B, the 'correctness' of a launch decision also involves considering:\n* **Magnitude of the difference:** Is the expected uplift practically significant? (See the Credible Interval for Uplift).\n* **Costs and risks:** What are the costs of implementation? What are the risks if B is actually slightly worse (there's still an 8% chance A is better or they are the same)?\n* **Business goals:** How does this uplift align with overall objectives?",
            "example": "If a weather forecast says there's a 92% chance of rain, it's a high probability. But your decision to cancel an outdoor picnic (the 'launch' decision) might also depend on how important the picnic is, whether you have a backup indoor venue, etc."
        },
        "Q: What if my control group's conversion rate in the A/B test is very different from its historical average?": {
            "answer": "This is a good flag to investigate before drawing firm conclusions from the test. It could indicate:\n1.  **Seasonality/Trends:** User behavior naturally changes over time.\n2.  **Different Traffic Mix:** The users in your test might be from different sources or demographics than usual.\n3.  **Instrumentation Error:** Are your tracking and data collection for the test set up correctly?\n4.  **Actual Change in Baseline:** Something fundamental might have shifted in overall user behavior recently.\nIt's important to understand why this discrepancy exists, as it affects the context and generalizability of your test results.",
            "example": "If your ice cream shop's historical average daily sales are 100 cones, but during a one-week test of a new flavor, the control (regular vanilla) only sells 50 cones per day, you'd want to understand why (e.g., was it unusually cold that week?) before concluding much about the new flavor's relative performance."
        },
        "Q: The A/B/n test shows Variation C is best overall based on the highest metric. Can I just ignore A and B and compare C only to the historical baseline to report the lift?": {
            "answer": "Not always safely. The direct comparison in your A/B/n test is between C and the *concurrent control (A)* run during the same period, with the same users, under the same conditions. This is the most reliable comparison. \n\nIf you compare C to a historical baseline: \n* You reintroduce all the problems of time-based differences (seasonality, traffic mix, etc.) that A/B testing is designed to control for.\n* The 'lift' might be artificially inflated or deflated due to these temporal factors, not just the impact of C. \nAlways prioritize the lift calculated against the concurrent control from the test itself.",
            "example": "In a race, even if a runner (C) finishes first, their performance improvement is best measured against their own previous times in similar conditions or against other runners (A, B) in that *same race*, not against a historical world record set years ago under different circumstances."
        }
    }
    for question, details in faqs.items():
        with st.expander(question):
            st.markdown(f"**A:** {details['answer']}")
            if "example" in details: st.markdown(f"**Analogy / Example:** {details['example']}")
    st.markdown("---")
    st.info("Content for this section will be reviewed and expanded as needed.")

def show_bayesian_guidelines_page():
    # Page for Bayesian analysis guidelines.
    st.header("Bayesian Analysis Guidelines 🧠")
    st.markdown("This section provides a guide to understanding and interpreting Bayesian A/B test results, complementing the direct outputs from the 'Analyze Results' page.")
    
    st.subheader("Core Bayesian Concepts in this App")
    st.markdown("""
    * **Prior Distribution:** Represents your beliefs about a parameter *before* seeing the test data.
        * For **binary outcomes (Conversion Rates)**, this app uses a `Beta(1,1)` prior by default. This is an "uninformative" or "flat" prior, meaning it assumes all conversion rates between 0% and 100% are equally likely before the test.
        * For **continuous outcomes (Means)**, the app uses a method that approximates the posterior distribution of the mean using a t-distribution derived from your sample data (sample mean, sample standard deviation, sample size). This approach implicitly uses non-informative priors for the underlying true mean and variance.
    * **Likelihood:** This is how your collected A/B test data informs the model about the parameters (e.g., conversion rate, mean value).
    * **Posterior Distribution:** Your updated belief about the parameter *after* combining the prior and the data (via the likelihood). This is the key output of Bayesian analysis.
        * For binary outcomes: Beta Prior + Binomial Likelihood = Beta Posterior.
        * For continuous outcomes: The posterior for the mean is approximated by a t-distribution.
    """)

    st.subheader("Interpreting Key Bayesian Outputs")
    st.markdown("**For Binary Outcomes (e.g., Conversion Rates):**")
    st.markdown("""
    * **Posterior Distribution Plot (CRs):** Visualizes the range of plausible values for the true conversion rate of each variation after seeing the data. Wider distributions mean more uncertainty.
    * **Posterior Mean CR & Credible Interval (CrI) for CR:**
        * *Posterior Mean CR:* The average conversion rate for a variation, based on the posterior distribution.
        * *CrI for CR:* We are X% confident (e.g., 95%) that the true conversion rate for this variation lies within this interval. Unlike frequentist Confidence Intervals, you *can* make direct probability statements about the parameter being in the interval.
    * **P(Variation > Control) (Probability of Being Better):** The probability that the variation's true underlying conversion rate is strictly greater than the control's true conversion rate. A high value (e.g., >95%) gives strong confidence the variation is an improvement.
        * *Important Note:* Even if this probability is high, also check the **Credible Interval for Uplift**. If that interval is very wide or very close to zero, the magnitude of the improvement might be small or uncertain, even if you're confident it's positive.
    * **Expected Uplift (Absolute %):** The average absolute improvement (or decline) in conversion rate you might expect from choosing a variation over the control, based on the posterior distributions.
    * **Credible Interval (CrI) for Uplift (Absolute %):** We are X% confident that the true absolute uplift (Variation CR - Control CR) lies within this interval.
        * If this interval includes 0, then 'no difference' or even a negative impact are plausible outcomes, even if P(Variation > Control) is high.
        * If the entire interval is above 0, you have strong evidence of a positive uplift.
        * If the entire interval is below 0, you have strong evidence of a negative impact.
    * **P(Being Best):** In an A/B/n test, this is the probability that a specific variation has the highest true conversion rate among all tested variations. Useful for selecting a winner when multiple variations are present.
    """)

    st.markdown("**For Continuous Outcomes (e.g., Average Order Value, Time on Page):**")
    st.markdown("""
    * **Posterior Distribution Plot (Means):** Visualizes the plausible values for the true mean of the metric for each variation, based on the t-distribution approximation of the posterior.
    * **Posterior Mean & Credible Interval (CrI) for Mean:**
        * *Posterior Mean:* The average value of the metric for a variation, based on its posterior distribution (will be very close to the sample mean).
        * *CrI for Mean:* We are X% confident that the true mean for this variation lies within this interval.
    * **P(Variation > Control) (Probability of Being Better):** The probability that the variation's true underlying mean is strictly greater than the control's true mean.
    * **Expected Difference (Absolute):** The average absolute difference (Variation Mean - Control Mean) you might expect.
    * **Credible Interval (CrI) for Difference (Absolute):** We are X% confident that the true absolute difference in means lies within this interval. Interpretation is similar to the CrI for Uplift in binary outcomes:
        * If it includes 0, 'no difference' is plausible.
        * If entirely above 0, evidence of a positive difference.
        * If entirely below 0, evidence of a negative difference.
    * **P(Being Best):** The probability that a specific variation has the highest true mean among all tested variations.
    """)
    
    st.subheader("Advantages of Bayesian A/B Testing")
    st.markdown("""
    * **Intuitive Results:** Probabilities like "P(Variation B is better than A) is 92%" are often more aligned with how business stakeholders think about risk and decisions.
    * **Good for Smaller Samples (with informative priors):** While this app uses uninformative priors by default, Bayesian methods can formally incorporate prior knowledge, which can be beneficial with limited data (though this feature is not yet in this app).
    * **Probability Statements about Hypotheses:** Bayesian analysis allows you to make direct probability statements about your hypotheses (e.g., "There is an X% chance that variation B's CR is above Y%").
    * **More Robust to "Peeking":** While not a license for continuous stopping, the interpretation of Bayesian posteriors is less affected by optional stopping compared to frequentist p-values. The posterior is simply your current state of belief given the data seen so far.
    * **Decision Making Frameworks:** Bayesian results (like expected loss/uplift) can directly feed into decision theory frameworks (a more advanced topic).
    """)
    st.info("This section will be further expanded with more detailed examples, discussion on the choice of priors for advanced users, and interpretation nuances for different scenarios.")


def show_roadmap_page():
    # Displays the roadmap and future features.
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

# Ensure the selected page function is callable
if callable(PAGES.get(selection)):
    page_function = PAGES[selection]
    page_function()
else:
    st.error("Selected page could not be loaded.")


st.sidebar.markdown("---")
st.sidebar.info("A/B Testing Guide & Analyzer | V0.7.0 (Cycle 7 - Bayesian Continuous Part 1)")
�
