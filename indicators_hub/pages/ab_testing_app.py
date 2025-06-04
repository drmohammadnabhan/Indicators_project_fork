import streamlit as st
import numpy as np
from scipy.stats import norm, ttest_ind, t as t_dist
import math
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from scipy.stats import beta as beta_dist, gaussian_kde
import matplotlib.pyplot as plt
from itertools import product # For creating segment combinations

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
    if summary_stats is None or summary_stats.empty:
        return None, "Summary statistics are empty or None for Bayesian binary analysis."
    if 'Variation' not in summary_stats.columns:
        original_var_col_name = summary_stats.columns[0] 
        if original_var_col_name != 'Variation': summary_stats = summary_stats.rename(columns={original_var_col_name: 'Variation'})
    
    for index, row in summary_stats.iterrows():
        var_name = row['Variation']
        users = int(row['Users']) if pd.notna(row['Users']) else 0
        conversions = int(row['Conversions']) if pd.notna(row['Conversions']) else 0

        alpha_post = prior_alpha + conversions; beta_post = prior_beta + (users - conversions)
        posterior_params[var_name] = {'alpha': alpha_post, 'beta': beta_post}
        
        if users > 0 and alpha_post > 0 and beta_post > 0 : 
            samples = beta_dist.rvs(alpha_post, beta_post, size=n_samples)
            results[var_name] = {
                'samples': samples, 'mean_cr': np.mean(samples), 'median_cr': np.median(samples), 
                'cr_ci_low': beta_dist.ppf((1-ci_level)/2, alpha_post, beta_post), 
                'cr_ci_high': beta_dist.ppf(1-(1-ci_level)/2, alpha_post, beta_post), 
                'alpha_post': alpha_post, 'beta_post': beta_post, 'diff_samples_vs_control': None 
            }
        else: 
             results[var_name] = {
                'samples': np.full(n_samples, np.nan), 'mean_cr': np.nan, 'median_cr': np.nan,
                'cr_ci_low': np.nan, 'cr_ci_high': np.nan, 'alpha_post': alpha_post, 'beta_post': beta_post,
                'diff_samples_vs_control': np.full(n_samples, np.nan), 'prob_better_than_control': np.nan,
                'uplift_ci_low': np.nan, 'uplift_ci_high': np.nan, 'expected_uplift_abs': np.nan, 'prob_best': np.nan
            }
    
    if control_group_name not in results or (results[control_group_name] and np.all(np.isnan(results[control_group_name]['samples']))): 
        for var_name_key in results: # Use var_name_key to avoid conflict
            results[var_name_key].update({'prob_better_than_control': np.nan, 'uplift_ci_low': np.nan, 
                                      'uplift_ci_high': np.nan, 'expected_uplift_abs': np.nan, 'prob_best': np.nan})
        err_msg = (f"Control group '{control_group_name}' not found in results for Bayesian analysis." 
                   if control_group_name not in results else 
                   f"Control group '{control_group_name}' has insufficient data for Bayesian comparison.")
        # st.warning(err_msg) # Warning moved to calling function for context
        return results, err_msg

    control_samples = results[control_group_name]['samples']
    
    for var_name, data in results.items():
        if var_name == control_group_name or np.all(np.isnan(data['samples'])): 
            data.update({'prob_better_than_control': None if var_name == control_group_name else np.nan, 
                         'uplift_ci_low': None if var_name == control_group_name else np.nan, 
                         'uplift_ci_high': None if var_name == control_group_name else np.nan, 
                         'expected_uplift_abs': None if var_name == control_group_name else np.nan})
            if np.all(np.isnan(data['samples'])) and var_name != control_group_name:
                 data['diff_samples_vs_control'] = np.full(n_samples, np.nan)
            continue

        var_samples = data['samples']; diff_samples = var_samples - control_samples
        data['diff_samples_vs_control'] = diff_samples
        valid_diff = diff_samples[~np.isnan(diff_samples)]
        if valid_diff.size > 0:
            data['prob_better_than_control'] = np.mean(valid_diff > 0)
            data['uplift_ci_low'] = np.percentile(valid_diff, (1-ci_level)/2 * 100)
            data['uplift_ci_high'] = np.percentile(valid_diff, (1-(1-ci_level)/2) * 100)
            data['expected_uplift_abs'] = np.mean(valid_diff)
        else:
            data.update({'prob_better_than_control': np.nan, 'uplift_ci_low': np.nan, 
                         'uplift_ci_high': np.nan, 'expected_uplift_abs': np.nan})
            
    all_var_names = summary_stats['Variation'].tolist()
    valid_ordered_vars = [name for name in all_var_names if name in results and not np.all(np.isnan(results[name]['samples']))]

    if not valid_ordered_vars:
        for var_name_key in all_var_names: 
            if var_name_key in results: results[var_name_key]['prob_best'] = np.nan
            else: results[var_name_key] = {'prob_best': np.nan}
        return results, "No variations with valid data for P(Best) calculation."
    
    all_samples_matrix = np.array([results[var]['samples'] for var in valid_ordered_vars])
    best_variation_counts = np.zeros(len(all_var_names)) 

    if all_samples_matrix.ndim == 2 and all_samples_matrix.shape[0] > 0 and all_samples_matrix.shape[1] == n_samples:
        valid_iterations = 0
        for i in range(n_samples):
            current_iter_samples = all_samples_matrix[:, i]
            if not np.all(np.isnan(current_iter_samples)):
                valid_iterations +=1
                best_idx_in_temp_matrix = np.nanargmax(current_iter_samples)
                best_var_name_this_iter = valid_ordered_vars[best_idx_in_temp_matrix]
                original_idx_for_counts = all_var_names.index(best_var_name_this_iter)
                best_variation_counts[original_idx_for_counts] += 1
        
        if valid_iterations > 0:
            prob_best = best_variation_counts / valid_iterations
        else:
            prob_best = np.full(len(all_var_names), np.nan)

        for i, var_name_key in enumerate(all_var_names):
            if var_name_key in results: 
                results[var_name_key]['prob_best'] = prob_best[i] if var_name_key in valid_ordered_vars else np.nan
            else: 
                 results[var_name_key] = {'prob_best': np.nan} 
    else: 
        for var_name_key in all_var_names:
            if var_name_key in results:
                is_single_valid = (len(valid_ordered_vars) == 1 and var_name_key == valid_ordered_vars[0])
                results[var_name_key]['prob_best'] = 1.0 if is_single_valid else (0.0 if len(valid_ordered_vars) > 1 else np.nan)
            else: 
                results[var_name_key] = {'prob_best': np.nan}
                
    return results, None

def run_bayesian_continuous_analysis(summary_stats, control_group_name, n_samples=10000, ci_level=0.95):
    # Performs Bayesian analysis for continuous outcomes.
    results = {}
    if summary_stats is None or summary_stats.empty:
        return None, "Summary statistics are empty or None for Bayesian continuous analysis."
    
    if 'Variation' not in summary_stats.columns:
        return None, "Summary statistics missing 'Variation' column."
    if not {'Users', 'Mean_Value', 'Std_Dev'}.issubset(summary_stats.columns):
        return None, "Summary statistics missing one or more required columns: 'Users', 'Mean_Value', 'Std_Dev'."

    for index, row in summary_stats.iterrows():
        var_name = row['Variation']
        n = int(row['Users']) if pd.notna(row['Users']) else 0
        sample_mean = row['Mean_Value'] if pd.notna(row['Mean_Value']) else np.nan
        sample_std_dev = row['Std_Dev'] if pd.notna(row['Std_Dev']) else np.nan
        
        current_samples = np.array([])
        posterior_mean_val = np.nan
        ci_low_val = np.nan
        ci_high_val = np.nan

        if n <= 1 or pd.isna(sample_mean) or pd.isna(sample_std_dev) or (sample_std_dev == 0 and n <=1) : 
            current_samples = np.full(n_samples, sample_mean if n == 1 and pd.notna(sample_mean) else np.nan) 
            posterior_mean_val = sample_mean if n == 1 and pd.notna(sample_mean) else np.nan
        elif sample_std_dev == 0 and n > 1: 
            current_samples = np.full(n_samples, sample_mean)
            posterior_mean_val = sample_mean
            ci_low_val = sample_mean
            ci_high_val = sample_mean
        else: 
            df = n - 1
            std_err = sample_std_dev / np.sqrt(n)
            t_samples = t_dist.rvs(df, size=n_samples)
            current_samples = sample_mean + (std_err * t_samples)
            posterior_mean_val = np.mean(current_samples)
            ci_low_val = np.percentile(current_samples, (1 - ci_level) / 2 * 100)
            ci_high_val = np.percentile(current_samples, (1 - (1 - ci_level) / 2) * 100)

        results[var_name] = {
            'samples': current_samples,
            'posterior_mean': posterior_mean_val,
            'mean_ci_low': ci_low_val,
            'mean_ci_high': ci_high_val,
            'diff_samples_vs_control': None 
        }

    if control_group_name not in results or (results[control_group_name] and np.all(np.isnan(results[control_group_name]['samples']))):
        for var_name_key in results: 
            results[var_name_key].update({'prob_better_than_control': np.nan, 'diff_ci_low': np.nan, 
                                      'diff_ci_high': np.nan, 'expected_diff_abs': np.nan, 'prob_best': np.nan})
        err_msg = (f"Control group '{control_group_name}' not found in results for Bayesian analysis." 
                   if control_group_name not in results else 
                   f"Control group '{control_group_name}' has insufficient data for Bayesian comparison.")
        return results, err_msg

    control_samples = results[control_group_name]['samples']
    
    for var_name, data in results.items():
        if var_name == control_group_name or np.all(np.isnan(data['samples'])):
            data.update({'prob_better_than_control': None if var_name == control_group_name else np.nan, 
                         'diff_ci_low': None if var_name == control_group_name else np.nan, 
                         'diff_ci_high': None if var_name == control_group_name else np.nan, 
                         'expected_diff_abs': None if var_name == control_group_name else np.nan})
            if np.all(np.isnan(data['samples'])) and var_name != control_group_name:
                 data['diff_samples_vs_control'] = np.full(n_samples, np.nan)
            continue

        var_samples = data['samples']
        diff_samples = var_samples - control_samples
        data['diff_samples_vs_control'] = diff_samples
        
        valid_diff_samples = diff_samples[~np.isnan(diff_samples)]
        if valid_diff_samples.size > 0:
            data['prob_better_than_control'] = np.mean(valid_diff_samples > 0)
            data['diff_ci_low'] = np.percentile(valid_diff_samples, (1 - ci_level) / 2 * 100)
            data['diff_ci_high'] = np.percentile(valid_diff_samples, (1 - (1 - ci_level) / 2) * 100)
            data['expected_diff_abs'] = np.mean(valid_diff_samples)
        else: 
            data.update({'prob_better_than_control': np.nan, 'diff_ci_low': np.nan, 
                         'diff_ci_high': np.nan, 'expected_diff_abs': np.nan})

    all_var_names_cont = summary_stats['Variation'].tolist()
    valid_var_names_for_pbest = [name for name in all_var_names_cont if name in results and not np.all(np.isnan(results[name]['samples']))]

    if not valid_var_names_for_pbest:
        for var_name_key in all_var_names_cont: 
            if var_name_key in results: results[var_name_key]['prob_best'] = np.nan
            else: results[var_name_key] = {'prob_best': np.nan} 
        return results, "No variations with valid data found for P(Best) calculation in continuous Bayesian analysis."

    all_samples_matrix_cont = np.array([results[var]['samples'] for var in valid_var_names_for_pbest])
    best_variation_counts_cont = np.zeros(len(all_var_names_cont))

    if all_samples_matrix_cont.ndim == 2 and all_samples_matrix_cont.shape[0] > 0 and all_samples_matrix_cont.shape[1] == n_samples:
        valid_iterations_for_pbest = 0
        for i in range(n_samples): 
            current_iter_samples = all_samples_matrix_cont[:, i]
            if not np.all(np.isnan(current_iter_samples)): 
                valid_iterations_for_pbest += 1
                best_idx_in_temp_matrix = np.nanargmax(current_iter_samples)
                best_var_name_this_iter = valid_var_names_for_pbest[best_idx_in_temp_matrix]
                original_idx_for_counts = all_var_names_cont.index(best_var_name_this_iter)
                best_variation_counts_cont[original_idx_for_counts] += 1
        
        if valid_iterations_for_pbest > 0:
            prob_best_cont = best_variation_counts_cont / valid_iterations_for_pbest
        else: 
            prob_best_cont = np.full(len(all_var_names_cont), np.nan)

        for i, var_name_key in enumerate(all_var_names_cont):
            if var_name_key in results:
                results[var_name_key]['prob_best'] = prob_best_cont[i] if var_name_key in valid_var_names_for_pbest else np.nan
            else: 
                results[var_name_key] = {'prob_best': np.nan} 
                
    else: 
        for var_name_key in all_var_names_cont:
            if var_name_key in results:
                is_single_valid_var = (len(valid_var_names_for_pbest) == 1 and var_name_key == valid_var_names_for_pbest[0])
                results[var_name_key]['prob_best'] = 1.0 if is_single_valid_var else (0.0 if len(valid_var_names_for_pbest) > 1 else np.nan)
            else:
                results[var_name_key] = {'prob_best': np.nan}
                
    return results, None


# --- Page Functions ---
def show_introduction_page():
    # Displays the introduction to A/B testing.
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
    * 🛡️ **Reduce Risk:** Test changes on a smaller scale before rolling them out to your entire user base, minimizing the impact of potentially negative changes.
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
    cycle_key_suffix_ds = "_c8_1" 
    metric_type_ss = st.radio("Select your primary metric type for sample size calculation:", ('Binary (e.g., Conversion Rate)', 'Continuous (e.g., Average Order Value)'), key=f"ss_metric_type_radio{cycle_key_suffix_ds}") 
    st.markdown("---")
    if metric_type_ss == 'Binary (e.g., Conversion Rate)':
        st.subheader("Sample Size Calculator (for Binary Outcomes)")
        st.markdown("**Calculator Inputs:**")
        cols_bin = st.columns(2)
        with cols_bin[0]: baseline_cr_percent = st.number_input(label="Baseline Conversion Rate (BCR) (%)", min_value=0.1, max_value=99.9, value=5.0, step=0.1, format="%.1f", help="Current CR of control (e.g., 5% for 5 out of 100).", key=f"ss_bcr{cycle_key_suffix_ds}")
        with cols_bin[1]: mde_abs_percent = st.number_input(label="Minimum Detectable Effect (MDE) - Absolute (%)", min_value=0.1, max_value=50.0, value=1.0, step=0.1, format="%.1f", help="Smallest absolute CR increase to detect (e.g., 1% for 5% to 6%).", key=f"ss_mde{cycle_key_suffix_ds}")
        cols2_bin = st.columns(2)
        with cols2_bin[0]: power_percent_bin = st.slider(label="Statistical Power (1 - \u03B2) (%)", min_value=50, max_value=99, value=80, step=1, format="%d%%", help="Probability of detecting an effect if one exists (typically 80-90%).", key=f"ss_power{cycle_key_suffix_ds}")
        with cols2_bin[1]: alpha_percent_bin = st.slider(label="Significance Level (\u03B1) (%) - Two-sided", min_value=1, max_value=20, value=5, step=1, format="%d%%", help="Risk of false positive (typically 1-5%).", key=f"ss_alpha{cycle_key_suffix_ds}")
        num_variations_ss_bin = st.number_input(label="Number of Variations (including Control)", min_value=2, value=2, step=1, help="Total versions (e.g., Control + 1 Var = 2).", key=f"ss_num_var{cycle_key_suffix_ds}")
        st.markdown("<p style='font-size: smaller; font-style: italic;'>Note: For binary, based on pairwise comparisons vs. control at specified α.</p>", unsafe_allow_html=True)
        if st.button("Calculate Sample Size (Binary)", key=f"ss_calc_button{cycle_key_suffix_ds}_bin"):
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
        with cols_cont[0]: baseline_mean = st.number_input(label="Baseline Mean (Control Group)", value=100.0, step=1.0, format="%.2f", help="Current average value of your metric for the control group.", key=f"ss_mean{cycle_key_suffix_ds}")
        with cols_cont[1]: std_dev = st.number_input(label="Standard Deviation (of the metric)", value=20.0, min_value=0.01, step=0.1, format="%.2f", help="Estimated standard deviation of your continuous metric. Get from historical data if possible. Must be > 0.", key=f"ss_stddev{cycle_key_suffix_ds}") 
        mde_abs_mean = st.number_input(label="Minimum Detectable Effect (MDE) - Absolute Mean Difference", value=5.0,min_value=0.01, step=0.1, format="%.2f", help="Smallest absolute difference in means you want to detect (e.g., $2 increase). Must be > 0.", key=f"ss_mde_mean{cycle_key_suffix_ds}")
        cols2_cont = st.columns(2)
        with cols2_cont[0]: power_percent_cont = st.slider(label="Statistical Power (1 - \u03B2) (%)", min_value=50, max_value=99, value=80, step=1, format="%d%%", help="Typically 80-90%.", key=f"ss_power_cont{cycle_key_suffix_ds}")
        with cols2_cont[1]: alpha_percent_cont = st.slider(label="Significance Level (\u03B1) (%) - Two-sided", min_value=1, max_value=20, value=5, step=1, format="%d%%", help="Typically 1-5%.", key=f"ss_alpha_cont{cycle_key_suffix_ds}")
        num_variations_ss_cont = st.number_input(label="Number of Variations (including Control)", min_value=2, value=2, step=1, help="Total versions.", key=f"ss_num_var_cont{cycle_key_suffix_ds}")
        st.markdown("<p style='font-size: smaller; font-style: italic;'>Note: For continuous, based on pairwise comparisons vs. control at specified α, assuming similar standard deviations across groups.</p>", unsafe_allow_html=True)
        if st.button("Calculate Sample Size (Continuous)", key=f"ss_calc_button{cycle_key_suffix_ds}_cont"):
            power, alpha = power_percent_cont/100.0, alpha_percent_cont/100.0
            sample_size, error_msg = calculate_continuous_sample_size(baseline_mean, std_dev, mde_abs_mean, power, alpha, num_variations_ss_cont)
            if error_msg: st.error(error_msg)
            elif sample_size:
                st.success("Calculation Successful!")
                target_mean_positive = baseline_mean + mde_abs_mean 
                target_mean_negative = baseline_mean - mde_abs_mean
                res_cols = st.columns(2); res_cols[0].metric("Required Sample Size PER Variation", f"{sample_size:,}"); res_cols[1].metric("Total Required Sample Size", f"{(sample_size * num_variations_ss_cont):,}")
                st.markdown(f"**Summary of Inputs Used:**\n- Baseline Mean: `{baseline_mean:.2f}`\n- Estimated Standard Deviation: `{std_dev:.2f}`\n- Absolute MDE (Mean Difference): `{mde_abs_mean:.2f}` (Targeting a mean of approx. `{target_mean_positive:.2f}` or `{target_mean_negative:.2f}` for variations)\n- Statistical Power: `{power_percent_cont}%` (1 - \u03B2)\n- Significance Level (\u03B1): `{alpha_percent_cont}%` (two-sided)\n- Number of Variations: `{num_variations_ss_cont}`.")
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

# --- Refactored Analysis Display Functions ---
def display_frequentist_analysis(df_for_analysis, metric_type, outcome_col, variation_col, control_name, alpha, summary_stats_key_suffix=""):
    """
    Helper function to calculate and display Frequentist analysis results.
    Can be used for overall data or a specific segment.
    Returns the summary statistics DataFrame for potential reuse.
    """
    if df_for_analysis is None or df_for_analysis.empty:
        st.warning(f"No data provided for Frequentist analysis {summary_stats_key_suffix}.")
        return None

    # Calculate summary statistics
    if metric_type == 'Binary':
        if '__outcome_processed__' not in df_for_analysis.columns:
            st.error(f"Internal error: '__outcome_processed__' column missing for binary frequentist analysis {summary_stats_key_suffix}.")
            return None
        summary_stats = df_for_analysis.groupby(variation_col).agg(
            Users=('__outcome_processed__', 'count'),
            Conversions=('__outcome_processed__', 'sum') 
        ).reset_index()
        summary_stats.rename(columns={variation_col: 'Variation'}, inplace=True)
        if summary_stats['Users'].sum() == 0: 
            st.warning(f"No users found in the provided data for binary Frequentist analysis {summary_stats_key_suffix}.")
            return None
        summary_stats['Metric Value (%)'] = (summary_stats['Conversions'] / summary_stats['Users'].replace(0, np.nan) * 100).round(2)
        metric_col_name_display = 'Metric Value (%)'
    else: # Continuous
        df_for_analysis[outcome_col] = pd.to_numeric(df_for_analysis[outcome_col], errors='coerce')
        df_cleaned = df_for_analysis.dropna(subset=[outcome_col])
        if df_cleaned.empty:
            st.warning(f"No valid numeric data in outcome column '{outcome_col}' for continuous Frequentist analysis {summary_stats_key_suffix}.")
            return None
        
        summary_stats = df_cleaned.groupby(variation_col).agg(
            Users=(outcome_col, 'count'),
            Mean_Value=(outcome_col, 'mean'),
            Std_Dev=(outcome_col, 'std'),
            Median_Value=(outcome_col, 'median'),
            Std_Err=(outcome_col, lambda x: x.std(ddof=1) / np.sqrt(x.count()) if x.count() > 0 and pd.notna(x.std(ddof=1)) else np.nan)
        ).reset_index()
        summary_stats.rename(columns={variation_col: 'Variation'}, inplace=True)
        if summary_stats['Users'].sum() == 0:
            st.warning(f"No users found in the provided data for continuous Frequentist analysis {summary_stats_key_suffix}.")
            return None
        for col_to_round in ['Mean_Value', 'Std_Dev', 'Median_Value', 'Std_Err']:
            if col_to_round in summary_stats.columns:
                summary_stats[col_to_round] = summary_stats[col_to_round].round(3)
        metric_col_name_display = 'Mean_Value'

    st.markdown("##### 📊 Descriptive Statistics")
    if metric_type == 'Binary':
        st.dataframe(summary_stats[['Variation', 'Users', 'Conversions', metric_col_name_display]].fillna('N/A (0 Users)'))
    else:
        st.dataframe(summary_stats[['Variation', 'Users', 'Mean_Value', 'Median_Value', 'Std_Dev', 'Std_Err']].fillna('N/A'))

    if metric_type == 'Continuous':
        for index, row in summary_stats.iterrows():
            if pd.notna(row['Std_Dev']) and row['Std_Dev'] == 0 and row['Users'] > 1:
                st.warning(f"⚠️ Variation '{row['Variation']}' has a Standard Deviation of 0. All outcome values are identical. This impacts t-test interpretation.")

    if metric_col_name_display in summary_stats.columns:
        chart_data = summary_stats.set_index('Variation')[metric_col_name_display].fillna(0)
        if not chart_data.empty:
            st.bar_chart(chart_data, y=metric_col_name_display)

    st.markdown(f"##### 📈 Comparison vs. Control ('{control_name}')")
    control_data_rows = summary_stats[summary_stats['Variation'] == control_name]
    if control_data_rows.empty:
        st.error(f"Control group '{control_name}' data missing in this segment/dataset.")
        return summary_stats 

    control_data = control_data_rows.iloc[0]
    comparison_results_freq = []

    if metric_type == 'Binary':
        control_users, control_conversions = control_data['Users'], control_data['Conversions']
        control_metric_val = control_conversions / control_users if control_users > 0 else 0
        for index, row in summary_stats.iterrows():
            var_name, var_users, var_conversions = row['Variation'], row['Users'], row['Conversions']
            if var_name == control_name: continue
            var_metric_val = var_conversions / var_users if var_users > 0 else 0
            p_val_disp, ci_disp, sig_disp, abs_disp, rel_disp = 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
            if control_users > 0 and var_users > 0:
                abs_uplift = var_metric_val - control_metric_val; abs_disp = f"{abs_uplift*100:.2f}"
                rel_disp = f"{(abs_uplift / control_metric_val) * 100:.2f}%" if control_metric_val != 0 else "N/A (Control CR is 0)"
                count, nobs = np.array([var_conversions, control_conversions]), np.array([var_users, control_users])
                if not (np.any(count < 0) or np.any(nobs <= 0) or np.any(count > nobs)):
                    try:
                        _, p_value = proportions_ztest(count, nobs, alternative='two-sided')
                        p_val_disp = f"{p_value:.4f}"
                        sig_bool = p_value < alpha; sig_disp = f"Yes (p={p_value:.4f})" if sig_bool else f"No (p={p_value:.4f})"
                        ci_low, ci_high = confint_proportions_2indep(var_conversions, var_users, control_conversions, control_users, method='wald', alpha=alpha)
                        ci_disp = f"[{ci_low*100:.2f}, {ci_high*100:.2f}]"
                    except Exception as e_prop_z: 
                        p_val_disp, ci_disp, sig_disp = 'Error', 'Error', 'Error'; st.caption(f"Z-test error for {var_name}: {e_prop_z}")
                else: sig_disp = 'N/A (Invalid counts/nobs for Z-test)'
            else: sig_disp = 'N/A (Zero users in control or variation)'
            comparison_results_freq.append({"Variation": var_name, "CR (%)": f"{var_metric_val*100:.2f}", "Abs. Uplift (%)": abs_disp, "Rel. Uplift (%)": rel_disp, "P-value": p_val_disp, f"CI {100*(1-alpha):.0f}% Diff (%)": ci_disp, "Significant?": sig_disp})
    
    elif metric_type == 'Continuous':
        control_mean = control_data['Mean_Value']
        control_group_data_raw = df_for_analysis[df_for_analysis[variation_col] == control_name][outcome_col].dropna()
        control_users_raw = len(control_group_data_raw)

        for index, row in summary_stats.iterrows():
            var_name, var_mean = row['Variation'], row['Mean_Value']
            if var_name == control_name: continue
            
            var_group_data_raw = df_for_analysis[df_for_analysis[variation_col] == var_name][outcome_col].dropna()
            var_users_raw = len(var_group_data_raw)
            p_val_disp, ci_disp, sig_disp, abs_disp, rel_disp = 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
            
            if control_users_raw > 1 and var_users_raw > 1:
                abs_diff_means = var_mean - control_mean if pd.notna(var_mean) and pd.notna(control_mean) else np.nan
                abs_disp = f"{abs_diff_means:.3f}" if pd.notna(abs_diff_means) else "N/A"
                rel_disp = f"{(abs_diff_means / control_mean) * 100:.2f}%" if pd.notna(abs_diff_means) and control_mean != 0 and pd.notna(control_mean) else "N/A"
                
                control_var_raw = control_group_data_raw.var(ddof=1)
                var_var_raw = var_group_data_raw.var(ddof=1)

                if pd.notna(control_var_raw) and pd.notna(var_var_raw) and control_var_raw == 0 and var_var_raw == 0:
                    if control_mean == var_mean: p_value = 1.0
                    else: p_value = 0.0 
                    p_val_disp = f"{p_value:.4f}"
                    sig_bool = p_value < alpha; sig_disp = f"Yes (p={p_value:.4f})" if sig_bool else f"No (p={p_value:.4f})"
                    ci_disp = f"[{abs_diff_means:.3f}, {abs_diff_means:.3f}] (Exact)" if pd.notna(abs_diff_means) else "N/A"
                else:
                    try:
                        t_stat, p_value = ttest_ind(var_group_data_raw, control_group_data_raw, equal_var=False, nan_policy='omit')
                        p_val_disp = f"{p_value:.4f}" if pd.notna(p_value) else "N/A (t-test error)"
                        if pd.notna(p_value):
                            sig_bool = p_value < alpha; sig_disp = f"Yes (p={p_value:.4f})" if sig_bool else f"No (p={p_value:.4f})"
                        
                        N1, N2 = var_users_raw, control_users_raw
                        s1_sq, s2_sq = var_var_raw, control_var_raw
                        
                        if N1 > 0 and N2 > 0 and pd.notna(s1_sq) and pd.notna(s2_sq) and pd.notna(abs_diff_means):
                            pooled_se_diff_sq = (s1_sq / N1) + (s2_sq / N2) if N1 > 0 and N2 > 0 else 0 
                            if pooled_se_diff_sq > 0 : 
                                pooled_se_diff = math.sqrt(pooled_se_diff_sq)
                                df_t_approx = min(N1 - 1, N2 - 1)
                                if df_t_approx > 0:
                                    t_crit = t_dist.ppf(1 - alpha / 2, df=df_t_approx)
                                    ci_low_mean_diff, ci_high_mean_diff = abs_diff_means - t_crit * pooled_se_diff, abs_diff_means + t_crit * pooled_se_diff
                                    ci_disp = f"[{ci_low_mean_diff:.3f}, {ci_high_mean_diff:.3f}]"
                                else: ci_disp = "N/A (df CI <=0)"
                            else: ci_disp = "N/A (SE diff is 0 or invalid)" 
                        else: ci_disp = "N/A (N or var issue for CI)"
                    except Exception as e_ttest: 
                        p_val_disp, ci_disp, sig_disp = 'Error', 'Error', 'Error'; st.caption(f"T-test error for {var_name}: {e_ttest}")
            else: sig_disp = 'N/A (N <= 1 in a group)'
            comparison_results_freq.append({"Variation": var_name, "Mean Value": f"{var_mean:.3f}" if pd.notna(var_mean) else "N/A", "Abs. Diff.": abs_disp, "Rel. Diff. (%)": rel_disp, "P-value": p_val_disp, f"CI {100*(1-alpha):.0f}% Diff.": ci_disp, "Significant?": sig_disp})

    if comparison_results_freq:
        comparison_df_freq = pd.DataFrame(comparison_results_freq)
        st.dataframe(comparison_df_freq)
        for _, row_data in comparison_df_freq.iterrows():
            if "Yes" in str(row_data["Significant?"]): st.caption(f"Frequentist: Diff between **{row_data['Variation']}** & control is significant at {alpha*100:.0f}% level (P-value: {row_data['P-value']}).")
            elif "No" in str(row_data["Significant?"]): st.caption(f"Frequentist: Diff between **{row_data['Variation']}** & control is not significant at {alpha*100:.0f}% level (P-value: {row_data['P-value']}).")
    
    if metric_type == 'Continuous': 
        st.markdown("##### Distribution of Outcomes by Variation (Box Plots)")
        try:
            if not df_for_analysis.empty and variation_col in df_for_analysis.columns and outcome_col in df_for_analysis.columns: 
                
                unique_vars_plot = sorted(df_for_analysis[variation_col].astype(str).unique()) 
                if control_name in unique_vars_plot: 
                    unique_vars_plot.insert(0, unique_vars_plot.pop(unique_vars_plot.index(control_name)))

                boxplot_data = []
                valid_labels = []
                for var_name_plot in unique_vars_plot:
                    data_series = df_for_analysis[df_for_analysis[variation_col].astype(str) == var_name_plot][outcome_col] 
                    if not data_series.empty and data_series.notna().any(): 
                        boxplot_data.append(data_series.dropna()) 
                        valid_labels.append(var_name_plot)
                
                if not boxplot_data or not valid_labels: 
                    st.caption(f"Not enough valid data for one or more variations to display box plots {summary_stats_key_suffix}.")
                else:
                    fig_box, ax_box = plt.subplots(); 
                    ax_box.boxplot(boxplot_data, labels=valid_labels, patch_artist=True) 
                    ax_box.set_title(f"Outcome Distributions: {outcome_col} by Variation"); 
                    ax_box.set_ylabel(outcome_col); 
                    ax_box.set_xlabel("Variation")
                    plt.xticks(rotation=45, ha="right") 
                    plt.tight_layout()
                    st.pyplot(fig_box); plt.close(fig_box)
            else: st.caption(f"Not enough data or columns missing to display box plots {summary_stats_key_suffix}.")
        except Exception as e_plot: st.warning(f"Could not generate box plots {summary_stats_key_suffix}: {e_plot}")
    return summary_stats

# --- NEW: Helper function for Bayesian Binary Display ---
def display_bayesian_binary_results(bayesian_results, summary_stats_for_ordering, control_name, alpha, section_title_prefix="Overall"):
    if not bayesian_results:
        st.info(f"{section_title_prefix} Bayesian analysis results for binary outcomes are not available or could not be computed.")
        return

    st.markdown(f"---"); st.subheader(f"{section_title_prefix} Bayesian Analysis Results (Binary Outcome)")
    st.markdown(f"Using a Beta(1,1) uninformative prior. Credible Intervals (CrI) at {100*(1-alpha):.0f}% level.")
    
    bayesian_data_disp_bin = []
    ordered_vars_for_display = summary_stats_for_ordering['Variation'].tolist() if summary_stats_for_ordering is not None else list(bayesian_results.keys())

    for var_name in ordered_vars_for_display:
        if var_name not in bayesian_results: continue 
        b_res = bayesian_results[var_name]
        prob_better_html = f"<span title=\"Probability that this variation's true conversion rate is higher than the control's. Also consider the Credible Interval for Uplift to understand magnitude and uncertainty.\">{b_res.get('prob_better_than_control',0)*100:.2f}%</span>" if b_res.get('prob_better_than_control') is not None else "N/A (Control)"
        cri_uplift_html = f"<span title=\"The range where the true uplift over control likely lies. If this interval includes 0, 'no difference' or a negative effect are plausible.\">[{b_res.get('uplift_ci_low', 0)*100:.2f}, {b_res.get('uplift_ci_high', 0)*100:.2f}]</span>" if b_res.get('uplift_ci_low') is not None else "N/A (Control)"
        bayesian_data_disp_bin.append({
            "Variation": var_name, "Posterior Mean CR (%)": f"{b_res.get('mean_cr',0)*100:.2f}", 
            f"{100*(1-alpha):.0f}% CrI for CR (%)": f"[{b_res.get('cr_ci_low',0)*100:.2f}, {b_res.get('cr_ci_high',0)*100:.2f}]", 
            "P(Better > Control) (%)": prob_better_html, 
            "Expected Uplift (abs %)": f"{b_res.get('expected_uplift_abs', 0)*100:.2f}" if b_res.get('expected_uplift_abs') is not None else "N/A (Control)", 
            f"{100*(1-alpha):.0f}% CrI for Uplift (abs %)": cri_uplift_html, 
            "P(Being Best) (%)": f"{b_res.get('prob_best',0)*100:.2f}"
        })
    if bayesian_data_disp_bin:
        bayesian_df_bin = pd.DataFrame(bayesian_data_disp_bin); st.markdown(bayesian_df_bin.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Plots
    if summary_stats_for_ordering is not None and 'Metric Value (%)' in summary_stats_for_ordering.columns:
        st.markdown("##### Posterior Distributions for Conversion Rates (Binary)"); fig_cr_bin, ax_cr_bin = plt.subplots()
        # ... (Plotting logic as before, ensure it uses summary_stats_for_ordering for x-axis limits etc.)
        observed_max_cr_for_plot = 0.0
        numeric_crs_bin = pd.to_numeric(summary_stats_for_ordering['Metric Value (%)'], errors='coerce')
        if not numeric_crs_bin.empty and numeric_crs_bin.notna().any(): observed_max_cr_for_plot = numeric_crs_bin.max() / 100.0
        else: observed_max_cr_for_plot = 0.1 
        posterior_max_cr_for_plot_bin = 0.0
        all_posterior_highs_bin = [res.get('cr_ci_high') for res in bayesian_results.values() if res.get('cr_ci_high') is not None]
        if all_posterior_highs_bin: posterior_max_cr_for_plot_bin = max(all_posterior_highs_bin)
        final_x_limit_candidate_bin = max(observed_max_cr_for_plot, posterior_max_cr_for_plot_bin)
        x_cr_plot_limit_bin = min(1.0, final_x_limit_candidate_bin + 0.05) 
        if x_cr_plot_limit_bin <= 0.01: x_cr_plot_limit_bin = 0.1 
        x_cr_range_bin = np.linspace(0, x_cr_plot_limit_bin, 300)
        max_density_bin = 0
        for var_name in ordered_vars_for_display: 
            if var_name not in bayesian_results: continue
            b_res = bayesian_results[var_name]
            alpha_p, beta_p = b_res.get('alpha_post', 1), b_res.get('beta_post', 1)
            if alpha_p > 0 and beta_p > 0 and not (np.isnan(alpha_p) or np.isnan(beta_p)): # Ensure valid params for pdf
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
        num_vars_to_plot_bin = sum(1 for var_name_uplift in bayesian_results if var_name_uplift != control_name and bayesian_results[var_name_uplift].get('diff_samples_vs_control') is not None and not np.all(np.isnan(bayesian_results[var_name_uplift].get('diff_samples_vs_control'))))
        if num_vars_to_plot_bin > 0:
            cols_diff_plots_bin = st.columns(min(num_vars_to_plot_bin, 3)); col_idx_bin = 0
            for var_name in ordered_vars_for_display: 
                if var_name == control_name or var_name not in bayesian_results: continue
                b_res = bayesian_results[var_name]
                diff_samples = b_res.get('diff_samples_vs_control')
                if diff_samples is None or np.all(np.isnan(diff_samples)): continue
                with cols_diff_plots_bin[col_idx_bin % min(num_vars_to_plot_bin, 3)]:
                    fig_diff_bin, ax_diff_bin = plt.subplots(); 
                    ax_diff_bin.hist(diff_samples[~np.isnan(diff_samples)], bins=50, density=True, alpha=0.6, label=f"{var_name} - {control_name}")
                    ax_diff_bin.axvline(0, color='grey', linestyle='--'); 
                    ax_diff_bin.axvline(b_res.get('expected_uplift_abs',0), color='red', linestyle=':', label=f"Mean Diff: {b_res.get('expected_uplift_abs',0)*100:.2f}%")
                    ax_diff_bin.set_title(f"Uplift: {var_name} vs {control_name}"); ax_diff_bin.set_xlabel("Difference in CR"); ax_diff_bin.set_ylabel("Density"); 
                    ax_diff_bin.legend(); st.pyplot(fig_diff_bin); plt.close(fig_diff_bin)
                    col_idx_bin +=1
    st.markdown("""**Interpreting Bayesian Results (Binary - Briefly):** (Full guidance in 'Bayesian Analysis Guidelines' section)
    - **Posterior Mean CR:** Average CR after data. - **CrI for CR:** Range for true CR. - **P(Better > Control):** Probability variation's true CR is higher. - **Expected Uplift:** Average expected improvement. - **CrI for Uplift:** Range for true uplift. If includes 0, 'no difference' is plausible. - **P(Being Best):** Probability variation has highest true CR.""")

# --- NEW: Helper function for Bayesian Continuous Display ---
def display_bayesian_continuous_results(bayesian_results, summary_stats_for_ordering, control_name, alpha, outcome_col_name, section_title_prefix="Overall"):
    if not bayesian_results:
        st.info(f"{section_title_prefix} Bayesian analysis results for continuous outcomes are not available or could not be computed.")
        return

    st.markdown(f"---"); st.subheader(f"{section_title_prefix} Bayesian Analysis Results (Continuous Outcome)")
    st.markdown(f"Using a t-distribution approximation for posteriors. Credible Intervals (CrI) at {100*(1-alpha):.0f}% level.")
    
    bayesian_data_disp_cont = []
    ordered_vars_for_display = summary_stats_for_ordering['Variation'].tolist() if summary_stats_for_ordering is not None else list(bayesian_results.keys())

    for var_name in ordered_vars_for_display:
        if var_name not in bayesian_results: continue
        b_res_cont = bayesian_results[var_name]
        
        tt_posterior_mean = "The average value of the metric for this variation, based on its posterior distribution (typically very close to the sample mean)."
        tt_cri_mean = f"The {100*(1-alpha):.0f}% Credible Interval for the true mean of this variation. We are {100*(1-alpha):.0f}% confident the true mean lies here."
        tt_prob_better = "Probability that this variation's true mean is greater than the control's true mean. Consider with the CrI for Difference."
        tt_exp_diff = "The average expected absolute difference (Variation Mean - Control Mean)."
        tt_cri_diff = f"The {100*(1-alpha):.0f}% Credible Interval for the true absolute difference in means. If it includes 0, 'no difference' is plausible."
        tt_prob_best = "Probability that this variation has the highest true mean among all tested variations."

        post_mean_disp = f"<span title='{tt_posterior_mean}'>{b_res_cont.get('posterior_mean', np.nan):.3f}</span>" if pd.notna(b_res_cont.get('posterior_mean')) else "N/A"
        mean_ci_disp = f"<span title='{tt_cri_mean}'>[{b_res_cont.get('mean_ci_low', np.nan):.3f}, {b_res_cont.get('mean_ci_high', np.nan):.3f}]</span>" if pd.notna(b_res_cont.get('mean_ci_low')) else "N/A"
        prob_better_disp = f"<span title='{tt_prob_better}'>{b_res_cont.get('prob_better_than_control', np.nan)*100:.2f}%</span>" if pd.notna(b_res_cont.get('prob_better_than_control')) else ("N/A (Control)" if var_name == control_name else "N/A")
        exp_diff_disp = f"<span title='{tt_exp_diff}'>{b_res_cont.get('expected_diff_abs', np.nan):.3f}</span>" if pd.notna(b_res_cont.get('expected_diff_abs')) else ("N/A (Control)" if var_name == control_name else "N/A")
        diff_ci_disp = f"<span title='{tt_cri_diff}'>[{b_res_cont.get('diff_ci_low', np.nan):.3f}, {b_res_cont.get('diff_ci_high', np.nan):.3f}]</span>" if pd.notna(b_res_cont.get('diff_ci_low')) else ("N/A (Control)" if var_name == control_name else "N/A")
        prob_best_disp = f"<span title='{tt_prob_best}'>{b_res_cont.get('prob_best', np.nan)*100:.2f}%</span>" if pd.notna(b_res_cont.get('prob_best')) else "N/A"
        
        bayesian_data_disp_cont.append({
            "Variation": var_name, "Posterior Mean": post_mean_disp,
            f"{100*(1-alpha):.0f}% CrI for Mean": mean_ci_disp,
            "P(Better > Control) (%)": prob_better_disp,
            "Expected Diff. (abs)": exp_diff_disp,
            f"{100*(1-alpha):.0f}% CrI for Diff. (abs)": diff_ci_disp,
            "P(Being Best) (%)": prob_best_disp
        })
    if bayesian_data_disp_cont:
        bayesian_df_cont = pd.DataFrame(bayesian_data_disp_cont)
        st.markdown(bayesian_df_cont.to_html(escape=False, index=False), unsafe_allow_html=True) 

        st.markdown("##### Key Bayesian Insights (Continuous):")
        for item in bayesian_data_disp_cont:
            # ... (Dynamic interpretation captions as before) ...
            var_name_bayes = item["Variation"]
            if var_name_bayes == control_name: continue 
            prob_better_val_str = item["P(Better > Control) (%)"] 
            prob_better_numeric = np.nan
            if "N/A" not in prob_better_val_str:
                try: prob_better_numeric = float(prob_better_val_str.split('>')[1].split('%')[0]) / 100
                except: pass 
            cri_diff_str = item[f"{100*(1-alpha):.0f}% CrI for Diff. (abs)"] 
            cri_diff_contains_zero = "N/A" not in cri_diff_str and "[" in cri_diff_str and "]" in cri_diff_str and \
                                        float(cri_diff_str.split('[')[1].split(',')[0]) < 0 < float(cri_diff_str.split(',')[1].split(']')[0])
            expected_diff_val_str = item["Expected Diff. (abs)"]
            expected_diff_numeric = np.nan
            if "N/A" not in expected_diff_val_str:
                try: expected_diff_numeric = float(expected_diff_val_str.split('>')[1].split('<')[0])
                except: pass
            
            if pd.notna(prob_better_numeric):
                insight = f"For **{var_name_bayes}** vs Control:"
                if prob_better_numeric > (1 - alpha): 
                    insight += f" Strong evidence it's better (P(Better) = {prob_better_numeric*100:.1f}%)."
                elif prob_better_numeric > 0.5:
                    insight += f" More likely better than not (P(Better) = {prob_better_numeric*100:.1f}%)."
                else:
                    insight += f" Less likely to be better (P(Better) = {prob_better_numeric*100:.1f}%)."
                
                if pd.notna(expected_diff_numeric):
                    insight += f" Expected difference is {expected_diff_numeric:.3f}."
                if "N/A" not in cri_diff_str:
                    insight += f" The {100*(1-alpha):.0f}% CrI for difference is {cri_diff_str.replace('<span>','').replace('</span>','')}" 
                    if cri_diff_contains_zero:
                        insight += ", which includes zero (suggesting the difference may not be practically significant or could be due to chance)."
                    else:
                        insight += "."
                st.caption(insight)
    
    # Plots
    st.markdown("##### Posterior Distributions for Means (Continuous)")
    fig_mean_cont, ax_mean_cont = plt.subplots()
    max_density_mean_cont = 0
    all_mean_samples_for_plot = []
    for var_name_plot in ordered_vars_for_display:
        if var_name_plot in bayesian_results:
            samples = bayesian_results[var_name_plot].get('samples')
            if samples is not None and samples.size > 0 and not np.all(np.isnan(samples)):
                all_mean_samples_for_plot.extend(samples[~np.isnan(samples)])
    
    if all_mean_samples_for_plot: 
        x_min_mean_cont, x_max_mean_cont = np.min(all_mean_samples_for_plot), np.max(all_mean_samples_for_plot)
        padding_mean_cont = (x_max_mean_cont - x_min_mean_cont) * 0.1 if (x_max_mean_cont - x_min_mean_cont) > 0 else 1.0
        x_range_mean_cont = np.linspace(x_min_mean_cont - padding_mean_cont, x_max_mean_cont + padding_mean_cont, 300)

        for var_name in ordered_vars_for_display:
            if var_name not in bayesian_results: continue
            b_res_cont = bayesian_results[var_name]
            samples = b_res_cont.get('samples')
            if samples is not None and samples.size > 0 and not np.all(np.isnan(samples)):
                valid_samples = samples[~np.isnan(samples)]
                if valid_samples.size > 1: 
                    kde = gaussian_kde(valid_samples)
                    pdf_values = kde(x_range_mean_cont)
                    ax_mean_cont.plot(x_range_mean_cont, pdf_values, label=f"{var_name}")
                    ax_mean_cont.fill_between(x_range_mean_cont, pdf_values, alpha=0.2)
                    max_density_mean_cont = max(max_density_mean_cont, np.nanmax(pdf_values))
                elif valid_samples.size == 1: 
                    ax_mean_cont.axvline(valid_samples[0], label=f"{var_name} (single data point)", linestyle="--", alpha=0.7)
        if max_density_mean_cont > 0: ax_mean_cont.set_ylim(0, max_density_mean_cont * 1.1)
        elif not all_mean_samples_for_plot: pass 
        else: ax_mean_cont.set_ylim(0,1) 
        ax_mean_cont.set_title("Posterior Distributions of Means"); ax_mean_cont.set_xlabel(f"Mean of {outcome_col_name}"); 
        ax_mean_cont.set_ylabel("Density"); ax_mean_cont.legend(); st.pyplot(fig_mean_cont); plt.close(fig_mean_cont)
    else:
        st.caption(f"Not enough valid sample data to plot posterior distributions of means for {section_title_prefix.lower()} results.")

    st.markdown("##### Posterior Distribution of Difference (Variation Mean - Control Mean)")
    num_vars_to_plot_cont_diff = sum(1 for var_name_diff in bayesian_results if var_name_diff != control_name and bayesian_results[var_name_diff].get('diff_samples_vs_control') is not None and not np.all(np.isnan(bayesian_results[var_name_diff].get('diff_samples_vs_control'))))
    if num_vars_to_plot_cont_diff > 0:
        cols_diff_plots_cont = st.columns(min(num_vars_to_plot_cont_diff, 2)) 
        col_idx_cont_diff = 0
        for var_name in ordered_vars_for_display:
            if var_name == control_name or var_name not in bayesian_results: continue
            b_res_cont = bayesian_results[var_name]
            diff_samples = b_res_cont.get('diff_samples_vs_control')
            if diff_samples is None or diff_samples.size == 0 or np.all(np.isnan(diff_samples)): continue
            
            with cols_diff_plots_cont[col_idx_cont_diff % min(num_vars_to_plot_cont_diff, 2)]:
                fig_diff_cont, ax_diff_cont = plt.subplots(); 
                valid_diff_samples_plot = diff_samples[~np.isnan(diff_samples)]
                if valid_diff_samples_plot.size > 0:
                    ax_diff_cont.hist(valid_diff_samples_plot, bins=50, density=True, alpha=0.6, label=f"{var_name} - {control_name}")
                    ax_diff_cont.axvline(0, color='grey', linestyle='--'); 
                    expected_diff_val = b_res_cont.get('expected_diff_abs', np.nan)
                    if pd.notna(expected_diff_val):
                        ax_diff_cont.axvline(expected_diff_val, color='red', linestyle=':', label=f"Mean Diff: {expected_diff_val:.3f}")
                    ax_diff_cont.set_title(f"Diff: {var_name} vs {control_name}"); 
                    ax_diff_cont.set_xlabel(f"Difference in Mean of {outcome_col_name}"); ax_diff_cont.set_ylabel("Density"); 
                    ax_diff_cont.legend(); st.pyplot(fig_diff_cont); plt.close(fig_diff_cont)
                else:
                    st.caption(f"Not enough valid difference data to plot for {var_name} vs {control_name} for {section_title_prefix.lower()} results.")
                col_idx_cont_diff +=1
    st.markdown("""**Interpreting Bayesian Results (Continuous - Briefly):** (Full guidance in 'Bayesian Analysis Guidelines' section)
    - **Posterior Mean:** Average value of the metric after data. - **CrI for Mean:** Range for true mean. - **P(Better > Control):** Probability variation's true mean is higher. - **Expected Difference:** Average expected difference from control. - **CrI for Difference:** Range for true difference. If includes 0, 'no difference' is plausible. - **P(Being Best):** Probability variation has highest true mean.""")


def show_analyze_results_page():
    # Page for analyzing A/B test results.
    st.header("Analyze Your A/B Test Results 📊")
    st.markdown("Upload your A/B test data (as a CSV file) to perform an analysis.")
    st.markdown("---")

    cycle_suffix = "_c8_1" 
    # Initialize session state variables
    if f'analysis_done{cycle_suffix}' not in st.session_state: st.session_state[f'analysis_done{cycle_suffix}'] = False
    if f'df_analysis{cycle_suffix}' not in st.session_state: st.session_state[f'df_analysis{cycle_suffix}'] = None
    if f'metric_type_analysis{cycle_suffix}' not in st.session_state: st.session_state[f'metric_type_analysis{cycle_suffix}'] = 'Binary'
    if f'variation_col_analysis{cycle_suffix}' not in st.session_state: st.session_state[f'variation_col_analysis{cycle_suffix}'] = None
    if f'outcome_col_analysis{cycle_suffix}' not in st.session_state: st.session_state[f'outcome_col_analysis{cycle_suffix}'] = None
    if f'success_value_analysis{cycle_suffix}' not in st.session_state: st.session_state[f'success_value_analysis{cycle_suffix}'] = None
    if f'control_group_name_analysis{cycle_suffix}' not in st.session_state: st.session_state[f'control_group_name_analysis{cycle_suffix}'] = None
    if f'alpha_for_analysis{cycle_suffix}' not in st.session_state: st.session_state[f'alpha_for_analysis{cycle_suffix}'] = 0.05
    if f'overall_freq_summary_stats{cycle_suffix}' not in st.session_state: st.session_state[f'overall_freq_summary_stats{cycle_suffix}'] = None
    if f'overall_bayesian_results_binary{cycle_suffix}' not in st.session_state: st.session_state[f'overall_bayesian_results_binary{cycle_suffix}'] = None
    if f'overall_bayesian_results_continuous{cycle_suffix}' not in st.session_state: st.session_state[f'overall_bayesian_results_continuous{cycle_suffix}'] = None 
    if f'metric_col_name{cycle_suffix}' not in st.session_state: st.session_state[f'metric_col_name{cycle_suffix}'] = None
    if f'segmentation_cols{cycle_suffix}' not in st.session_state: st.session_state[f'segmentation_cols{cycle_suffix}'] = [] 
    if f'segmented_freq_results{cycle_suffix}' not in st.session_state: st.session_state[f'segmented_freq_results{cycle_suffix}'] = {}
    if f'segmented_bayesian_results{cycle_suffix}' not in st.session_state: st.session_state[f'segmented_bayesian_results{cycle_suffix}'] = {} # NEW for this version

    uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"], key=f"file_uploader{cycle_suffix}")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            st.session_state[f'df_analysis{cycle_suffix}'] = df 
            st.success("File Uploaded Successfully!")
            st.markdown("**Data Preview (first 5 rows):**"); st.dataframe(df.head())
            st.markdown("---")

            st.subheader("1. Map Data & Select Metric Type")
            columns = df.columns.tolist() 
            map_col1, map_col2, map_col3 = st.columns(3)
            with map_col1: st.session_state[f'variation_col_analysis{cycle_suffix}'] = st.selectbox("Select 'Variation ID' column:", options=columns, index=0 if columns else -1, key=f"var_col{cycle_suffix}")
            with map_col2: st.session_state[f'outcome_col_analysis{cycle_suffix}'] = st.selectbox("Select 'Outcome' column:", options=columns, index=len(columns)-1 if len(columns)>1 else (0 if columns else -1), key=f"out_col{cycle_suffix}")
            with map_col3: st.session_state[f'metric_type_analysis{cycle_suffix}'] = st.radio("Select Metric Type for Outcome Column:", ('Binary', 'Continuous'), key=f"metric_type_analysis_radio{cycle_suffix}", horizontal=True)

            if st.session_state[f'metric_type_analysis{cycle_suffix}'] == 'Binary':
                outcome_col = st.session_state[f'outcome_col_analysis{cycle_suffix}']
                if outcome_col and outcome_col in df.columns:
                    unique_outcomes = df[outcome_col].unique()
                    if len(unique_outcomes) == 1: st.warning(f"Outcome column '{outcome_col}' has only one value: `{unique_outcomes[0]}`.")
                    elif len(unique_outcomes) > 2 and len(unique_outcomes) <=10 : st.warning(f"Outcome column '{outcome_col}' has >2 unique values: `{unique_outcomes}`. Please select the value representing 'Conversion' or 'Success'.")
                    elif len(unique_outcomes) > 10: st.warning(f"Outcome column '{outcome_col}' has many unique values ({len(unique_outcomes)}). Ensure this is a binary outcome and select the success value.")

                    if len(unique_outcomes) > 0:
                        str_options = [str(val) for val in unique_outcomes]
                        default_success_idx = 0
                        common_success_indicators = ['1', 'True', 'true', 'yes', 'Yes', 'Success', 'Converted']
                        for indicator in common_success_indicators:
                            if indicator in str_options: 
                                default_success_idx = str_options.index(indicator)
                                break
                        
                        success_value_str = st.selectbox(f"Which value in '{outcome_col}' is 'Conversion' (Success)?", options=str_options, index=default_success_idx, key=f"succ_val{cycle_suffix}")
                        
                        original_dtype = df[outcome_col].dtype
                        if success_value_str.lower() == 'nan' and any(pd.isna(val) for val in unique_outcomes): st.session_state[f'success_value_analysis{cycle_suffix}'] = np.nan
                        elif pd.api.types.is_numeric_dtype(original_dtype) and not pd.api.types.is_bool_dtype(original_dtype):
                            try: st.session_state[f'success_value_analysis{cycle_suffix}'] = original_dtype.type(success_value_str)
                            except ValueError: st.session_state[f'success_value_analysis{cycle_suffix}'] = success_value_str 
                        elif pd.api.types.is_bool_dtype(original_dtype): st.session_state[f'success_value_analysis{cycle_suffix}'] = (success_value_str.lower() == 'true') 
                        else: st.session_state[f'success_value_analysis{cycle_suffix}'] = success_value_str
                    else: st.warning(f"Could not determine distinct values in outcome column '{outcome_col}'.")
                elif outcome_col: st.warning(f"Selected outcome column '{outcome_col}' not found in the uploaded CSV.")
                else: st.warning("Please select a valid outcome column.")
            else: 
                st.session_state[f'success_value_analysis{cycle_suffix}'] = None 
                outcome_col = st.session_state[f'outcome_col_analysis{cycle_suffix}']
                if outcome_col and outcome_col in df.columns and not pd.api.types.is_numeric_dtype(df[outcome_col]):
                    st.error(f"For 'Continuous' metric type, the outcome column '{outcome_col}' must be numeric. Current type: {df[outcome_col].dtype}. Please select a numeric column or check your data.")
            
            st.markdown("---"); st.subheader("2. Select Your Control Group & Analysis Alpha")
            var_col_sel = st.session_state[f'variation_col_analysis{cycle_suffix}']
            if var_col_sel and var_col_sel in df.columns and st.session_state[f'df_analysis{cycle_suffix}'] is not None:
                variation_names = st.session_state[f'df_analysis{cycle_suffix}'][var_col_sel].astype(str).unique().tolist() 
                if variation_names: st.session_state[f'control_group_name_analysis{cycle_suffix}'] = st.selectbox("Select 'Control Group':", options=variation_names, index=0, key=f"ctrl_grp{cycle_suffix}")
                else: st.warning(f"No unique variations found in column '{var_col_sel}'.")
            elif var_col_sel: st.warning(f"Selected variation column '{var_col_sel}' not found in the uploaded CSV.")
            else: st.warning("Please select a valid variation column.")
            st.session_state[f'alpha_for_analysis{cycle_suffix}'] = st.slider("Significance Level (\u03B1) for Analysis (%)", 1, 20, 5, 1, key=f"alpha_analysis{cycle_suffix}_slider") / 100.0
            
            st.markdown("---"); st.subheader("3. Optional: Segmentation Analysis")
            st.markdown("Select one or more columns to segment your results by. Analysis will be performed for the overall data and then for each segment.")
            
            potential_segment_cols = [col for col in columns if col not in [st.session_state[f'variation_col_analysis{cycle_suffix}'], st.session_state[f'outcome_col_analysis{cycle_suffix}']] ]
            good_segment_cols = []
            if df is not None:
                for col in potential_segment_cols:
                    if df[col].nunique(dropna=True) <= 20: 
                        good_segment_cols.append(col)
                    else:
                        st.caption(f"Column '{col}' has >20 unique values and is excluded from segmentation options for performance.")

            if not good_segment_cols:
                st.info("No suitable columns found for segmentation (e.g., columns might have too many unique values, or only variation/outcome columns remain).")
                st.session_state[f'segmentation_cols{cycle_suffix}'] = [] 
            else:
                st.session_state[f'segmentation_cols{cycle_suffix}'] = st.multiselect(
                    "Select segmentation column(s):", options=good_segment_cols,
                    default=st.session_state.get(f'segmentation_cols{cycle_suffix}', []), 
                    help="Results will be broken down by unique combinations of values in these columns."
                )
            
            st.markdown("---") 
            analysis_button_label = f"🚀 Run Analysis ({st.session_state[f'metric_type_analysis{cycle_suffix}']} Outcome)"
            if st.button(analysis_button_label, key=f"run_analysis_button{cycle_suffix}"):
                st.session_state[f'analysis_done{cycle_suffix}'] = False
                st.session_state[f'overall_freq_summary_stats{cycle_suffix}'] = None
                st.session_state[f'overall_bayesian_results_binary{cycle_suffix}'] = None
                st.session_state[f'overall_bayesian_results_continuous{cycle_suffix}'] = None
                st.session_state[f'segmented_freq_results{cycle_suffix}'] = {} 
                st.session_state[f'segmented_bayesian_results{cycle_suffix}'] = {} # Reset segmented Bayesian results

                valid_setup = True
                var_col_val = st.session_state[f'variation_col_analysis{cycle_suffix}']
                out_col_val = st.session_state[f'outcome_col_analysis{cycle_suffix}']
                control_val = st.session_state[f'control_group_name_analysis{cycle_suffix}']
                df_val = st.session_state[f'df_analysis{cycle_suffix}']

                if not (var_col_val and var_col_val in df_val.columns and \
                        out_col_val and out_col_val in df_val.columns and \
                        control_val is not None):
                    st.error("Please complete all column mapping and control group selections with valid columns from the CSV."); valid_setup = False
                
                if st.session_state[f'metric_type_analysis{cycle_suffix}'] == 'Binary' and st.session_state[f'success_value_analysis{cycle_suffix}'] is None:
                    st.error("For Binary outcome, please specify the 'Conversion (Success)' value."); valid_setup = False
                
                if st.session_state[f'metric_type_analysis{cycle_suffix}'] == 'Continuous':
                    if not (out_col_val and out_col_val in df_val.columns): 
                        st.error("Please select a valid outcome column for continuous analysis."); valid_setup = False
                    elif not pd.api.types.is_numeric_dtype(df_val[out_col_val]):
                        st.error(f"For 'Continuous' metric type, outcome column '{out_col_val}' must be numeric."); valid_setup = False
                
                if valid_setup:
                    try:
                        overall_df_for_analysis = df_val.copy() 
                        metric_type = st.session_state[f'metric_type_analysis{cycle_suffix}']
                        alpha = st.session_state[f'alpha_for_analysis{cycle_suffix}']

                        if metric_type == 'Binary':
                            success_val_bin = st.session_state[f'success_value_analysis{cycle_suffix}']
                            if pd.isna(success_val_bin): 
                                overall_df_for_analysis['__outcome_processed__'] = overall_df_for_analysis[out_col_val].isna().astype(int)
                            else: 
                                overall_df_for_analysis['__outcome_processed__'] = (overall_df_for_analysis[out_col_val] == success_val_bin).astype(int)
                            st.session_state[f'metric_col_name{cycle_suffix}'] = 'Metric Value (%)'
                        elif metric_type == 'Continuous':
                            overall_df_for_analysis[out_col_val] = pd.to_numeric(overall_df_for_analysis[out_col_val], errors='coerce')
                            st.session_state[f'metric_col_name{cycle_suffix}'] = 'Mean_Value'
                        
                        # --- Overall Analysis ---
                        st.markdown("---"); st.subheader(f"Overall Frequentist Analysis Results ({metric_type} Outcome)")
                        overall_summary_stats = display_frequentist_analysis(
                            overall_df_for_analysis, metric_type, out_col_val, var_col_val, control_val, alpha, 
                            summary_stats_key_suffix="_overall"
                        )
                        st.session_state[f'overall_freq_summary_stats{cycle_suffix}'] = overall_summary_stats

                        if overall_summary_stats is not None:
                            if metric_type == 'Binary':
                                bayes_bin_res, bayes_bin_err = run_bayesian_binary_analysis(overall_summary_stats, control_val, ci_level=(1-alpha))
                                if bayes_bin_err: st.error(f"Overall Bayesian Binary Analysis Error: {bayes_bin_err}")
                                else: st.session_state[f'overall_bayesian_results_binary{cycle_suffix}'] = bayes_bin_res
                            elif metric_type == 'Continuous':
                                bayes_cont_res, bayes_cont_err = run_bayesian_continuous_analysis(overall_summary_stats, control_val, ci_level=(1-alpha))
                                if bayes_cont_err: st.error(f"Overall Bayesian Continuous Analysis Error: {bayes_cont_err}")
                                else: st.session_state[f'overall_bayesian_results_continuous{cycle_suffix}'] = bayes_cont_res
                        
                        st.session_state[f'analysis_done{cycle_suffix}'] = True 

                        # --- Segmentation Logic ---
                        selected_segment_cols = st.session_state.get(f'segmentation_cols{cycle_suffix}', [])
                        if selected_segment_cols:
                            segment_group_col = '__segment_group__'
                            missing_seg_cols = [sc for sc in selected_segment_cols if sc not in overall_df_for_analysis.columns]
                            if missing_seg_cols:
                                st.error(f"Selected segmentation column(s) not found: {', '.join(missing_seg_cols)}")
                            else:
                                overall_df_for_analysis[segment_group_col] = overall_df_for_analysis[selected_segment_cols].astype(str).agg(' | '.join, axis=1)
                                unique_segments = overall_df_for_analysis[segment_group_col].unique()
                                
                                st.markdown("---"); st.subheader("Segmented Analysis Results")
                                if not unique_segments.size:
                                     st.info("No unique segments found based on selected columns.")
                                else:
                                    st.info(f"Found {len(unique_segments)} unique segment(s). Performing analysis for each.")

                                for segment_value in unique_segments:
                                    with st.expander(f"Segment: {segment_value}", expanded=False):
                                        st.markdown(f"#### Frequentist Analysis for Segment: {segment_value}")
                                        segment_df = overall_df_for_analysis[overall_df_for_analysis[segment_group_col] == segment_value].copy()
                                        
                                        segment_summary_stats = display_frequentist_analysis(
                                            segment_df, metric_type, out_col_val, var_col_val, control_val, alpha,
                                            summary_stats_key_suffix=f"_segment_{segment_value.replace(' | ','_')}"
                                        )
                                        st.session_state[f'segmented_freq_results{cycle_suffix}'][segment_value] = {'summary_stats': segment_summary_stats}
                                        
                                        # Bayesian Analysis for Segment
                                        if segment_summary_stats is not None and not segment_summary_stats.empty:
                                            if metric_type == 'Binary':
                                                seg_bayes_bin_res, seg_bayes_bin_err = run_bayesian_binary_analysis(segment_summary_stats, control_val, ci_level=(1-alpha))
                                                if seg_bayes_bin_err: st.error(f"Segment '{segment_value}' Bayesian Binary Analysis Error: {seg_bayes_bin_err}")
                                                else: 
                                                    st.session_state[f'segmented_bayesian_results{cycle_suffix}'][segment_value] = seg_bayes_bin_res
                                                    display_bayesian_binary_results(seg_bayes_bin_res, segment_summary_stats, control_val, alpha, section_title_prefix=f"Segment '{segment_value}'")
                                            elif metric_type == 'Continuous':
                                                seg_bayes_cont_res, seg_bayes_cont_err = run_bayesian_continuous_analysis(segment_summary_stats, control_val, ci_level=(1-alpha))
                                                if seg_bayes_cont_err: st.error(f"Segment '{segment_value}' Bayesian Continuous Analysis Error: {seg_bayes_cont_err}")
                                                else:
                                                    st.session_state[f'segmented_bayesian_results{cycle_suffix}'][segment_value] = seg_bayes_cont_res
                                                    display_bayesian_continuous_results(seg_bayes_cont_res, segment_summary_stats, control_val, alpha, out_col_val, section_title_prefix=f"Segment '{segment_value}'")
                                        else:
                                            st.warning(f"Skipping Bayesian analysis for segment '{segment_value}' due to lack of valid summary statistics.")
                        else:
                             st.session_state[f'segmented_freq_results{cycle_suffix}'] = {} 
                             st.session_state[f'segmented_bayesian_results{cycle_suffix}'] = {}
                    except Exception as e: st.error(f"An error occurred during analysis setup or execution: {e}"); st.exception(e)
        except Exception as e: st.error(f"Error reading/processing CSV: {e}"); st.exception(e)
    else: st.info("Upload a CSV file to begin analysis.")

    # --- Display Overall Bayesian Results (after Frequentist, if analysis_done) ---
    if st.session_state[f'analysis_done{cycle_suffix}'] and not st.session_state.get(f'segmentation_cols{cycle_suffix}', []): # Only show if no segmentation or after overall freq
        alpha_display = st.session_state[f'alpha_for_analysis{cycle_suffix}']
        metric_type_display = st.session_state[f'metric_type_analysis{cycle_suffix}']
        overall_summary_stats_for_bayes_ordering = st.session_state.get(f'overall_freq_summary_stats{cycle_suffix}')
        control_name_display = st.session_state[f'control_group_name_analysis{cycle_suffix}']
        outcome_col_display = st.session_state[f'outcome_col_analysis{cycle_suffix}']

        if metric_type_display == 'Binary':
            bayesian_results_to_display = st.session_state.get(f'overall_bayesian_results_binary{cycle_suffix}')
            display_bayesian_binary_results(bayesian_results_to_display, overall_summary_stats_for_bayes_ordering, control_name_display, alpha_display, section_title_prefix="Overall")
        elif metric_type_display == 'Continuous':
            bayesian_results_to_display_cont = st.session_state.get(f'overall_bayesian_results_continuous{cycle_suffix}')
            display_bayesian_continuous_results(bayesian_results_to_display_cont, overall_summary_stats_for_bayes_ordering, control_name_display, alpha_display, outcome_col_display, section_title_prefix="Overall")
    
    st.markdown("---")
    st.info("Full segmented Bayesian analysis and detailed interpretation guidance are planned for future updates!")


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
        * For **binary outcomes (Conversion Rates)**, this app uses a `Beta(1,1)` prior by default. This is an "uninformative" or "flat" prior, meaning it assumes all conversion rates between 0% and 100% are equally likely before the test. This is a common starting point when strong prior information is unavailable.
        * For **continuous outcomes (Means)**, the app uses a method that approximates the posterior distribution of the mean using a t-distribution derived from your sample data (sample mean, sample standard deviation, sample size). This approach implicitly uses non-informative (or "vague") priors for the underlying true mean and variance, which is a common simplification. More advanced Bayesian models for continuous data might involve explicitly setting more informative priors (e.g., Normal prior for the mean, Inverse-Gamma prior for the variance).
    * **Likelihood:** This is how your collected A/B test data informs the model about the parameters.
        * For binary data (conversions/non-conversions), the likelihood is typically Binomial.
        * For continuous data (assuming the data points are approximately normally distributed within each group), the likelihood is Normal.
    * **Posterior Distribution:** Your updated belief about the parameter *after* combining the prior and the data (via the likelihood). This is the key output of Bayesian analysis.
        * For binary outcomes: Beta Prior + Binomial Likelihood = Beta Posterior. The Beta distribution is convenient as it's the "conjugate prior" for the Binomial likelihood.
        * For continuous outcomes: When using uninformative priors for the mean and variance, and a Normal likelihood, the marginal posterior distribution for the mean (after integrating out the unknown variance) follows a t-distribution. This is analogous to how the t-distribution arises in frequentist inference for means with unknown variance.
    """)

    st.subheader("Interpreting Key Bayesian Outputs")
    st.markdown("**For Binary Outcomes (e.g., Conversion Rates):**")
    st.markdown("""
    * **Posterior Distribution Plot (CRs):** Visualizes the range of plausible values for the true conversion rate of each variation after seeing the data. Wider distributions mean more uncertainty.
    * **Posterior Mean CR & Credible Interval (CrI) for CR:**
        * *Posterior Mean CR:* The average conversion rate for a variation, based on the posterior distribution. It's your best guess for the true CR.
        * *CrI for CR:* We are X% confident (e.g., 95%) that the true conversion rate for this variation lies within this interval. Unlike frequentist Confidence Intervals, you *can* make direct probability statements about the parameter being in the interval.
    * **P(Variation > Control) (Probability of Being Better):** The probability that the variation's true underlying conversion rate is strictly greater than the control's true conversion rate. A high value (e.g., >95%) gives strong confidence the variation is an improvement.
        * *Important Note:* Even if this probability is high, also check the **Credible Interval for Uplift**. If that interval is very wide or very close to zero (e.g., [-0.1%, 0.3%]), the magnitude of the improvement might be small or uncertain, even if you're confident it's positive. A P(Better) of 96% with an uplift CrI of [2.0%, 5.0%] is more compelling.
    * **Expected Uplift (Absolute %):** The average absolute improvement (or decline) in conversion rate you might expect from choosing a variation over the control, based on the posterior distributions of the difference.
    * **Credible Interval (CrI) for Uplift (Absolute %):** We are X% confident that the true absolute uplift (Variation CR - Control CR) lies within this interval.
        * If this interval includes 0, then 'no difference' or even a negative impact are plausible outcomes. The wider the interval, the more uncertain the uplift.
        * If the entire interval is above 0, you have strong evidence of a positive uplift.
        * If the entire interval is below 0, you have strong evidence of a negative impact.
    * **P(Being Best):** In an A/B/n test, this is the probability that a specific variation has the highest true conversion rate among all tested variations. Useful for selecting a winner when multiple variations are present, but also consider the magnitudes of differences and their CrIs.
    """)

    st.markdown("**For Continuous Outcomes (e.g., Average Order Value, Time on Page):**")
    st.markdown("""
    * **Posterior Distribution Plot (Means):** Visualizes the plausible values for the true mean of the metric for each variation, based on the t-distribution approximation of the posterior. Similar to binary, wider distributions indicate more uncertainty about the true mean.
    * **Posterior Mean & Credible Interval (CrI) for Mean:**
        * *Posterior Mean:* The average value of the metric for a variation, based on its posterior distribution (will be very close to the sample mean given the uninformative prior approach).
        * *CrI for Mean:* We are X% confident that the true mean for this variation lies within this interval. For example, a 95% CrI of [\$10.50, \$12.30] for Average Order Value (AOV) means there's a 95% probability the true AOV for that variation is between \$10.50 and \$12.30.
    * **P(Variation > Control) (Probability of Being Better):** The probability that the variation's true underlying mean is strictly greater than the control's true mean. A high value suggests the variation is likely better.
        * *Example:* If P(Var_B_Mean > Control_Mean) = 97%, there's a 97% chance that Variation B truly has a higher average outcome than Control. This is a direct probabilistic statement about the hypothesis.
    * **Expected Difference (Absolute):** The average absolute difference (Variation Mean - Control Mean) you might expect, based on the posterior distribution of the difference. This is the mean of the difference samples.
        * *Example:* If the Expected Difference for Variation B vs Control is +\$1.50 for AOV, you expect, on average, Variation B to result in an AOV that is \$1.50 higher than Control.
    * **Credible Interval (CrI) for Difference (Absolute):** We are X% confident that the true absolute difference in means lies within this interval. This is crucial for understanding the magnitude and uncertainty of the difference.
        * *Example 1 (Uncertain but likely positive):* A 95% CrI for Difference of [-\$0.50, +\$3.50] for AOV, with an Expected Difference of +\$1.50. While the expected difference is positive and P(Better) might be high, the interval includes \$0. This means it's plausible there's no real difference, or even a slight negative effect. The decision to launch might depend on the cost/risk.
        * *Example 2 (Strongly positive):* A 95% CrI for Difference of [+\$0.50, +\$2.50]. Here, the entire interval is above zero, providing strong evidence that the variation has a positive impact, with the true effect likely between \$0.50 and \$2.50.
    * **P(Being Best):** The probability that a specific variation has the highest true mean among all tested variations. Useful for ranking, but always consider the CrIs for differences to ensure the "best" is meaningfully better than others, especially the control.
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

if callable(PAGES.get(selection)):
    page_function = PAGES[selection]
    page_function()
else:
    st.error("Selected page could not be loaded.")


st.sidebar.markdown("---")
st.sidebar.info("A/B Testing Guide & Analyzer | V0.8.2 (Cycle 8 - Segmented Bayesian)")
