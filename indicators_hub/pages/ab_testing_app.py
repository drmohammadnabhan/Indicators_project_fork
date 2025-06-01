import streamlit as st
import numpy as np
from scipy.stats import norm
import math
import pandas as pd # Added for Cycle 4
from statsmodels.stats.proportion import proportions_ztest # Added for Cycle 4
from statsmodels.stats.proportion import confint_proportions_2indep # Added for CI for diff

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

# --- Page Functions ---
def show_introduction_page():
    # ... (Content from Cycle 1 - V0.2.2)
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
    # ... (Content from Cycle 2 - V0.2.2 - including Sample Size Calculator, Formula, and Impacts expanders)
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
    num_variations_ss = st.number_input( # Renamed to avoid key conflict if we have num_variations elsewhere
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
        st.markdown(r"""Adjusting the input parameters for the sample size calculator has direct consequences on the number of users you'll need. Understanding these trade-offs is key for planning your A/B tests effectively: ... * **Baseline Conversion Rate (BCR):** ... * **Minimum Detectable Effect (MDE):** ... * **Statistical Power (1 - $\beta$):** ... * **Significance Level ($\alpha$):** ... * **Number of Variations:** ... Balancing these factors is key...""")
    st.markdown("---")
    st.subheader("Common Pitfalls in A/B Test Design & Execution")
    st.markdown("Avoiding these common mistakes can significantly improve the quality and reliability of your A/B tests.")
    pitfalls = {
        "Too Short Test Duration / Insufficient Sample Size": {"what": "Ending a test before collecting enough data or running it for an arbitrarily short period.", "problem": "Results may be statistically underpowered or due to random noise.", "howto": "Calculate sample size beforehand and run tests for at least one full business cycle.", "analogy": "Judging a marathon by the first 100 meters."},
        "Ignoring Statistical Significance / Power": {"what": "Making decisions without considering if differences are statistically significant or if the test had enough power.", "problem": "You might implement changes with no real impact or discard truly better variations.", "howto": "Always check p-values/CIs (Frequentist) or probabilities/CrIs (Bayesian). Ensure adequate power.", "analogy": "Getting 2 heads in 3 coin flips doesn't mean a biased coin without more data."},
        "Testing Too Many Things at Once (in a Single Variation)": {"what": "Changing multiple elements in one variation.", "problem": "Impossible to isolate which change caused the performance difference.", "howto": "Test one significant change at a time to understand its specific impact.", "analogy": "Changing multiple ingredients in a recipe; if it's bad, which ingredient was it?"},
        "External Factors Affecting the Test": {"what": "Events outside your test (holidays, campaigns, outages) influencing behavior.", "problem": "Can skew results, making variations appear different due to external factors.", "howto": "Be aware of the test environment. Document external events. Consider restarting if data is heavily skewed.", "analogy": "Measuring plant growth with different fertilizers during a surprise heatwave."},
        "Regression to the Mean": {"what": "Initial extreme results tending to become closer to the average over time.", "problem": "Stopping tests early based on extreme initial results can be misleading.", "howto": "Run tests for their planned duration and sample size.", "analogy": "A basketball player's hot streak eventually cools down to their average performance."},
        "Not Segmenting Results (When Appropriate)": {"what": "Only looking at overall average results, not how different user segments reacted.", "problem": "Overall flat results might hide significant wins in one segment and losses in another.", "howto": "Analyze performance for important, predefined user segments. (Coming to this app later!)", "analogy": "A new song might have mixed reviews overall, but specific age groups might love/hate it."},
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

# --- NEW Analyze Results Page for Cycle 4 ---
def show_analyze_results_page():
    st.header("Analyze Your A/B Test Results 📊")
    st.markdown("Upload your A/B test data (as a CSV file) to perform a Frequentist analysis for **binary outcomes**.")
    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File Uploaded Successfully!")
            st.markdown("**Data Preview (first 5 rows):**")
            st.dataframe(df.head())
            st.markdown("---")

            st.subheader("1. Map Your Data Columns")
            columns = df.columns.tolist()
            
            col1_map, col2_map = st.columns(2)
            with col1_map:
                variation_col = st.selectbox(
                    "Select your 'Variation ID' column:", 
                    options=columns, 
                    index=0,
                    help="This column should contain the names or identifiers of your test groups (e.g., 'Control', 'Variation A', 'Group B')."
                )
            with col2_map:
                outcome_col = st.selectbox(
                    "Select your 'Outcome' column (Binary - e.g., Converted/Not):", 
                    options=columns, 
                    index=len(columns)-1 if len(columns)>1 else 0,
                    help="This column should indicate if a conversion occurred (e.g., 1 for conversion, 0 for no conversion; or 'Yes', 'No')."
                )

            # Determine unique values in outcome column to identify success event
            if outcome_col:
                unique_outcomes = df[outcome_col].unique()
                if len(unique_outcomes) == 1:
                    st.warning(f"The outcome column '{outcome_col}' only has one value: `{unique_outcomes[0]}`. A binary outcome requires two distinct values.")
                    success_value_options = [unique_outcomes[0]]
                elif len(unique_outcomes) > 2:
                    st.warning(f"The outcome column '{outcome_col}' has more than two unique values: `{unique_outcomes}`. For binary analysis, please select the value that represents a 'conversion' or success.")
                    success_value_options = unique_outcomes
                elif len(unique_outcomes) == 2:
                    success_value_options = unique_outcomes
                else: # Should not happen if column exists
                    success_value_options = []

                if success_value_options:
                    success_value_str = st.selectbox(
                        f"Which value in '{outcome_col}' represents a 'Conversion' (Success)?",
                        options=[str(val) for val in success_value_options], # Ensure options are strings for selectbox
                        help="Select the value that indicates the desired outcome happened."
                    )
                    # Attempt to convert success_value_str back to its original type for comparison
                    original_dtype = df[outcome_col].dtype
                    if pd.api.types.is_numeric_dtype(original_dtype) and not pd.api.types.is_bool_dtype(original_dtype):
                        try:
                            success_value = original_dtype.type(success_value_str)
                        except ValueError:
                            success_value = success_value_str # Fallback if conversion fails
                    elif pd.api.types.is_bool_dtype(original_dtype):
                         success_value = (success_value_str.lower() == 'true')
                    else: # String or other
                        success_value = success_value_str

            st.markdown("---")
            st.subheader("2. Select Your Control Group")
            if variation_col:
                variation_names = df[variation_col].unique().tolist()
                control_group_name = st.selectbox(
                    "Select your 'Control Group' name:",
                    options=variation_names,
                    index=0,
                    help="Choose the variation that represents your baseline or original version."
                )
            
            st.markdown("---")
            alpha_analysis = st.slider("Significance Level (α) for Analysis (%)", 1, 10, 5, 1, key="alpha_analysis",
                                       help="Set the significance level (alpha) for hypothesis testing. Typically 5%.") / 100.0


            if st.button("🚀 Run Frequentist Analysis (Binary Outcome)", key="run_analysis_button"):
                if not variation_col or not outcome_col or not control_group_name or not success_value_options:
                    st.error("Please complete all column mapping and control group selections.")
                else:
                    try:
                        # Create a binary 'converted' column based on user's selection
                        df['__converted_binary__'] = (df[outcome_col] == success_value).astype(int)

                        # Descriptive Statistics
                        st.subheader("📊 Descriptive Statistics")
                        summary_stats = df.groupby(variation_col).agg(
                            Users=('__converted_binary__', 'count'),
                            Conversions=('__converted_binary__', 'sum')
                        ).reset_index()
                        summary_stats['Conversion Rate (%)'] = (summary_stats['Conversions'] / summary_stats['Users'] * 100).round(2)
                        st.dataframe(summary_stats)

                        # Plot Conversion Rates
                        if not summary_stats.empty:
                             st.bar_chart(summary_stats.set_index(variation_col)['Conversion Rate (%)'])
                        
                        # Comparison Metrics vs Control
                        st.subheader(f"📈 Comparison vs. Control ('{control_group_name}')")
                        
                        control_data = summary_stats[summary_stats[variation_col] == control_group_name]
                        if control_data.empty:
                            st.error(f"Control group '{control_group_name}' not found in the data or has no users.")
                        else:
                            control_users = control_data['Users'].iloc[0]
                            control_conversions = control_data['Conversions'].iloc[0]
                            control_cr = control_conversions / control_users if control_users > 0 else 0

                            comparison_results = []
                            for index, row in summary_stats.iterrows():
                                var_name = row[variation_col]
                                if var_name == control_group_name:
                                    continue # Skip control group itself for comparison table

                                var_users = row['Users']
                                var_conversions = row['Conversions']
                                var_cr = var_conversions / var_users if var_users > 0 else 0

                                if control_users == 0 or var_users == 0:
                                    p_value, ci_low, ci_high, abs_uplift, rel_uplift, significant = 'N/A (Zero users in a group)', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
                                else:
                                    abs_uplift = var_cr - control_cr
                                    rel_uplift = (abs_uplift / control_cr) * 100 if control_cr > 0 else float('inf')
                                    
                                    # Z-test for proportions
                                    count = np.array([var_conversions, control_conversions])
                                    nobs = np.array([var_users, control_users])
                                    
                                    if np.any(nobs == 0) or np.any(count > nobs): # Additional check
                                        p_value, significant = 'N/A (Invalid counts/nobs)', 'N/A'
                                        ci_low_diff, ci_high_diff = 'N/A', 'N/A'
                                    else:
                                        z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
                                        significant_bool = p_value < alpha_analysis
                                        significant = f"Yes (p={p_value:.4f})" if significant_bool else f"No (p={p_value:.4f})"
                                        
                                        # Confidence interval for the difference
                                        # Using confint_proportions_2indep for difference
                                        # Note: statsmodels might not have a direct CI for proportion *difference* easily.
                                        # Let's calculate it manually for now, or use a simpler representation.
                                        # For simplicity, we'll focus on p-value for now and add CI for difference later if needed for V1 freq.
                                        # For now, let's just state the p-value and significance.
                                        # A more robust way might be to use `proportion_confint` for each group and compare.
                                        # However, the prompt asked for CI for Uplift.
                                        # statsmodels confint_proportions_2indep is for risk difference by default with method='wald'
                                        ci_low_diff, ci_high_diff = confint_proportions_2indep(var_conversions, var_users, control_conversions, control_users, method='wald', alpha=alpha_analysis)


                                comparison_results.append({
                                    "Variation": var_name,
                                    "Conversion Rate (%)": f"{var_cr*100:.2f}",
                                    "Absolute Uplift (%)": f"{abs_uplift*100:.2f}",
                                    "Relative Uplift (%)": f"{rel_uplift:.2f}%" if control_cr > 0 else "N/A",
                                    "P-value (vs Control)": f"{p_value:.4f}" if isinstance(p_value, float) else p_value,
                                    f"CI {100*(1-alpha_analysis):.0f}% for Diff. (%)": f"[{ci_low_diff*100:.2f}, {ci_high_diff*100:.2f}]" if isinstance(ci_low_diff, float) else "N/A",
                                    "Statistically Significant?": significant
                                })
                            
                            if comparison_results:
                                comparison_df = pd.DataFrame(comparison_results)
                                st.dataframe(comparison_df)
                                for _, row in comparison_df.iterrows():
                                    if "Yes" in str(row["Statistically Significant?"]):
                                        st.caption(f"The difference between **{row['Variation']}** and control ('{control_group_name}') is statistically significant at the {alpha_analysis*100:.0f}% level ({row['P-value (vs Control)']}). This means there's strong evidence against the null hypothesis (that there's no difference).")
                                    elif "No" in str(row["Statistically Significant?"]):
                                         st.caption(f"The difference between **{row['Variation']}** and control ('{control_group_name}') is not statistically significant at the {alpha_analysis*100:.0f}% level ({row['P-value (vs Control)']}). This means we don't have enough evidence to reject the null hypothesis (that there's no difference).")


                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")
                        st.exception(e) # Shows full traceback for debugging

        except Exception as e:
            st.error(f"Error reading or processing CSV file: {e}")
    else:
        st.info("Upload a CSV file to begin analysis.")
    
    st.markdown("---")
    st.info("Bayesian analysis and further enhancements coming in Cycle 5!")


def show_interpret_results_page():
    # ... (Content from Cycle 1 - V0.2.2)
    st.header("Interpreting Results & Detailed Decision Guidance 🧐")
    st.write("Guidance on how to interpret your A/B test results and make decisions will be implemented in Cycle 9.")
    st.info("Coming soon: Understanding statistical vs. practical significance, next steps!")


def show_faq_page():
    # ... (Content from Cycle 3 - V0.3)
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
    "FAQ on Misinterpretations": show_faq_page,
    "Roadmap / Possible Future Features": show_roadmap_page
}
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page_function = PAGES[selection]
page_function()

st.sidebar.markdown("---")
st.sidebar.info("A/B Testing Guide & Analyzer | V0.4 (Cycle 4)")
