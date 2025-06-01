import streamlit as st
import numpy as np
from scipy.stats import norm
import math
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

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
    st.markdown("Upload your A/B test data (as a CSV file) to perform a Frequentist analysis for **binary outcomes**.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"], key="file_uploader_cycle4")

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
                    help="This column should contain the names or identifiers of your test groups (e.g., 'Control', 'Variation A', 'Group B').",
                    key="variation_col_select"
                )
            with col2_map:
                outcome_col = st.selectbox(
                    "Select your 'Outcome' column (Binary - e.g., Converted/Not):", 
                    options=columns, 
                    index=len(columns)-1 if len(columns)>1 else 0,
                    help="This column should indicate if a conversion occurred (e.g., 1 for conversion, 0 for no conversion; or 'Yes', 'No').",
                    key="outcome_col_select"
                )

            success_value_options = [] 
            success_value = None     

            if outcome_col:
                unique_outcomes = df[outcome_col].unique()
                if len(unique_outcomes) == 1:
                    st.warning(f"The outcome column '{outcome_col}' only has one value: `{unique_outcomes[0]}`. A binary outcome requires two distinct values for meaningful analysis.")
                    success_value_options = unique_outcomes 
                elif len(unique_outcomes) > 2:
                    st.warning(f"The outcome column '{outcome_col}' has more than two unique values: `{unique_outcomes}`. For binary analysis, please select the value that represents a 'conversion' or success.")
                    success_value_options = unique_outcomes
                elif len(unique_outcomes) == 2:
                    success_value_options = unique_outcomes
                
                if len(success_value_options) > 0:
                    success_value_str = st.selectbox(
                        f"Which value in '{outcome_col}' represents a 'Conversion' (Success)?",
                        options=[str(val) for val in success_value_options],
                        index = 0, 
                        help="Select the value that indicates the desired outcome happened.",
                        key="success_value_select"
                    )
                    
                    original_dtype = df[outcome_col].dtype
                    if success_value_str.lower() == 'nan' and any(pd.isna(val) for val in success_value_options):
                         success_value = np.nan
                    elif pd.api.types.is_numeric_dtype(original_dtype) and not pd.api.types.is_bool_dtype(original_dtype):
                        try:
                            success_value = original_dtype.type(success_value_str)
                        except ValueError: 
                            success_value = success_value_str 
                    elif pd.api.types.is_bool_dtype(original_dtype):
                         success_value = (success_value_str.lower() == 'true') 
                    else: 
                        success_value = success_value_str
                else:
                    st.warning(f"Could not determine distinct values in outcome column '{outcome_col}' or it is empty. Please check your data.")


            st.markdown("---")
            st.subheader("2. Select Your Control Group")
            control_group_name = None 
            if variation_col:
                variation_names = df[variation_col].unique().tolist()
                if variation_names:
                    control_group_name = st.selectbox(
                        "Select your 'Control Group' name:",
                        options=variation_names,
                        index=0,
                        help="Choose the variation that represents your baseline or original version.",
                        key="control_group_select"
                    )
                else:
                    st.warning(f"No unique variation names found in column '{variation_col}'.")
            
            st.markdown("---")
            alpha_analysis = st.slider("Significance Level (\u03B1) for Analysis (%)", 1, 10, 5, 1, key="alpha_analysis_slider_cycle4",
                                       help="Set the significance level (alpha) for hypothesis testing. Typically 5%.") / 100.0


            if st.button("🚀 Run Frequentist Analysis (Binary Outcome)", key="run_analysis_button_cycle4"):
                if not variation_col or not outcome_col or control_group_name is None or success_value is None: 
                    st.error("Please complete all column mapping, success value identification, and control group selections.")
                else:
                    try:
                        if pd.isna(success_value):
                            df['__converted_binary__'] = df[outcome_col].isna().astype(int)
                        else:
                            df['__converted_binary__'] = (df[outcome_col] == success_value).astype(int)

                        st.subheader("📊 Descriptive Statistics")
                        summary_stats = df.groupby(variation_col).agg(
                            Users=('__converted_binary__', 'count'),
                            Conversions=('__converted_binary__', 'sum')
                        ).reset_index()
                        
                        if summary_stats['Users'].sum() == 0:
                            st.error("No users found after grouping. Please check your column selections or data.")
                        else:
                            summary_stats['Conversion Rate (%)'] = (summary_stats['Conversions'] / summary_stats['Users'].replace(0, np.nan) * 100).round(2)
                            st.dataframe(summary_stats.fillna('N/A (0 Users)'))

                            chart_data = summary_stats.set_index(variation_col)['Conversion Rate (%)'].fillna(0)
                            if not chart_data.empty:
                                st.bar_chart(chart_data)
                            
                            st.subheader(f"📈 Comparison vs. Control ('{control_group_name}')")
                            
                            control_data_rows = summary_stats[summary_stats[variation_col] == control_group_name]
                            if control_data_rows.empty:
                                st.error(f"Control group '{control_group_name}' not found in the data or has no users after processing.")
                            else:
                                control_data = control_data_rows.iloc[0]
                                control_users = control_data['Users']
                                control_conversions = control_data['Conversions']
                                control_cr = control_conversions / control_users if control_users > 0 else 0

                                comparison_results = []
                                for index, row in summary_stats.iterrows():
                                    var_name = row[variation_col]
                                    if var_name == control_group_name:
                                        continue 

                                    var_users = row['Users']
                                    var_conversions = row['Conversions']
                                    var_cr = var_conversions / var_users if var_users > 0 else 0

                                    p_value_display, ci_display, significant_display = 'N/A', 'N/A', 'N/A'
                                    abs_uplift_display = 'N/A'
                                    rel_uplift_display = 'N/A'

                                    if control_users > 0 and var_users > 0:
                                        abs_uplift = var_cr - control_cr
                                        abs_uplift_display = f"{abs_uplift*100:.2f}"
                                        rel_uplift_display = f"{(abs_uplift / control_cr) * 100:.2f}%" if control_cr > 0 else "N/A (Control CR is 0)"
                                        
                                        count = np.array([var_conversions, control_conversions])
                                        nobs = np.array([var_users, control_users])
                                        
                                        if np.any(count < 0) or np.any(nobs <= 0) or np.any(count > nobs):
                                            p_value_display, significant_display = 'N/A (Invalid counts/nobs for test)', 'N/A'
                                            ci_display = 'N/A'
                                        else:
                                            try:
                                                z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
                                                p_value_display = f"{p_value:.4f}"
                                                significant_bool = p_value < alpha_analysis
                                                significant_display = f"Yes (p={p_value:.4f})" if significant_bool else f"No (p={p_value:.4f})"
                                                
                                                ci_low_diff, ci_high_diff = confint_proportions_2indep(
                                                    var_conversions, var_users, 
                                                    control_conversions, control_users, 
                                                    method='wald', alpha=alpha_analysis
                                                )
                                                ci_display = f"[{ci_low_diff*100:.2f}, {ci_high_diff*100:.2f}]"
                                            except Exception as e_test: 
                                                st.warning(f"Could not compute stats for {var_name} vs Control: {e_test}")
                                                p_value_display, ci_display, significant_display = 'Error', 'Error', 'Error'
                                    else:
                                         significant_display = 'N/A (Zero users in control or variation)'

                                    comparison_results.append({
                                        "Variation": var_name,
                                        "Conversion Rate (%)": f"{var_cr*100:.2f}",
                                        "Absolute Uplift (%)": abs_uplift_display,
                                        "Relative Uplift (%)": rel_uplift_display,
                                        "P-value (vs Control)": p_value_display,
                                        f"CI {100*(1-alpha_analysis):.0f}% for Diff. (%)": ci_display,
                                        "Statistically Significant?": significant_display
                                    })
                                
                                if comparison_results:
                                    comparison_df = pd.DataFrame(comparison_results)
                                    st.dataframe(comparison_df)
                                    for _, row_data in comparison_df.iterrows(): 
                                        if "Yes" in str(row_data["Statistically Significant?"]):
                                            st.caption(f"The difference between **{row_data['Variation']}** and control ('{control_group_name}') is statistically significant at the {alpha_analysis*100:.0f}% level. P-value: {row_data['P-value (vs Control)']}. This means there's strong evidence against the null hypothesis (that there's no difference).")
                                        elif "No" in str(row_data["Statistically Significant?"]):
                                             st.caption(f"The difference between **{row_data['Variation']}** and control ('{control_group_name}') is not statistically significant at the {alpha_analysis*100:.0f}% level. P-value: {row_data['P-value (vs Control)']}. This means we don't have enough evidence to reject the null hypothesis (that there's no difference).")
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")
                        st.exception(e) 
        except Exception as e:
            st.error(f"Error reading or processing CSV file: {e}")
            st.exception(e)
    else:
        st.info("Upload a CSV file to begin analysis for Binary Outcomes (Frequentist).")
    
    st.markdown("---")
    st.info("Bayesian analysis and further enhancements coming in Cycle 5!")


def show_interpret_results_page():
    st.header("Interpreting Results & Detailed Decision Guidance 🧐")
    st.write("Guidance on how to interpret your A/B test results and make decisions will be implemented in Cycle 9.")
    st.info("Coming soon: Understanding statistical vs. practical significance, next steps!")

def show_faq_page():
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

def show_roadmap_page():
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
