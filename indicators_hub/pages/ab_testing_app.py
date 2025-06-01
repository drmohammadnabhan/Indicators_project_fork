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
        if p2 >=1:
             return None, f"MDE ({mde_abs*100:.2f}%) results in a target conversion rate of {p2*100:.2f}% or more, which is not possible. Adjust BCR or MDE."
        if p2 <=0:
             return None, f"MDE ({mde_abs*100:.2f}%) results in a target conversion rate of {p2*100:.2f}% or less, which is not possible. Adjust BCR or MDE."

    z_alpha_half = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    variance_p1 = p1 * (1 - p1)
    variance_p2 = p2 * (1 - p2)
    numerator = (z_alpha_half + z_beta)**2 * (variance_p1 + variance_p2)
    denominator = mde_abs**2
    
    if denominator == 0: 
        return None, "MDE cannot be zero (already checked by mde_abs > 0)."

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

def show_design_test_page():
    st.header("Designing Your A/B Test 📐")
    st.markdown("A crucial step in designing an A/B test is determining the appropriate sample size. This calculator will help you estimate the number of users needed per variation for tests with **binary outcomes** (e.g., conversion rates, click-through rates).")
    st.markdown("---")
    
    st.subheader("Sample Size Calculator (for Binary Outcomes)")
    # ... (Sample Size Calculator code from Cycle 2 - V0.2.2 - remains here) ...
    # For brevity in this response, I'm not repeating the full calculator UI code here, 
    # but it should be the same as the end of Cycle 2.
    # Assume the full calculator code (inputs, button, results, formula expander, impacts expander) is here.
    
    # --- Placeholder for Calculator UI from previous cycle ---
    st.markdown("**Calculator Inputs:**")
    cols = st.columns(2)
    with cols[0]:
        baseline_cr_percent = st.number_input(
            label="Baseline Conversion Rate (BCR) (%) ", # Added space for uniqueness if needed
            min_value=0.1, max_value=99.9, value=5.0, step=0.1, format="%.1f",
            help="The current conversion rate of your control group (Version A). For example, if 5 out of 100 users convert, your BCR is 5%."
        )
    with cols[1]:
        mde_abs_percent = st.number_input(
            label="Minimum Detectable Effect (MDE) - Absolute (%) ", # Added space
            min_value=0.1, max_value=50.0, value=1.0, step=0.1, format="%.1f",
            help="The smallest *absolute* improvement you want to detect (e.g., a 1% absolute increase from BCR). A smaller MDE requires a larger sample size."
        )
    cols2 = st.columns(2)
    with cols2[0]:
        power_percent = st.slider(
            label="Statistical Power (1 - \u03B2) (%) ", # Added space
            min_value=50, max_value=99, value=80, step=1, format="%d%%",
            help="The probability of detecting an effect if there is one (typically 80-90%). Higher power reduces the chance of a false negative but requires more samples."
        )
    with cols2[1]:
        alpha_percent = st.slider(
            label="Significance Level (\u03B1) (%) - Two-sided ", # Added space
            min_value=1, max_value=20, value=5, step=1, format="%d%%",
            help="The probability of detecting an effect when there isn't one (typically 1-5%). This is your risk tolerance for a false positive. A two-sided test is assumed."
        )
    num_variations = st.number_input(
        label="Number of Variations (including Control) ", # Added space
        min_value=2, value=2, step=1,
        help="Total number of versions you are testing (e.g., Control + 1 Variation = 2; Control + 2 Variations = 3)."
    )
    st.markdown(
        "<p style='font-size: smaller; font-style: italic;'>Note: This calculator determines sample size based on pairwise comparisons against the control group, each at the specified significance level (\u03B1).</p>", 
        unsafe_allow_html=True
    )
    if st.button("Calculate Sample Size "): # Added space
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
                st.metric(label="Required Sample Size PER Variation ", value=f"{sample_size_per_variation:,}") # Added space
            with res_cols[1]:
                st.metric(label="Total Required Sample Size ", value=f"{(sample_size_per_variation * num_variations):,}") # Added space
            st.markdown(f"""
            **Summary of Inputs Used:**
            - Baseline Conversion Rate (BCR): `{baseline_cr_percent:.1f}%`
            - Absolute MDE: `{mde_abs_percent:.1f}%` (Targeting a CR of at least `{target_cr_percent:.2f}%` for variations)
            - Statistical Power: `{power_percent}%` (1 - \u03B2) 
            - Significance Level (\u03B1): `{alpha_percent}%` (two-sided)
            - Number of Variations: `{num_variations}`
            This means you'll need approximately **{sample_size_per_variation:,} users/observations for your control group** and **{sample_size_per_variation:,} users/observations for each of your other test variations** to confidently detect the specified MDE.
            """)
            with st.expander("Show Formula Used for Sample Size Calculation "): # Added space
                st.markdown("The sample size ($n$) per variation for comparing two proportions is commonly calculated using:")
                st.latex(r''' n = \frac{(Z_{\alpha/2} + Z_{\beta})^2 \cdot (p_1(1-p_1) + p_2(1-p_2))}{(p_2 - p_1)^2} ''')
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
    with st.expander("💡 Understanding Input Impacts on Sample Size "): # Added space
        st.markdown(r"""
        Adjusting the input parameters for the sample size calculator has direct consequences on the number of users you'll need... 
        * **Baseline Conversion Rate (BCR):** ...
        * **Minimum Detectable Effect (MDE):** ...
        * **Statistical Power (1 - $\beta$):** ...
        * **Significance Level ($\alpha$):** ...
        * **Number of Variations:** ...
        Balancing these factors is key...
        """) # Ellipses for brevity, actual text from previous cycle is here
    st.markdown("---")
    # --- End of Placeholder for Calculator ---

    st.subheader("Common Pitfalls in A/B Test Design & Execution")
    st.markdown("Avoiding these common mistakes can significantly improve the quality and reliability of your A/B tests.")

    pitfalls = {
        "Too Short Test Duration / Insufficient Sample Size": {
            "what": "Ending a test before collecting enough data (as determined by a sample size calculation) or running it for an arbitrarily short period (e.g., just one day).",
            "problem": "Results will likely be statistically underpowered, meaning you might not detect a real effect (false negative), or observed differences could be due to random noise rather than a true effect.",
            "howto": "Always calculate your required sample size *before* starting the test using a tool like the one above. Run the test until you've reached that sample size per variation. Also, aim to run tests for at least one full week (or business cycle) to account for daily fluctuations.",
            "analogy": "Imagine trying to judge a marathon winner based on who is leading after the first 100 meters. Early leaders can fade, and strong finishers might start slow. You need to let the race run its course to see the true outcome."
        },
        "Ignoring Statistical Significance / Power": {
            "what": "Making decisions based on observed differences without considering if those differences are statistically significant, or running tests without enough power to detect meaningful effects.",
            "problem": "You might implement changes that have no real impact (or even a negative one) if you ignore significance. If power is too low, you might discard truly better variations because the test couldn't reliably detect their superiority.",
            "howto": "Always check the p-value and confidence intervals (Frequentist) or probabilities and credible intervals (Bayesian). Ensure your sample size calculation targets adequate power (typically 80% or higher).",
            "analogy": "Flipping a coin 3 times and getting 2 heads doesn't mean the coin is biased. You need more flips (data) to be confident. Statistical significance helps you determine if your 'win' is more than just a lucky streak."
        },
        "Testing Too Many Things at Once (in a Single Variation)": {
            "what": "Changing multiple elements (e.g., headline, button color, image, and layout) all in one variation compared to the control.",
            "problem": "If the variation wins (or loses), you won't know which specific change caused the difference. It's impossible to isolate the impact of individual elements.",
            "howto": "Test one significant change at a time if you want to understand the 'why' behind performance differences. If you must test a redesigned page, understand that you're testing the 'package' of changes, not individual components.",
            "analogy": "If you take a new medicine and also change your diet and start exercising on the same day, and you feel better, you can't be sure if it was the medicine, the diet, the exercise, or a combination. Each should be evaluated separately if you want to know its specific effect."
        },
        "External Factors Affecting the Test": {
            "what": "Events outside of your test (e.g., holidays, major news, marketing campaigns, competitor actions, site outages) that can influence user behavior and test results.",
            "problem": "These factors can skew results, making one variation appear better or worse due to external influences rather than its own merit. For example, a promotional campaign might temporarily boost conversions for all variations, masking the true difference between them.",
            "howto": "Be aware of the environment during your test. Try to avoid running tests during highly volatile periods if possible. Document any significant external events that occur. If a major event skews data, you might need to discard the affected period or restart the test.",
            "analogy": "Trying to measure plant growth with two different fertilizers while a surprise heatwave (or frost) affects both plants. The extreme weather could be a bigger factor in their growth than the fertilizers themselves."
        },
        "Regression to the Mean": {
            "what": "A statistical phenomenon where an initial extreme result (very high or very low) tends to be followed by results that are closer to the average over time.",
            "problem": "If you stop a test early based on an unusually good (or bad) initial result for a variation, that performance might not hold up in the long run. The early result could have been due to chance.",
            "howto": "Run tests for their planned duration and sample size, even if early results look extreme. This allows performance to stabilize and gives a more accurate picture.",
            "analogy": "A basketball player who hits 5 three-pointers in a row might seem like they'll never miss, but over the course of a full game (or season), their shooting percentage will likely regress closer to their actual average skill level."
        },
        "Not Segmenting Results (When Appropriate)": {
            "what": "Only looking at the overall average result of a test and not exploring how different user segments (e.g., new vs. returning, mobile vs. desktop, different traffic sources) reacted.",
            "problem": "A variation might be a winner overall but perform poorly for a key segment, or vice-versa. An overall 'flat' result could hide significant wins in one segment and significant losses in another, canceling each other out.",
            "howto": "After checking overall results, plan to analyze performance for important, predefined user segments. (This app's 'Analyze Results' section will help with this in a later cycle!). Ensure segments are large enough for meaningful analysis.",
            "analogy": "A new type of music might have mixed reviews overall, but if you segment by age group, you might find teenagers love it while older adults dislike it. The overall average wouldn't tell you this nuanced story."
        },
        "Peeking at Results Too Often (and Stopping Early)": {
            "what": "Constantly monitoring test results and stopping the test as soon as statistical significance is reached, especially if using traditional frequentist methods.",
            "problem": "This practice dramatically increases the chance of a Type I error (false positive). P-values fluctuate, and by watching them and stopping at a 'lucky' moment, you're likely to find a significant result that isn't actually real. This is sometimes called 'p-hacking'.",
            "howto": "Determine your sample size and test duration *in advance*. Avoid making decisions based on interim results unless you are using specific sequential testing methodologies (which have different statistical rules, not covered by standard calculators). Bayesian methods are generally more robust to 'peeking' as they update probabilities continuously.",
            "analogy": "If you keep flipping a coin and stop the moment you get three heads in a row, you might conclude the coin is biased. But if you committed to flipping it 100 times, that early streak might just be chance."
        },
        "Simpson's Paradox": {
            "what": "A phenomenon where a trend appears in several different groups of data but disappears or reverses when these groups are combined.",
            "problem": "Looking only at aggregated data can lead to incorrect conclusions. For example, Variation B might look better than A overall, but when segmented (e.g., by traffic source), A is actually better than B for *every single traffic source*.",
            "howto": "Be aware of potential confounding variables and consider analyzing important segments. If you see a surprising overall result, try to break it down by key groups to ensure the trend holds.",
            "analogy": "Imagine two hospitals. Hospital A has a higher overall patient survival rate than Hospital B. However, Hospital A might specialize in less severe cases, while Hospital B takes on more critical, high-risk patients. If you look *within* each risk category (e.g., low-risk patients, high-risk patients), Hospital B might actually have a better survival rate for *both* groups. The overall average was misleading due to the different patient mixes."
        }
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
    st.header("Analyze Results 📊")
    st.write("Functionality for uploading data and analyzing A/B test results will be implemented starting in Cycle 4.")
    st.info("Coming soon: Data upload, Frequentist and Bayesian analysis, segmentation!")

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
            if "example" in details:
                st.markdown(f"**Analogy / Example:** {details['example']}")
    st.markdown("---")
    st.info("Content for this section will be reviewed and expanded as needed.")


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
st.sidebar.info("A/B Testing Guide & Analyzer | V0.3 (Cycle 3)")
