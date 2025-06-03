import streamlit as st
import numpy as np # Make sure numpy is imported if not already
import pandas as pd # Make sure pandas is imported if not already

# Ensure other necessary imports like math, norm, ttest_ind, beta_dist, plt are at the top of your main script.

def show_faq_page(): # CORRECTED AND FULL FAQ CONTENT
    st.header("FAQ on Common Misinterpretations ❓")
    st.markdown("This section addresses some common questions and misinterpretations that arise when looking at A/B test results.")

    faqs = {
        "Q: My p-value is 0.06 (or just above 0.05). Does this mean my variation *almost* worked or is 'on the verge' of significance?": {
            "answer": """Not exactly. In traditional hypothesis testing, a result is either statistically significant (if p < α, your chosen significance level) or it isn't. A p-value of 0.06 means that *if there were truly no difference between your variations (the null hypothesis is true)*, you'd see data as extreme as yours, or more extreme, about 6% of the time due to random chance alone. It doesn't imply 'almost significant' in a way that suggests a slightly larger sample would guarantee significance. It simply means you didn't meet your pre-defined threshold for rejecting the null hypothesis.""",
            "example": """Think of it like a high jump. If the bar is set at 2 meters (your significance level), and you jump 1.98 meters, you didn't clear the bar. You were close, but you didn't clear it. The p-value reflects the evidence against the null hypothesis; a p-value of 0.06 is weaker evidence than a p-value of 0.01."""
        },
        "Q: If a test isn't statistically significant, does it mean there's no difference between my variations?": {
            "answer": """No, not necessarily. A non-significant result means your test did not find *sufficient evidence* to conclude that a difference exists (at your chosen significance level and with your current sample size/power). A real difference might still exist, but it could be: 
1. Smaller than the Minimum Detectable Effect (MDE) your test was powered to find. 
2. Your test might have had low statistical power, making it hard to detect a true difference even if it was reasonably large. 
3. There truly is no meaningful difference, or the difference is negligible.""",
            "example": """Imagine looking for a specific type of small fish in a large, murky pond with a small net. If you don't catch any, it doesn't mean the fish aren't there. Your net (test power/sample size) might have been too small, or the fish too few or too hard to see (small effect size)."""
        },
        "Q: My A/B test showed Variation B was significantly better. Why did my overall conversion rate drop after I implemented it to 100% of users?": {
            "answer": """This can be frustrating and can happen for several reasons: 
1. **Regression to the Mean:** The performance observed during the specific test period might have been an overestimation of the true long-term effect. 
2. **Novelty Effect or Scarcity Effect During Test:** Users might have reacted positively (or negatively) to the change simply because it was new, or if the test implied a limited-time offer. This effect might wear off post-launch. 
3. **Segmentation Issues (Simpson's Paradox):** The variation might have performed well for a large segment during the test, but if the overall traffic mix changes post-launch, or if it performed poorly for other crucial segments, the overall result could differ. 
4. **External Factors:** Were there different market conditions, campaigns, or site issues post-launch compared to the test period? 
5. **Type I Error (False Positive):** Even with a 5% significance level, there's a 1 in 20 chance that a statistically significant result is due to random chance. 
6. **Implementation Issues:** Was the winning variation implemented *exactly* as it was tested? Any small differences in code or UX could alter performance.""",
            "example": """A new song might shoot up the charts initially due to hype (novelty effect), but its long-term popularity (post-launch performance) might be lower once the initial excitement fades."""
        },
        "Q: Can I combine results from two separate A/B tests (e.g., run on different weeks) to get a larger sample size?": {
            "answer": """Generally, this is not recommended. A/B tests rely on the principle of comparing variations under the *same conditions* at the *same time*. If you run tests at different times, user behavior, traffic sources, seasonality, or other external factors could be different between the two periods, making a direct combination of data statistically invalid and potentially misleading.""",
            "example": """Trying to combine data from a lemonade stand's sales on a hot sunny week with sales data from a cold, rainy week. The conditions are too different to fairly compare or combine the results as if they were from one single experiment."""
        },
        "Q: Is a 200% lift with a small sample size (e.g., 100 users) more impressive than a 10% lift with a large sample size (e.g., 100,000 users)?": {
            "answer": """Not necessarily. While a 200% lift sounds dramatic, results from very small sample sizes are highly volatile and have wide confidence/credible intervals. This means the 'true' lift could be much lower, much higher, or even negative. A 10% lift observed with a large sample size is likely to be much more stable, reliable, and closer to the true underlying effect. Always look at the confidence/credible intervals and the statistical significance, not just the point estimate of the lift.""",
            "example": """If one person buys a $100 item from a new 2-person visitor group, that's a 50% conversion rate and huge revenue per visitor for that tiny sample. If 1,000 people buy a $5 item from a 100,000 visitor group, the overall impact is much larger and the metrics are more reliable, even if the per-item value is smaller."""
        },
        "Q: My Bayesian test shows P(B>A) = 92%. Does this mean there's a 92% chance I'll see this exact observed uplift if I roll it out?": {
            "answer": """No. P(B>A) = 92% means there's a 92% probability that the *true underlying parameter* of Variation B (e.g., its true long-term conversion rate) is greater than the true underlying parameter of Variation A. The actual uplift you observe in any given period (during the test or post-rollout) will still have some variability. The P(B>A) gives you confidence in the *direction* of the effect. To understand the *magnitude* of the potential uplift, you should look at the posterior distribution for the difference or the credible interval for the lift.""",
            "example": """If a weather forecast says there's a 92% chance of rain, it means it's very likely to rain. It doesn't tell you exactly *how much* it will rain (the magnitude). For that, you'd look at other parts of the forecast, like '0.5 to 1 inch expected'."""
        },
        "Q: What if my control group's conversion rate in the test is very different from its historical average?": {
            "answer": """This is a good flag to investigate. Possible reasons include: 
1. **Seasonality/Trends:** User behavior changes over time. 
2. **Different Traffic Mix:** The users in your test period might be different from your historical average (e.g., more mobile users, different marketing channels driving traffic). 
3. **Instrumentation Error:** Double-check your tracking and data collection for the test. 
4. **Actual Change in Baseline:** Something fundamental might have changed on your site or in the market. 
While the A/B test still validly compares variations *within the test period*, a significant shift in the baseline might make it harder to extrapolate the observed lift to long-term performance if the conditions causing the shift don't persist.""",
            "example": """If your ice cream shop's historical daily average sales are 100 cones, but during a week-long new flavor test it's only 50 cones (perhaps due to cold weather), the *percentage lift* of a new flavor might still be calculable against that week's 50-cone baseline, but predicting future sales based on that lift needs to account for the unusual baseline."""
        },
         "Q: The A/B/n test shows Variation C is best overall. Can I just assume it's also significantly better than Variation B without looking at that specific comparison?": {
            "answer": """Not always safely. While C might have the highest overall metric or highest probability of being best, the difference between C and B might be very small and not statistically significant (or the probability C>B might be low). It's good practice to look at key pairwise comparisons, especially between your top-performing variations, to understand the nuances. For example, C might be best, but B might be almost as good and much easier/cheaper to implement.""",
            "example": """In a race, even if a runner finishes first, their margin over the second-place runner could be a fraction of a second (not a decisive win) or several seconds (a clear win). You'd look at the gap to understand the true performance difference."""
        }
    }
    for question, details in faqs.items():
        with st.expander(question):
            st.markdown(f"**A:** {details['answer']}")
            if "example" in details: st.markdown(f"**Analogy / Example:** {details['example']}")
    st.markdown("---")
    st.info("Content for this section will be reviewed and expanded as needed.")
    reviewed and expanded as needed.")

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
