import streamlit as st

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

# --- Page Functions ---

def show_introduction_page():
    st.header("Introduction to A/B Testing 🧪")
    st.markdown("Welcome to the A/B Testing Guide & Analyzer! This tool is designed to help beginners understand and conduct A/B tests effectively.")
    st.markdown("---")

    st.subheader("What is A/B Testing? (The Elevator Pitch)")
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
    st.markdown("Here are a few key terms you'll encounter frequently. We'll explain them in more detail as they come up in the app.")
    
    basic_terms = {
        "Control (Version A)": "The existing, unchanged version that you're comparing against. It acts as a baseline.",
        "Variation (Version B, C, etc.)": "A modified version that you're testing to see if it performs differently than the control.",
        "Conversion / Goal / Metric": "The specific action or outcome you are measuring to determine success (e.g., a sign-up, a purchase, a click).",
        "Conversion Rate (CR)": "The percentage of users who complete the desired goal, out of the total number of users in that group."
    }

    for term, definition in basic_terms.items():
        st.markdown(f"**{term}:** {definition}")

    with st.expander("📖 Learn more about other common A/B testing terms... (Coming in a future cycle!)"):
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
    2.  📐 **Design Your Test & Calculate Sample Size:** Determine how many users you need for a reliable test. (➡️ *Our "Designing Your A/B Test" section will help here!*)
    3.  🚀 **Run Your Test & Collect Data:** Implement the test and gather data on how each variation performs. (This step happens on your platform/website.)
    4.  📊 **Analyze Your Results:** Process the collected data to compare the performance of your variations. (➡️ *Our "Analyze Results" section is built for this!*)
    5.  🧐 **Interpret Results & Make a Decision:** Understand what the results mean and decide on the next steps. (➡️ *Our "Interpreting Results & Detailed Decision Guidance" section will guide you.*)
    """)
    st.markdown("---")
    
    st.subheader("Where This App Fits In")
    st.markdown("""
    This application aims to be your companion for the critical stages of A/B testing:
    * Helping you **design robust tests** by calculating the necessary sample size.
    * Enabling you to **analyze the data** you've collected using both Frequentist and Bayesian statistical approaches.
    * Guiding you in **interpreting those results** to make informed, data-driven decisions.
    * Providing **educational content** (like common pitfalls and FAQs) to improve your A/B testing knowledge.
    
    We're excited to help you on your A/B testing journey!
    """)

def show_design_test_page():
    st.header("Designing Your A/B Test 📐")
    st.write("Content for designing your test, including the Sample Size Calculator, will be implemented in Cycle 2.")
    st.info("Coming soon: Sample Size Calculator and more!")

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
    st.markdown("We have many exciting features planned! Here's a look at what's on our backburner for future development:")
    
    if FUTURE_FEATURES:
        for feature, description in FUTURE_FEATURES.items():
            st.markdown(f"- **{feature}:** {description}")
    else:
        st.write("No future features currently listed.")
    st.markdown("---")
    st.markdown("Your feedback can help us prioritize! Let us know which of these would be most valuable to you.")


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

# Display the selected page
page_function = PAGES[selection]
page_function()

st.sidebar.markdown("---")
st.sidebar.info("A/B Testing Guide & Analyzer | V0.1 (Cycle 1)")
