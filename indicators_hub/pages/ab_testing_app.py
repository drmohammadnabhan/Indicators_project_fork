import streamlit as st
import numpy as np
from scipy.stats import norm
import math
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

# ... (Keep all other functions and setup from the previous V0.4 script:
# st.set_page_config, FUTURE_FEATURES, calculate_binary_sample_size, 
# show_introduction_page, show_design_test_page, show_interpret_results_page, 
# show_faq_page, show_roadmap_page, and the main navigation logic)
# Only the show_analyze_results_page function is updated below.

def show_analyze_results_page():
    st.header("Analyze Your A/B Test Results 📊")
    st.markdown("Upload your A/B test data (as a CSV file) to perform a Frequentist analysis for **binary outcomes**.")
    st.markdown("---")

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

            success_value_options = [] # Initialize
            success_value = None      # Initialize

            if outcome_col:
                unique_outcomes = df[outcome_col].unique()
                if len(unique_outcomes) == 1:
                    st.warning(f"The outcome column '{outcome_col}' only has one value: `{unique_outcomes[0]}`. A binary outcome requires two distinct values for meaningful analysis.")
                    # We can still let them pick it, but analysis might be trivial
                    success_value_options = unique_outcomes 
                elif len(unique_outcomes) > 2:
                    st.warning(f"The outcome column '{outcome_col}' has more than two unique values: `{unique_outcomes}`. For binary analysis, please select the value that represents a 'conversion' or success.")
                    success_value_options = unique_outcomes
                elif len(unique_outcomes) == 2:
                    success_value_options = unique_outcomes
                # else: # No unique outcomes, means column might be empty or all NaN, pandas unique() handles this.
                      # If success_value_options remains empty, the next block won't run.
                
                # *** CORRECTED CONDITION HERE ***
                if len(success_value_options) > 0:
                    success_value_str = st.selectbox(
                        f"Which value in '{outcome_col}' represents a 'Conversion' (Success)?",
                        options=[str(val) for val in success_value_options],
                        index = 0, # Default to the first option
                        help="Select the value that indicates the desired outcome happened."
                    )
                    
                    # Attempt to convert success_value_str back to its original type for comparison
                    original_dtype = df[outcome_col].dtype
                    # Handle NaN explicitly if it's an option and selected
                    if success_value_str.lower() == 'nan' and any(pd.isna(val) for val in success_value_options):
                         success_value = np.nan
                    elif pd.api.types.is_numeric_dtype(original_dtype) and not pd.api.types.is_bool_dtype(original_dtype):
                        try:
                            success_value = original_dtype.type(success_value_str)
                        except ValueError: # Fallback if direct type conversion fails (e.g. float string for int column)
                            success_value = success_value_str 
                    elif pd.api.types.is_bool_dtype(original_dtype):
                         success_value = (success_value_str.lower() == 'true') # Handles 'True' or 'False' strings
                    else: # String or other non-numeric, non-bool
                        success_value = success_value_str
                else:
                    st.warning(f"Could not determine distinct values in outcome column '{outcome_col}' or it is empty.")


            st.markdown("---")
            st.subheader("2. Select Your Control Group")
            control_group_name = None # Initialize
            if variation_col:
                variation_names = df[variation_col].unique().tolist()
                if variation_names:
                    control_group_name = st.selectbox(
                        "Select your 'Control Group' name:",
                        options=variation_names,
                        index=0,
                        help="Choose the variation that represents your baseline or original version."
                    )
                else:
                    st.warning(f"No unique variation names found in column '{variation_col}'.")
            
            st.markdown("---")
            alpha_analysis = st.slider("Significance Level (α) for Analysis (%)", 1, 10, 5, 1, key="alpha_analysis_slider",
                                       help="Set the significance level (alpha) for hypothesis testing. Typically 5%.") / 100.0


            if st.button("🚀 Run Frequentist Analysis (Binary Outcome)", key="run_analysis_button_cycle4"):
                if not variation_col or not outcome_col or control_group_name is None or success_value is None: # success_value check added
                    st.error("Please complete all column mapping, success value identification, and control group selections.")
                else:
                    try:
                        # Create a binary 'converted' column based on user's selection
                        # Handle potential NaN in success_value for comparison
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
                            return # Exit if no data

                        summary_stats['Conversion Rate (%)'] = (summary_stats['Conversions'] / summary_stats['Users'].replace(0, np.nan) * 100).round(2) # Avoid division by zero
                        st.dataframe(summary_stats.fillna('N/A (0 Users)')) # Show N/A if CR couldn't be calculated

                        if not summary_stats.empty:
                             # Ensure 'Variation ID' is the index for bar_chart if that's how you want it labeled
                             chart_data = summary_stats.set_index(variation_col)['Conversion Rate (%)'].fillna(0) # Fill NaN with 0 for plotting
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
                                    
                                    # Check for invalid counts/nobs before ztest
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
                                        except Exception as e_test: # Catch errors from ztest or confint
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
                                for _, row in comparison_df.iterrows(): # Changed from comparison_df.iterrows() to iterate over the DataFrame
                                    if "Yes" in str(row["Statistically Significant?"]):
                                        st.caption(f"The difference between **{row['Variation']}** and control ('{control_group_name}') is statistically significant at the {alpha_analysis*100:.0f}% level. P-value: {row['P-value (vs Control)']}. This means there's strong evidence against the null hypothesis (that there's no difference).")
                                    elif "No" in str(row["Statistically Significant?"]):
                                         st.caption(f"The difference between **{row['Variation']}** and control ('{control_group_name}') is not statistically significant at the {alpha_analysis*100:.0f}% level. P-value: {row['P-value (vs Control)']}. This means we don't have enough evidence to reject the null hypothesis (that there's no difference).")
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

# --- Main app structure using PAGES dictionary ---
# Ensure all other page functions (show_introduction_page, show_design_test_page, etc.)
# are defined as in the previous V0.4/V0.3/V0.2.2 versions.
# For brevity, I'm not re-pasting them here if they are unchanged from the last full script provided.

# If you are replacing the whole script, ensure you have the definitions for:
# show_introduction_page()
# show_design_test_page()
# calculate_binary_sample_size() - already included above
# show_interpret_results_page()
# show_faq_page()
# show_roadmap_page()
# And the main navigation logic:
# PAGES = { ... }
# selection = st.sidebar.radio(...)
# page_function = PAGES[selection]
# page_function()
# st.sidebar.info(...)
