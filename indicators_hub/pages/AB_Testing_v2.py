import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist # For t-distribution (sampling and ppf)
import math # For math.ceil and math.isinf

# --- Helper Function for Bayesian Analysis (Continuous) - ISOLATED FOR TESTING ---
def run_bayesian_continuous_analysis(summary_stats_df, control_group_name, n_samples=10000, ci_level=0.95):
    """
    Performs a simplified Bayesian analysis for continuous outcomes.
    Approximates posteriors of means using t-distributions based on sample stats.
    summary_stats_df needs columns: 'Variation', 'Users', 'Mean_Value', 'Std_Dev'
    """
    st.write("--- Inside run_bayesian_continuous_analysis ---") # Checkpoint 1
    results = {}
    
    # Ensure 'Variation' column exists
    if 'Variation' not in summary_stats_df.columns:
        st.error("`summary_stats_df` must contain a 'Variation' column.")
        return None, "Missing 'Variation' column in summary_stats_df."

    st.write("Input summary_stats_df head:", summary_stats_df.head()) # Checkpoint 2

    for index, row in summary_stats_df.iterrows():
        var_name = row['Variation']
        # Ensure columns exist before trying to access them
        if not all(col in row for col in ['Users', 'Mean_Value', 'Std_Dev']):
            st.error(f"Missing one or more required columns (Users, Mean_Value, Std_Dev) for variation '{var_name}'.")
            results[var_name] = { # Populate with NaNs to avoid KeyErrors later if var_name is used
                'samples': np.array([np.nan] * n_samples), 'posterior_mean_of_mean': np.nan, 
                'mean_ci_low': np.nan, 'mean_ci_high': np.nan, 'df': 0, 'loc': np.nan, 
                'scale': np.nan, 'diff_samples_vs_control': None, 'prob_better_than_control': None,
                'uplift_ci_low': None, 'uplift_ci_high': None, 'expected_uplift_abs': None, 'prob_best': 0.0
            }
            continue

        n = int(row['Users'])
        mean = row['Mean_Value']
        std_dev = row['Std_Dev']
        st.write(f"Processing {var_name}: N={n}, Mean={mean}, SD={std_dev}") # Checkpoint 3

        if n < 2 or pd.isna(mean) or pd.isna(std_dev) or std_dev <= 0:
            st.warning(f"Insufficient data or invalid stats for variation '{var_name}'. N={n}, Mean={mean}, SD={std_dev}. Skipping sampling.")
            results[var_name] = {
                'samples': np.array([np.nan] * n_samples), 
                'posterior_mean_of_mean': np.nan, 'mean_ci_low': np.nan, 
                'mean_ci_high': np.nan, 'df': n-1 if n>=1 else 0, 
                'loc': mean, 'scale': std_dev / np.sqrt(n) if n > 0 and std_dev > 0 else np.nan, 
                'diff_samples_vs_control': None, 'prob_better_than_control': None,
                'uplift_ci_low': None, 'uplift_ci_high': None, 'expected_uplift_abs': None, 'prob_best': 0.0
            }
            continue

        df_t = n - 1
        loc_t = mean
        scale_t = std_dev / np.sqrt(n)
        
        st.write(f"For {var_name}: df_t={df_t}, loc_t={loc_t}, scale_t={scale_t}") # Checkpoint 4
        if scale_t <= 0 or pd.isna(scale_t): # scale must be positive
            st.warning(f"Invalid scale ({scale_t}) for t-distribution for variation '{var_name}'. Skipping sampling.")
            results[var_name] = {
                'samples': np.array([np.nan] * n_samples), 'posterior_mean_of_mean': np.nan,
                'mean_ci_low': np.nan, 'mean_ci_high': np.nan, 'df': df_t, 'loc': loc_t, 'scale': scale_t,
                'diff_samples_vs_control': None, 'prob_better_than_control': None, 
                'uplift_ci_low': None, 'uplift_ci_high': None, 'expected_uplift_abs': None, 'prob_best': 0.0
            }
            continue

        samples = t_dist.rvs(df=df_t, loc=loc_t, scale=scale_t, size=n_samples)
        mean_ci_low = t_dist.ppf((1-ci_level)/2, df=df_t, loc=loc_t, scale=scale_t)
        mean_ci_high = t_dist.ppf(1-(1-ci_level)/2, df=df_t, loc=loc_t, scale=scale_t)

        results[var_name] = {
            'samples': samples, 'posterior_mean_of_mean': np.mean(samples), 
            'mean_ci_low': mean_ci_low, 'mean_ci_high': mean_ci_high, 
            'df': df_t, 'loc': loc_t, 'scale': scale_t, 
            'diff_samples_vs_control': None
        }
    st.write("Finished sampling individual posteriors. Results dict keys:", list(results.keys())) # Checkpoint 5

    if control_group_name not in results or pd.isna(results[control_group_name].get('posterior_mean_of_mean')):
        st.error(f"Control group '{control_group_name}' data insufficient/invalid for Bayesian analysis after sampling.")
        return None, f"Control group '{control_group_name}' data insufficient/invalid."
    
    control_samples = results[control_group_name]['samples']
    if np.all(np.isnan(control_samples)):
        st.error(f"Control group '{control_group_name}' posterior samples are all NaN.")
        return results, f"Control group '{control_group_name}' posterior samples could not be generated."

    st.write("Processing comparisons vs control.") # Checkpoint 6
    for var_name, data in results.items():
        if var_name == control_group_name or np.all(np.isnan(data['samples'])):
            data['prob_better_than_control'] = None; data['uplift_ci_low'] = None; data['uplift_ci_high'] = None; data['expected_uplift_abs'] = None
            continue
        var_samples = data['samples']; diff_samples = var_samples - control_samples 
        data['diff_samples_vs_control'] = diff_samples; data['prob_better_than_control'] = np.mean(diff_samples > 0)
        data['uplift_ci_low'] = np.nanpercentile(diff_samples, (1-ci_level)/2 * 100); data['uplift_ci_high'] = np.nanpercentile(diff_samples, (1-(1-ci_level)/2) * 100)
        data['expected_uplift_abs'] = np.nanmean(diff_samples)

    st.write("Processing P(Best).") # Checkpoint 7
    all_var_names_from_summary = summary_stats_df['Variation'].tolist() # Use this for consistent ordering
    ordered_var_names_in_results = [name for name in all_var_names_from_summary if name in results and not np.all(np.isnan(results[name]['samples']))]
    
    if not ordered_var_names_in_results:
        st.warning("No variations with valid data for P(Best) calculation.")
        for var_name in all_var_names_from_summary: 
            if var_name in results: results[var_name]['prob_best'] = 0.0
            else: results[var_name] = {'prob_best': 0.0} # Should not happen if logic above is correct
        return results, "No variations with valid data for P(Best)."

    all_samples_matrix = np.array([results[var]['samples'] for var in ordered_var_names_in_results])
    best_variation_counts = np.zeros(len(all_var_names_from_summary))

    if all_samples_matrix.ndim == 2 and all_samples_matrix.shape[0] > 0 and all_samples_matrix.shape[1] == n_samples:
        for i in range(n_samples):
            current_iter_samples = all_samples_matrix[:, i]
            if np.all(np.isnan(current_iter_samples)): continue
            best_idx_in_ordered_list = np.nanargmax(current_iter_samples)
            best_var_name_this_iter = ordered_var_names_in_results[best_idx_in_ordered_list]
            if best_var_name_this_iter in all_var_names_from_summary:
                original_idx_for_counts = all_var_names_from_summary.index(best_var_name_this_iter)
                best_variation_counts[original_idx_for_counts] += 1
            
        prob_best = best_variation_counts / n_samples
        for i, var_name in enumerate(all_var_names_from_summary):
            if var_name in results: results[var_name]['prob_best'] = prob_best[i]
            elif var_name not in results : results[var_name] = {'prob_best': 0.0} 
            elif 'prob_best' not in results[var_name]: results[var_name]['prob_best'] = 0.0 
    else: 
        st.warning("Sample matrix for P(Best) calculation is not as expected.")
        for var_name in all_var_names_from_summary:
             if var_name in results: results[var_name]['prob_best'] = 1.0 if len(ordered_var_names_in_results) == 1 and var_name == ordered_var_names_in_results[0] else 0.0
             else: results[var_name] = {'prob_best': 0.0} # Should not happen
    st.write("--- Exiting run_bayesian_continuous_analysis ---") # Checkpoint 8
    return results, None

# --- Streamlit App for Testing ---
st.title("Cycle 7 Debugger: Bayesian Continuous Analysis")

# Create a dummy summary_stats DataFrame (mimicking output from frequentist part)
data = {
    'Variation': ['Control', 'TreatmentA', 'TreatmentB', 'TreatmentC_few_users', 'TreatmentD_zero_sd'],
    'Users': [100, 105, 98, 1, 50], # N for each group
    'Mean_Value': [25.5, 27.1, 26.8, 30.0, 22.0], # Sample Mean for each group
    'Std_Dev': [5.1, 5.3, 5.0, 2.0, 0.0] # Sample Std Dev for each group
}
sample_summary_stats = pd.DataFrame(data)

st.subheader("Sample Input Data (Summary Stats):")
st.dataframe(sample_summary_stats)

control_options = sample_summary_stats['Variation'].tolist()
control_group = st.selectbox("Select Control Group:", control_options, key="control_debug")
ci_level_input = st.slider("Credible Interval Level:", 0.80, 0.99, 0.95, 0.01, key="ci_debug")

if st.button("Run Bayesian Continuous Analysis Test", key="run_debug_button"):
    st.write(f"Control selected: {control_group}")
    st.write(f"CI Level: {ci_level_input}")
    
    bayesian_results, error_msg = run_bayesian_continuous_analysis(
        sample_summary_stats.copy(), # Pass a copy
        control_group,
        ci_level=ci_level_input
    )

    if error_msg:
        st.error(f"Error from Bayesian function: {error_msg}")
    
    if bayesian_results:
        st.subheader("Raw Bayesian Results Dictionary:")
        st.json(bayesian_results) # Display the full results dictionary

        st.subheader("Formatted Table (Example):")
        # Example of how you might format it (similar to the main app)
        display_table = []
        for var_name_disp, res_disp in bayesian_results.items():
            display_table.append({
                "Variation": var_name_disp,
                "Approx. Posterior Mean": f"{res_disp.get('posterior_mean_of_mean', np.nan):.3f}",
                f"{ci_level_input*100:.0f}% CrI for Mean": f"[{res_disp.get('mean_ci_low', np.nan):.3f}, {res_disp.get('mean_ci_high', np.nan):.3f}]",
                "P(Mean > Control Mean) (%)": f"{res_disp.get('prob_better_than_control', 0)*100:.1f}%" if res_disp.get('prob_better_than_control') is not None else "N/A",
                "Expected Diff.": f"{res_disp.get('expected_uplift_abs', np.nan):.3f}" if res_disp.get('expected_uplift_abs') is not None else "N/A",
                "P(Best Mean) (%)": f"{res_disp.get('prob_best', 0)*100:.1f}%"
            })
        st.dataframe(pd.DataFrame(display_table))

        # Example Plotting (Posterior of Means)
        st.subheader("Approx. Posterior Distributions of Group Means")
        fig, ax = plt.subplots()
        min_x_plot, max_x_plot = float('inf'), float('-inf')

        # Determine plot range
        for var_name_plot, res_plot in bayesian_results.items():
            if 'samples' in res_plot and not np.all(np.isnan(res_plot['samples'])):
                min_x_plot = min(min_x_plot, np.nanmin(res_plot['samples']))
                max_x_plot = max(max_x_plot, np.nanmax(res_plot['samples']))
        
        if math.isinf(min_x_plot) or math.isinf(max_x_plot) or pd.isna(min_x_plot) or pd.isna(max_x_plot) : # Fallback range if something went wrong
            min_x_plot, max_x_plot = 0,1 
            if not sample_summary_stats['Mean_Value'].empty:
                 min_x_plot = sample_summary_stats['Mean_Value'].min() - sample_summary_stats['Std_Dev'].mean() if pd.notna(sample_summary_stats['Mean_Value'].min()) else 0
                 max_x_plot = sample_summary_stats['Mean_Value'].max() + sample_summary_stats['Std_Dev'].mean() if pd.notna(sample_summary_stats['Mean_Value'].max()) else 1
                 if pd.isna(min_x_plot) or pd.isna(max_x_plot) or min_x_plot >= max_x_plot : min_x_plot, max_x_plot = 0,1


        x_range_plot = np.linspace(min_x_plot - abs(min_x_plot*0.1), max_x_plot + abs(max_x_plot*0.1), 300)
        max_density_plot = 0

        for var_name_plot, res_plot in bayesian_results.items():
            if res_plot.get('df', 0) > 0 and pd.notna(res_plot.get('scale')) and res_plot.get('scale') > 0 and pd.notna(res_plot.get('loc')):
                pdf_values = t_dist.pdf(x_range_plot, df=res_plot['df'], loc=res_plot['loc'], scale=res_plot['scale'])
                ax.plot(x_range_plot, pdf_values, label=f"{var_name_plot}")
                ax.fill_between(x_range_plot, pdf_values, alpha=0.2)
                if np.any(pdf_values) and not np.all(np.isnan(pdf_values)): max_density_plot = max(max_density_plot, np.nanmax(pdf_values))
        
        if max_density_plot > 0: ax.set_ylim(0, max_density_plot * 1.1)
        else: ax.set_ylim(0,1)
        
        ax.set_title("Approx. Posterior Distributions of Group Means")
        ax.set_xlabel("Mean Value")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig) # Close the figure

else:
    st.info("Click the button to run the test analysis.")
