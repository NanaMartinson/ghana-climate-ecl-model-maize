# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 02:38:32 2025

@author: Nana Nshira Martinson
"""

# Ghana Climate Risk ECL Model
# Climate-Adjusted Expected Credit Loss for Agricultural Lending

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("GHANA CLIMATE RISK ECL MODEL")
print("$100M Agricultural Portfolio | 2004-2023\n")

# FILE PATHS - NOTE: THESE PATHS MUST BE UPDATED TO YOUR LOCAL ENVIRONMENT

AGRIC_DATA_PATH = "FAOSTAT Ghana Maize Production Data (2004 - 2023).xls"
CLIMATE_DATA_PATH = "Nasa Power Data Ghana 2004-2024 (Rainfall, Insolation, Temperature).xlsx"

# STEP 1: LOAD DATA
print("STEP 1: Loading Data")

# Agricultural data
try:
    # Attempt to read as .xls (xlrd for older files)
    df_agric = pd.read_excel(AGRIC_DATA_PATH, engine='xlrd')
except:
    # Fallback to openpyxl for newer .xlsx files or issues with xlrd
    df_agric = pd.read_excel(AGRIC_DATA_PATH, engine='openpyxl')

df_agric.columns = df_agric.columns.str.strip()
col_map = {}
for col in df_agric.columns:
    if 'year' in col.lower():
        col_map[col] = 'Year'
    elif 'production' in col.lower():
        col_map[col] = 'Production_Quantity'
    elif 'yield' in col.lower():
        col_map[col] = 'Yield_kg_ha'

df_agric = df_agric.rename(columns=col_map)
df_agric['Year'] = pd.to_numeric(df_agric['Year'], errors='coerce')
df_agric['Production_Quantity'] = pd.to_numeric(df_agric['Production_Quantity'], errors='coerce')
df_agric['Yield_kg_ha'] = pd.to_numeric(df_agric['Yield_kg_ha'], errors='coerce')
df_agric['Yield_tonnes_ha'] = df_agric['Yield_kg_ha'] / 1000
# Recalculate area using the converted tonnes
df_agric['Area_ha'] = df_agric['Production_Quantity'] / df_agric['Yield_tonnes_ha']
df_agric = df_agric[(df_agric['Year'] >= 2004) & (df_agric['Year'] <= 2023)].reset_index(drop=True)

print(f"Production: {df_agric['Production_Quantity'].min():,.0f} - {df_agric['Production_Quantity'].max():,.0f} tonnes")
print(f"Area: {df_agric['Area_ha'].iloc[0]:,.0f} → {df_agric['Area_ha'].iloc[-1]:,.0f} ha (+{((df_agric['Area_ha'].iloc[-1]/df_agric['Area_ha'].iloc[0]-1)*100):.1f}%)\n")

# Climate data
# Insolation data is loaded but not used in the final model features list due to a high correlation with Temperature
insolation_df = pd.read_excel(CLIMATE_DATA_PATH, sheet_name='Insolation')
temperature_df = pd.read_excel(CLIMATE_DATA_PATH, sheet_name='Temperature')
precipitation_df = pd.read_excel(CLIMATE_DATA_PATH, sheet_name='Precipitation')

insolation_df = insolation_df[(insolation_df['YEAR'] >= 2004) & (insolation_df['YEAR'] <= 2023)]
temperature_df = temperature_df[(temperature_df['YEAR'] >= 2004) & (temperature_df['YEAR'] <= 2023)]
precipitation_df = precipitation_df[(precipitation_df['YEAR'] >= 2004) & (precipitation_df['YEAR'] <= 2023)]

month_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
major_season = ['APR', 'MAY', 'JUN', 'JUL']
minor_season = ['SEP', 'OCT', 'NOV']

climate_features_list = []
for year in range(2004, 2024):
    temp_year = temperature_df[temperature_df['YEAR'] == year]
    precip_year = precipitation_df[precipitation_df['YEAR'] == year]
    if len(temp_year) == 0:
        continue
    
    temp_monthly = temp_year[month_cols].mean(axis=0)
    precip_monthly = precip_year[month_cols].mean(axis=0)
    
    features = {
        'Year': year,
        'Major_Temp': temp_monthly[major_season].mean(),
        'Major_Precip': precip_monthly[major_season].sum(),
        'Minor_Precip': precip_monthly[minor_season].sum(),
        'Early_Precip': precip_monthly[['APR', 'MAY']].sum(),
        'Flowering_Temp': temp_monthly['JUN'],
    }
    climate_features_list.append(features)

df_climate = pd.DataFrame(climate_features_list)
print(f"Climate data: {len(df_climate)} years\n")

# Merge
df = df_climate.merge(df_agric[['Year', 'Production_Quantity', 'Yield_tonnes_ha', 'Area_ha']], on='Year', how='inner')

# STEP 2: DROUGHT IDENTIFICATION
print("STEP 2: Drought Identification (Based on Yield)")

# Calculate yield rank for drought identification
df['Yield_Pct'] = df['Yield_tonnes_ha'].rank(pct=True) * 100
df['Major_Temp_Pct'] = df['Major_Temp'].rank(pct=True) * 100
# Drought defined by low yield OR high temperature
df['Is_Drought'] = (df['Yield_Pct'] <= 20) | (df['Major_Temp_Pct'] >= 80)
drought_years = df[df['Is_Drought']]['Year'].tolist()
print(f"Drought years (based on yield/temp): {drought_years}\n")


# STEP 3: BUILD YIELD MODEL (Strategic Change: Modeling YIELD instead of PRODUCTION)
print("STEP 3: Build YIELD Model (to capture stronger climate signal)")

df['Year_Index'] = df['Year'] - df['Year'].min()
# ----------------------------------------------------------------------
# 3.1 TRENDING
slope_y, int_y, r_y, _, _ = stats.linregress(df['Year_Index'], df['Yield_tonnes_ha'])
df['Yield_Trend'] = int_y + slope_y * df['Year_Index']
df['Yield_Detrend'] = df['Yield_tonnes_ha'] - df['Yield_Trend']

print(f"Yield Trend: +{slope_y:.3f} tonnes/ha/year (R² = {r_y**2:.3f})")

# 3.2 CLIMATE LINKAGE (Ridge Regression)
climate_vars = ['Major_Temp', 'Major_Precip', 'Minor_Precip', 'Early_Precip', 'Flowering_Temp']
X_y = df[climate_vars]
y_y = df['Yield_Detrend'] # Target is now DETRENDED YIELD 
scaler_y = StandardScaler()
X_y_scaled = scaler_y.fit_transform(X_y)

model_y = Ridge(alpha=5.0, random_state=42)
model_y.fit(X_y_scaled, y_y)

# 3.3 PREDICTION & CONTRIBUTION
y_pred_y = model_y.predict(X_y_scaled)
df['Yield_Pred'] = df['Yield_Trend'] + y_pred_y
r2_y_full = r2_score(df['Yield_tonnes_ha'], df['Yield_Pred'])

# *** CRITICAL FIX: Calculating contribution relative to DETRENDED VARIANCE 
var_climate_y = np.var(y_pred_y)
var_detrend_y = np.var(df['Yield_Detrend']) # Variance of the target variable
contrib_y_corrected = var_climate_y / var_detrend_y 
# This now shows what % of the year-to-year variation is explained by climate features.

print(f"Combined Yield R²: {r2_y_full:.3f}")
print(f"Climate contribution to DETRENDED YIELD variance: {contrib_y_corrected*100:.1f}%\n")
# ----------------------------------------------------------------------

# STEP 4: PORTFOLIO
print("STEP 4: Portfolio Setup")

portfolio = {
    'Total': 100_000_000,
    'Short_Term': {'amt': 40_000_000, 'term': 1.5},
    'Medium_Term': {'amt': 40_000_000, 'term': 3.0},
    'Long_Term': {'amt': 20_000_000, 'term': 7.0}
}

annual_pd = 0.0678
lgd_full = 0.231
lgd_with_girsal = 0.231 * 0.30
girsal_rate = 0.50

for name, info in portfolio.items():
    if name == 'Total':
        continue
    lifetime_pd = 1 - (1 - annual_pd) ** info['term']
    info['pd'] = lifetime_pd
    print(f"{name}: ${info['amt']/1e6:.0f}M, PD={lifetime_pd:.2%}")

# STEP 5: PD MULTIPLIER FRAMEWORK (Now based on YIELD Deviation)
print("\nSTEP 5: PD Multiplier Framework (Based on Yield Deviation)")

def get_pd_multiplier_trend_relative(predicted_yield, expected_trend_yield, historical_std_yield):
    """
    Calculate PD multiplier based on continuous yield deviation from expected trend.
    
    Linear relationship: Each 1 standard deviation below trend increases PD by 15% 
    (Aggressive stress test)
    """
    deviation = predicted_yield - expected_trend_yield
    z_score = deviation / historical_std_yield
    
    # CRITICAL CHANGE: Using 15% increase per standard deviation for aggressive stress-testing
    if z_score < 0:
        multiplier = 1.00 + abs(z_score) * 0.15 
        # Adjusted thresholds for better categorization with aggressive 15% sensitivity
        if z_score <= -0.75: 
            risk_category = "High Risk"
        elif z_score <= -0.25:
            risk_category = "Elevated Risk"
        else:
            risk_category = "Moderate Risk"
    else:
        multiplier = 1.00
        risk_category = "Normal Risk"
    
    return multiplier, risk_category, z_score

print("Framework: Continuous PD adjustment based on YIELD deviation from trend")
print("  Each 1 std dev below trend → +15% PD increase (Aggressive Stress Test)")
print("  Linear relationship: Multiplier = 1.00 + |z_score| × 0.15\n")

# STEP 6: SCENARIOS (Yield-Based Projection)
print("STEP 6: Climate Scenarios (Yield-Based Projection)")

years_to_2024 = 2024 - df['Year'].max()
expected_yield_2024 = df['Yield_Trend'].iloc[-1] + slope_y * years_to_2024

years_to_2030 = 2030 - df['Year'].max()
expected_yield_2030 = df['Yield_Trend'].iloc[-1] + slope_y * years_to_2030

print(f"Expected Yield (based on trend):")
print(f"  2024: {expected_yield_2024:.3f} tonnes/ha")
print(f"  2030: {expected_yield_2030:.3f} tonnes/ha\n")

historical_std_yield = df['Yield_tonnes_ha'].std()

scenarios = {
    'Baseline_2024': {
        'Major_Temp': df['Major_Temp'].mean(),
        'Major_Precip': df['Major_Precip'].mean(),
        'Minor_Precip': df['Minor_Precip'].mean(),
        'Early_Precip': df['Early_Precip'].mean(),
        'Flowering_Temp': df['Flowering_Temp'].mean(),
        'expected': expected_yield_2024,
    },
    'Moderate_Drought': {
        'Major_Temp': df['Major_Temp'].quantile(0.80),
        'Major_Precip': df['Major_Precip'].quantile(0.25),
        'Minor_Precip': df['Minor_Precip'].quantile(0.30),
        'Early_Precip': df['Early_Precip'].quantile(0.20),
        'Flowering_Temp': df['Flowering_Temp'].quantile(0.85),
        'expected': expected_yield_2024,
    },
    'Severe_Drought': {
        'Major_Temp': df['Major_Temp'].max() + 0.5,
        'Major_Precip': df['Major_Precip'].min() * 0.90,
        'Minor_Precip': df['Minor_Precip'].min() * 0.85,
        'Early_Precip': df['Early_Precip'].min() * 0.85,
        'Flowering_Temp': df['Flowering_Temp'].max() + 0.5,
        'expected': expected_yield_2024,
    },
    'Climate_2030': {
        'Major_Temp': df['Major_Temp'].mean() + 2.0,
        'Major_Precip': df['Major_Precip'].mean() * 0.75,
        'Minor_Precip': df['Minor_Precip'].mean() * 0.70,
        'Early_Precip': df['Early_Precip'].mean() * 0.70,
        'Flowering_Temp': df['Flowering_Temp'].mean() + 2.5,
        'expected': expected_yield_2030,
    }
}

scenario_results = {}

for sc_name, sc_cond in scenarios.items():
    sc_df = pd.DataFrame([{k: v for k, v in sc_cond.items() if k != 'expected'}])
    sc_scaled = scaler_y.transform(sc_df[climate_vars])
    deviation = model_y.predict(sc_scaled)[0]
    
    expected_base_yield = sc_cond['expected']
    predicted_yield = expected_base_yield + deviation
    predicted_yield = max(0.5, predicted_yield) # Set a floor for yield

    # Calculate Z-score and Multiplier based on YIELD
    mult, risk, z = get_pd_multiplier_trend_relative(predicted_yield, expected_base_yield, historical_std_yield)
    
    scenario_results[sc_name] = {
        'yield_t_ha': predicted_yield,
        'expected_yield': expected_base_yield,
        'yield_shortfall': predicted_yield - expected_base_yield,
        'multiplier': mult,
        'risk': risk,
        'z': z
    }

print("Scenario Results (Yield-Based):")
for sc_name, res in scenario_results.items():
    print(f"\n{sc_name}:")
    print(f"  Yield: {res['yield_t_ha']:.3f} t/ha")
    print(f"  Expected: {res['expected_yield']:.3f} t/ha")
    print(f"  Shortfall: {res['yield_shortfall']:.3f} t/ha")
    print(f"  Z-score: {res['z']:.2f}")
    print(f"  PD Multiplier: {res['multiplier']:.2f}x ({res['risk']})")

# STEP 7: ECL CALCULATION
print("\n\nSTEP 7: ECL Calculation - WITH vs WITHOUT GIRSAL\n")

def calc_ecl_with_girsal(loan_amt, pd, mult, girsal, lgd_ins, lgd_unins):
    """Calculate ECL with GIRSAL insurance coverage"""
    adj_pd = pd * mult
    insured = loan_amt * girsal
    uninsured = loan_amt * (1 - girsal)
    return insured * adj_pd * lgd_ins + uninsured * adj_pd * lgd_unins

def calc_ecl_without_girsal(loan_amt, pd, mult, lgd):
    """Calculate ECL without insurance"""
    return loan_amt * pd * mult * lgd

ecl_with = {}
ecl_without = {}
ecl_sensitivity_data = {}

def run_ecl_calculation(scenario_name, multiplier):
    """Calculates ECL totals for a given scenario multiplier."""
    total_with = 0
    total_without = 0
    
    # Print header for the scenario detail table
    print(f"{scenario_name} (PD Multiplier: {multiplier:.2f}x):")
    print(f"{'Loan':<15} {'With GIRSAL':>15} {'Without GIRSAL':>18} {'Difference':>15}")
    
    for loan_name, loan_info in portfolio.items():
        if loan_name == 'Total':
            continue
        
        ecl_w = calc_ecl_with_girsal(loan_info['amt'], loan_info['pd'], multiplier, 
                                        girsal_rate, lgd_with_girsal, lgd_full)
        ecl_wo = calc_ecl_without_girsal(loan_info['amt'], loan_info['pd'], multiplier, lgd_full)
        
        total_with += ecl_w
        total_without += ecl_wo
        
        print(f"{loan_name:<15} ${ecl_w/1e6:>14.2f}M ${ecl_wo/1e6:>17.2f}M ${(ecl_wo-ecl_w)/1e6:>14.2f}M")
    
    print(f"{'TOTAL':<15} ${total_with/1e6:>14.2f}M ${total_without/1e6:>17.2f}M ${(total_without-total_with)/1e6:>14.2f}M\n")
    
    ecl_with[scenario_name] = total_with
    ecl_without[scenario_name] = total_without
    
for sc_name, res in scenario_results.items():
    run_ecl_calculation(sc_name, res['multiplier'])


# GIRSAL SENSITIVITY ANALYSIS
print("\nGIRSAL SENSITIVITY ANALYSIS: Coverage Impact on ECL\n")

coverage_rates = [0.00, 0.25, 0.50, 0.75, 1.00]

def run_sensitivity(scenario_name, multiplier):
    """Runs and prints the GIRSAL coverage sensitivity for a given scenario."""
    
    print("-" * 110)
    print(f"{scenario_name} Scenario (PD Multiplier: {multiplier:.2f}x)")
    # Updated headers to be crystal clear
    print(f"{'Coverage':<12} {'ECL':>12} {'Change vs. Current (50%)':>25} {'Reduction vs. Uninsured (0%)':>25}")

    ecl_by_coverage = []
    
    for coverage in coverage_rates:
        total_ecl = 0
        for loan_name, loan_info in portfolio.items():
            if loan_name == 'Total':
                continue
            
            insured = loan_info['amt'] * coverage
            uninsured = loan_info['amt'] * (1 - coverage)
            adj_pd = loan_info['pd'] * multiplier
            
            ecl = insured * adj_pd * lgd_with_girsal + uninsured * adj_pd * lgd_full
            total_ecl += ecl
        
        ecl_by_coverage.append((coverage, total_ecl))

    # Store data for the last scenario run (Severe_Drought) for plotting
    ecl_sensitivity_data[scenario_name] = ecl_by_coverage
    
    # Find base case (50% coverage) and uninsured case (0% coverage) for this scenario
    base_ecl_50 = [ecl for cov, ecl in ecl_by_coverage if cov == 0.50][0]
    uninsured_ecl_0 = [ecl for cov, ecl in ecl_by_coverage if cov == 0.00][0]

    for coverage, total_ecl in ecl_by_coverage:
        change_vs_current = total_ecl - base_ecl_50
        reduction_vs_uninsured = uninsured_ecl_0 - total_ecl
        marker = " ← Current" if coverage == 0.50 else ""
        
        print(f"{coverage*100:>10.0f}%  ${total_ecl/1e6:>11.2f}M  ${change_vs_current/1e6:>23.2f}M  ${reduction_vs_uninsured/1e6:>23.2f}M{marker}")

# Run sensitivity for both Baseline and Severe Drought
run_sensitivity('Baseline_2024', scenario_results['Baseline_2024']['multiplier'])
run_sensitivity('Severe_Drought', scenario_results['Severe_Drought']['multiplier'])


# EXECUTIVE SUMMARY
print("\n\nEXECUTIVE SUMMARY\n")

baseline_with = ecl_with['Baseline_2024']
baseline_without = ecl_without['Baseline_2024']
girsal_value = baseline_without - baseline_with

print(f"Portfolio: ${portfolio['Total']/1e6:.0f}M | Ghana Maize (2004-2023)")
# IMPORTANT: Update output here to reflect YIELD model results
print(f"Model: YIELD (R² = {r2_y_full:.3f}) | Climate = {contrib_y_corrected*100:.0f}% of detrended variance\n")

# Calculate key findings from actual data
yield_start = df_agric['Yield_tonnes_ha'].iloc[0]
yield_end = df_agric['Yield_tonnes_ha'].iloc[-1]
yield_growth_factor = yield_end / yield_start
yield_growth_pct = ((yield_end / yield_start) - 1) * 100

area_start = df_agric['Area_ha'].iloc[0]
area_end = df_agric['Area_ha'].iloc[-1]
area_growth_pct = ((area_end / area_start) - 1) * 100

trend_r2_y = r_y**2
slope_yield = slope_y * 1000 # convert to kg/ha/year

print(f"KEY FINDINGS:")
print(f"  • Yield growth: {yield_start:.2f} → {yield_end:.2f} t/ha ({yield_growth_factor:.1f}x increase, +{slope_yield:,.0f} kg/ha/year)")
print(f"  • Area expansion: +{area_growth_pct:.1f}%")
print(f"  • Climate explains {contrib_y_corrected*100:.1f}% of DETRENDED YIELD variance, Trend explains {trend_r2_y*100:.0f}% of total YIELD variance")
print(f"  • {len(drought_years)} drought years identified: {drought_years}\n")

print(f"GIRSAL INSURANCE VALUE:")
print(f"  With GIRSAL: ${baseline_with/1e6:.2f}M ({baseline_with/portfolio['Total']:.2%})")
print(f"  Without GIRSAL: ${baseline_without/1e6:.2f}M ({baseline_without/portfolio['Total']:.2%})")
print(f"  GIRSAL Value: ${girsal_value/1e6:.2f}M ({girsal_value/baseline_without*100:.0f}% reduction)\n")

print(f"CLIMATE RISK INSIGHTS (Yield-Based):")
print(f"  • Severe drought triggers {scenario_results['Severe_Drought']['multiplier']:.2f}x PD multiplier")
print(f"  • Risk = falling below expected YIELD growth")
print(f"  • Government insurance is CRITICAL for agricultural lending viability\n")

print("MODEL COMPLETE\n")

# VISUALIZATIONS
print("Generating visualizations...")

# Set dark mode style
plt.style.use('dark_background')

# Plot 1: ECL Comparison Across Scenarios
plt.figure(figsize=(12, 7)) # Increased width slightly
scenarios_plot = list(ecl_with.keys())
ecl_with_values = [ecl_with[s]/1e6 for s in scenarios_plot]
ecl_without_values = [ecl_without[s]/1e6 for s in scenarios_plot]
x = np.arange(len(scenarios_plot))
width = 0.35

bar_with = plt.bar(x - width/2, ecl_with_values, width, label='With GIRSAL (50% Coverage)', color='#00D9FF')
bar_without = plt.bar(x + width/2, ecl_without_values, width, label='Uninsured (0% Coverage)', color='#FF6B35')

# Add labels above the bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'${height:.2f}M',
                 ha='center', va='bottom', fontsize=10, color='white')

add_labels(bar_with)
add_labels(bar_without)

plt.xlabel('Scenario', fontsize=12)
plt.ylabel('ECL ($M)', fontsize=12)
plt.title('Expected Credit Loss by Scenario: Stress-Testing GIRSAL Value', fontsize=14, fontweight='bold')
# Set x-ticks horizontally (rotation=0)
plt.xticks(x, scenarios_plot, rotation=0, ha='center') 
# Move legend to the bottom, outside the plot area
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
# Set y-axis to start from a visible minimum and increase max for labels
y_min = 2.5 
y_max = max(max(ecl_with_values), max(ecl_without_values)) * 1.10 # Adjusted max for label space
plt.ylim(y_min, y_max)
plt.grid(True, alpha=0.2, axis='y')
plt.tight_layout(rect=[0, 0.1, 1, 1]) # Adjust layout to make space for the legend
plt.savefig('ecl_comparison.png', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')

# Plot 2: GIRSAL Coverage Sensitivity
plt.figure(figsize=(10, 6))
# Use the Severe Drought data stored in the dictionary for plotting
severe_drought_data = ecl_sensitivity_data['Severe_Drought']
coverage_pcts = [cov*100 for cov, _ in severe_drought_data]
ecl_values = [ecl/1e6 for _, ecl in severe_drought_data]

plt.plot(coverage_pcts, ecl_values, 'o-', color='#00D9FF', linewidth=3, markersize=10)
plt.axvline(x=50, color='#FF3366', linestyle='--', linewidth=2, label='Current (50%)')
plt.fill_between(coverage_pcts, ecl_values, alpha=0.3, color='#00D9FF')
plt.xlabel('GIRSAL Coverage (%)', fontsize=12)
plt.ylabel('ECL ($M)', fontsize=12)
# Dynamic title now includes the Severe Drought PD Multiplier
severe_drought_mult = scenario_results['Severe_Drought']['multiplier']
plt.title(f'Insurance Coverage Impact on ECL (Severe Drought Stress: {severe_drought_mult:.2f}x PD)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('coverage_sensitivity.png', dpi=300, bbox_inches='tight', facecolor='#1a1a1a')

plt.show()

print("Visualizations saved:")
print("  - ecl_comparison.png")
print("  - coverage_sensitivity.png")
