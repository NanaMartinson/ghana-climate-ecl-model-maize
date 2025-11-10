# **Ghana Climate Risk ECL Model for Maize Production (2003-2023)**

Author: Nana Nshira Martinson  
Contact: [nanamartinsonnnm@gmail.com](mailto:nanamartinsonnnm@gmail.com)  
Date: November 2025

## **Executive Summary**

This project presents a Climate-Adjusted Expected Credit Loss (ECL) model designed to stress-test a $100 million Ghanaian maize lending portfolio against future climate shocks. The core methodology uses detrended crop yield to establish a statistically robust link between climate anomalies and credit risk, a critical requirement for forward-looking financial stability assessments.

The core finding confirms that while overall yield growth is strong, climate is the largest single non-economic driver of yield volatility, triggering a non-linear jump in Probability of Default (PD) that is effectively mitigated by the GIRSAL credit guarantee scheme.

### **Key Insights & Financial Impact**

| Key Metric | Finding (Yield-Based Model) | Relevance |
| ----- | ----- | ----- |
| Climate Contribution to Yield | 33.4% of detrended yield variance explained by climate variables. | Systemic Risk: This confirms climate is the largest single non-economic driver of yield volatility. This 33.4% represents the non-diversifiable, systemic portion of portfolio risk, demanding targeted adaptation funding. |
| Severe Drought Stress | PD Multiplier of 1.09 (a 9% increase in PD). | Portfolio Instability: Confirms that climate operates as a tail risk—a sudden, non-linear shock that increases the default rate and threatens financial stability, despite strong long-term growth. |
| GIRSAL Value (Baseline) | $1.57 Million in ECL reduction. | Risk Transfer Efficiency: The scheme provides a 35% reduction in expected losses under normal conditions, de-risking commercial bank exposure. |
| GIRSAL Value (Under Stress) | $1.71 Million in ECL reduction. | Proof of Resilience: The absolute dollar value of the guarantee increases under the Severe Drought scenario, proving the mechanism is most effective when financial stability is threatened. |

### Expected Credit Loss (ECL) by Scenario

| Scenario | PD Multiplier | ECL With GIRSAL | ECL Without GIRSAL |
| :---: | :---: | :---: | :---: |
| Baseline 2024 | 1.00 | $2.91m | $4.47m |
| Moderate Drought | 1.03 | $3.00m | $4.61m |
| Severe Drought | 1.09 | $3.17m | $4.88m |
| Climate 2030 | 1.10 | $3.20m | $4.93m |

## **Methodology & Model Structure**

### Portfolio Structure

The model is stress-tested against a hypothetical $100 million portfolio segmented by term length, with credit risk parameters derived from the GEMs database. The 50% GIRSAL coverage applies uniformly across all loan segments.

| Term Segment | Exposure ($M) | Term (Years) | Lifetime PD (Annual PD: 6.78%) |
| :---: | :---: | :---: | :---: |
| Short-Term | $40m | 1.5 | 10.00% |
| Medium-Term | $40m | 3.0 | 18.99% |
| Long-Term | $20m | 7.0 | 38.83% |
| **Total (Weighted Average)** | **$100M** | **3.2** | **20.72%** |

### GIRSAL Risk Parameters

The Ghana Incentive-Based Risk Sharing System for Agricultural Lending (GIRSAL) is a credit guarantee scheme that reduces the Loss Given Default (LGD) exposure for participating lenders.

Coverage Rate: 50% of the loan principal is insured (girsal\_rate \= 0.50).

LGD Parameters: Full LGD (uninsured) is 23.1%. LGD on the insured portion is significantly reduced to 6.93% (23.1% x 30%).

### Model Structure (Hybrid Approach)

Trend Component: A Linear Regression of Yield (t/ha) against Time to capture structural growth (e.g., technology adoption, area expansion). Trend explains 85% of total yield variance.

Climate Component (Ridge Regression): Climate features (Temperature, Precipitation) are used to model the detrended yield. Climate explains 33.4% of this detrended variance. Ridge Regression was selected to model the relationship between climate features and yield volatility. This choice is deliberate because it stabilizes the model when integrating related, highly correlated climate variables (e.g., Temperature and Precipitation), ensuring the coefficients are reliable. The model focuses only on the most critical biophysical drivers of yield failure (key seasonal temperatures and rainfall totals) and excludes redundant features like Insolation.

Credit Component (PD Multiplier): Climate-induced yield shortfalls are converted into a Z-score, which is mapped to a dynamic PD multiplier to stress-test the portfolio.

### PD Multiplier Framework

Scenario-based yield shortfalls (measured by standard deviation from the expected trend, or Z-score) are mapped to a continuous PD multiplier. This framework ensures that credit risk increases non-linearly only when the climate shock breaches a critical threshold.

PD Multiplier \= 1.00 \+ (Z-score x 0.15)

A shortfall of 0.61 standard deviations (Severe Drought) results in a 1.09 times PD multiplier (a 9% increase in default risk). This aggressive sensitivity is justified by the high physical impact of weath on crop yield.

## **Limitations & Future Development**

This model is a strong proof of concept for integrating climate physical risk into expected credit loss (ECL models). To transition to a production-ready model for use in lending decisions, the following limitations must be addressed:

| Constraint Category | Current Limitation | Future Development Required |
| :---: | ----- | ----- |
| Data Granularity | Relies on 20 years of national average climate (NASA POWER) data | Spatial Disaggregation: Integrate localised meteorological data to model regional risk variability. |
| Portfolio Level | Assumes uniform risk for the entire $100m portfolio; lacks actual loan-level data for calibration. | Loan-Level Integration: Secure bank-specific historical default data to empirically validate PD and LGD parameters and calibrate the PD multiplier magnitude (0.15). |
| Model Scope | Focuses solely on physical risk (weather) for a single crop (Maize). | Multi-Factor Stress-Testing: Incorporate transition risk (policy/regulation) and key economic variables (commodity prices, exchange rates) that also drive repayment capacity. |
| Assumptions | The PD multiplier sensitivity of 0.15 is based on regulatory guidance and stress-testing requirements for conservative estimates (expert judgment). | Empirical Validation: Test and calibrate PD multiplier thresholds against historical default observations during past drought years. |

**Reference List**

Maize Yield (t/ha) and Production (tonnes) Data  
Food and Agriculture Organization of the United Nations (FAO). (n.d.). FAOSTAT: Ghana Maize Production Quantity & Yield (2004-2024). Retrieved August 2025, from [https://www.fao.org/faostat/en/\#data/QCL](https://www.fao.org/faostat/en/#data/QCL)

Climate Data (Monthly Precipitation and Temperature)  
National Aeronautics and Space Administration (NASA). (n.d.). NASA POWER: Precipitation, Temperature, and Solar Radiation for Ghana (2004-2024) \[Data set\]. NASA Langley Research Center (LaRC) Applied Sciences Program. Retrieved August 2025, from [https://power.larc.nasa.gov/data-access-viewer/](https://power.larc.nasa.gov/data-access-viewer/)

Credit Risk Parameters  
Global Emerging Markets Risk Database (GEMs). (n.d.). Default and Recovery Statistics: Private and Public Lending 1994-2023. Retrieved August 2025, from [https://www.gemsriskdatabase.org/](https://www.gemsriskdatabase.org/)

Loan Insurance LGD Reduction Parameters  
Ghana Incentive-Based Risk Sharing System for Agricultural Lending (GIRSAL). (n.d.). Agricultural Credit Guarantee Scheme \[Policy document\]. Retrieved August 2025, from [https://www.girsal.com/agricultural-credit-guarantee-scheme/](https://www.girsal.com/agricultural-credit-guarantee-scheme/)

