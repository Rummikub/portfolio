# Urban Heat Island Hotspot Prediction  
## EY 2025 AI & Data Challenge

## Overview

This project was developed as part of the EY 2025 AI & Data Challenge, which focused on modeling and predicting the Urban Heat Island (UHI) effect in dense metropolitan environments.

Urban heat islands emerge when built infrastructure, reduced vegetation, and limited surface water alter local thermal dynamics. In some cities, temperature differences between urban cores and surrounding rural areas can exceed 10°C, intensifying public health risks, energy demand, and infrastructure stress. These impacts disproportionately affect vulnerable populations, including older adults, children, outdoor workers, and low-income communities.

The objective of this project was to:

1. Predict micro-scale temperature variations across urban grid regions  
2. Identify and quantify the environmental drivers contributing to localized heat hotspots  
3. Assess how such a modeling framework could generalize to other cities and datasets  

---

## Problem Statement

Urban planners and policymakers require fine-grained thermal predictions to support evidence-based interventions.

Accurate micro-scale temperature modeling enables:

- Identification of high-risk heat clusters  
- Prioritization of green infrastructure investment  
- Optimization of cooling center placement  
- Data-informed zoning and land-use planning  

The challenge required constructing a supervised machine learning model to forecast localized temperatures at meter-level resolution. Model performance was evaluated using least-squares error metrics (R² and Mean Squared Error) against observed ground temperature measurements.

---

## Data Sources

This project integrates multi-source environmental and climate datasets:

- TerraClimate gridded climate data  
- Urban land surface and land-use indicators  
- Spatially referenced temperature sensor measurements  
- Time-series climate variables  

The primary modeling objective was to predict localized temperature intensity across urban micro-clusters while preserving spatial coherence.

---

## Methodology

### 1. Data Engineering

- Aggregated high-resolution geospatial climate grids  
- Engineered lag-based temporal features to capture heat persistence  
- Computed rolling statistical aggregates (mean, variance)  
- Integrated spatial context variables derived from environmental indicators  
- Standardized feature distributions to stabilize model training  

### 2. Feature Engineering

- Vegetation density and greenness proxies  
- Urban density approximations  
- Surface material and land-use indicators  
- Long-term climate trend features  
- Interaction terms capturing nonlinear environmental relationships  

### 3. Modeling Approach

Evaluated multiple ensemble regression models:

- Gradient Boosting Regressor  
- Random Forest Regressor  
- XGBoost Regressor  

Applied:

- K-Fold cross-validation for robustness  
- Hyperparameter tuning via grid search  
- Feature importance analysis for interpretability  
- Residual diagnostics to assess spatial bias  

---

## Model Evaluation

Primary evaluation metrics:

- R² (coefficient of determination)  
- Mean Squared Error (MSE)  

Key findings:

- Model performance was highly sensitive to feature engineering depth  
- Spatial aggregation granularity significantly influenced predictive accuracy  
- High-dimensional environmental inputs increased overfitting risk without proper regularization  
- Vegetation-related and density-derived features consistently ranked among the strongest predictors  
