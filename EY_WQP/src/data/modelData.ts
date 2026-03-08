// Real model performance data extracted from enhanced_model_217_v3 notebook

export interface ModelResult {
  name: string;
  fullName: string;
  ta: number;
  ec: number;
  drp: number;
}

// 12-model CV R² comparison from notebook output
export const modelComparisonData: ModelResult[] = [
  { name: "XGB", fullName: "XGBoost", ta: 0.8514, ec: 0.8683, drp: 0.7157 },
  { name: "RF_500", fullName: "Random Forest", ta: 0.8464, ec: 0.8649, drp: 0.7089 },
  { name: "LGB", fullName: "LightGBM", ta: 0.8458, ec: 0.8571, drp: 0.6945 },
  { name: "GB_500", fullName: "Gradient Boosting", ta: 0.8439, ec: 0.8592, drp: 0.6870 },
  { name: "ET_500", fullName: "ExtraTrees", ta: 0.8394, ec: 0.8535, drp: 0.7157 },
  { name: "LGB_reg", fullName: "LightGBM (reg)", ta: 0.8399, ec: 0.8500, drp: 0.6812 },
  { name: "XGB_deep", fullName: "XGBoost (deep)", ta: 0.8380, ec: 0.8496, drp: 0.6750 },
  { name: "RF_reg", fullName: "Random Forest (reg)", ta: 0.8354, ec: 0.8528, drp: 0.6920 },
  { name: "XGB_reg", fullName: "XGBoost (reg)", ta: 0.8283, ec: 0.8420, drp: 0.6680 },
  { name: "GB_800", fullName: "Gradient Boosting (800)", ta: 0.8245, ec: 0.8396, drp: 0.6590 },
  { name: "LGB_deep", fullName: "LightGBM (deep)", ta: 0.8165, ec: 0.8254, drp: 0.6450 },
  { name: "GB_1000", fullName: "Gradient Boosting (1000)", ta: 0.7789, ec: 0.7953, drp: 0.6120 },
];

// Best model per target
export const bestModels = {
  ta: { model: "XGBoost", cvR2: 0.8514, inSampleR2: 0.9499 },
  ec: { model: "XGBoost", cvR2: 0.8683, inSampleR2: 0.9641 },
  drp: { model: "ExtraTrees", cvR2: 0.7157, inSampleR2: 0.9200 },
};

// Training dataset stats
export const trainingStats = {
  samples: 9319,
  features: 29,
  locations: 200,
  dataSources: 3,
  targetStats: {
    ta: { mean: 119.1, std: 74.7, min: 4.8, max: 361.7, skew: 0.54 },
    ec: { mean: 485.0, std: 341.9, min: 15.1, max: 1506.0, skew: 0.93 },
    drp: { mean: 43.5, std: 51.0, min: 5.0, max: 195.0, skew: 1.64 },
  },
};

// Feature importance per target from notebook (top 10 each)
export interface FeatureImportance {
  feature: string;
  importance: number;
  description: string;
}

export const featureImportanceByTarget: Record<string, FeatureImportance[]> = {
  "Total Alkalinity": [
    { feature: "Longitude", importance: 0.2106, description: "East-west position — linked to geological and land-use patterns" },
    { feature: "Latitude", importance: 0.1539, description: "North-south position — climate zones affect water chemistry" },
    { feature: "swir22_log", importance: 0.0675, description: "Log short-wave infrared — sensitive to soil moisture and minerals" },
    { feature: "pet", importance: 0.0642, description: "Potential evapotranspiration — water loss to atmosphere" },
    { feature: "pet_log", importance: 0.0377, description: "Log evapotranspiration — non-linear climate effects" },
    { feature: "MNDWI", importance: 0.0370, description: "Modified water index — detects open water surfaces" },
    { feature: "nir_log", importance: 0.0317, description: "Log near-infrared — vegetation health near rivers" },
    { feature: "green_swir22_ratio", importance: 0.0290, description: "Band ratio — water clarity and sediment levels" },
    { feature: "swir22", importance: 0.0263, description: "Raw short-wave infrared — minerals and soil composition" },
    { feature: "swir16_log", importance: 0.0247, description: "Log SWIR band — subtle spectral variation" },
  ],
  "Electrical Conductance": [
    { feature: "Longitude", importance: 0.2821, description: "East-west position — strongest predictor, geology and salinity" },
    { feature: "Latitude", importance: 0.1369, description: "North-south position — climate gradient affects dissolved salts" },
    { feature: "swir22_log", importance: 0.0746, description: "Log SWIR — highly sensitive to mineral content" },
    { feature: "pet", importance: 0.0329, description: "Evapotranspiration — higher PET concentrates salts" },
    { feature: "swir16_log", importance: 0.0296, description: "Log SWIR band — spectral signatures of saline soils" },
    { feature: "green_swir22_ratio", importance: 0.0272, description: "Water clarity index — turbidity and dissolved minerals" },
    { feature: "pet_log", importance: 0.0255, description: "Log evapotranspiration — non-linear climate on salinity" },
    { feature: "year", importance: 0.0255, description: "Temporal trend — conductance changed over 2011-2015" },
    { feature: "MNDWI_sq", importance: 0.0251, description: "Squared water index — extreme wet/dry conditions" },
    { feature: "NDWI", importance: 0.0249, description: "Water index — open water detection from satellite" },
  ],
  "Dissolved Reactive Phosphorus": [
    { feature: "Latitude", importance: 0.2609, description: "North-south position — tied to agricultural intensity" },
    { feature: "Longitude", importance: 0.2522, description: "East-west position — land use drives phosphorus runoff" },
    { feature: "year", importance: 0.0544, description: "Temporal trend — phosphorus changed over 5 years" },
    { feature: "pet_sq", importance: 0.0456, description: "Squared evapotranspiration — extreme climate effects" },
    { feature: "pet", importance: 0.0408, description: "Evapotranspiration — dry conditions concentrate phosphorus" },
    { feature: "pet_log", importance: 0.0382, description: "Log evapotranspiration — non-linear climate-phosphorus" },
    { feature: "month_cos", importance: 0.0245, description: "Seasonal cycle — winter/summer patterns" },
    { feature: "day_of_year", importance: 0.0191, description: "Fine-grained seasonality — rainfall timing" },
    { feature: "month", importance: 0.0172, description: "Calendar month — seasonal agricultural cycles" },
    { feature: "MNDWI", importance: 0.0167, description: "Water index — surface water extent" },
  ],
};

// V3 methodology summary
export const methodology = {
  version: "V217_v3",
  baseScore: "0.227 R²",
  finalScore: "0.29 R²",
  scaler: "RobustScaler",
  cvFolds: 5,
  tweaks: [
    "Winsorized targets (1st-99th percentile) to reduce outlier influence",
    "DRP log-transform comparison (raw won over log1p)",
    "Wider model search (12 candidates including regularized variants)",
    "Smarter clipping (0.5th-99.5th percentile, 1.1x upper bound)",
    "No new features added (avoid noise/overfitting)",
  ],
};
