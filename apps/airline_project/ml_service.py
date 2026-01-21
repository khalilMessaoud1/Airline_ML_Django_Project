"""
ML Service - Runs all models when a client is created
"""
import pickle
import joblib
import os
import numpy as np
from django.conf import settings
from .models import Client, MLPrediction

MODELS_DIR = os.path.join(settings.BASE_DIR, 'ml_models', 'khalil')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')
KMEANS_PATH = os.path.join(MODELS_DIR, 'kmeans.joblib')
KNN_PATH = os.path.join(MODELS_DIR, 'knn.joblib')
SEASON_KNN_PATH = os.path.join(MODELS_DIR, 'season_knn.joblib')
# Cache pour les modèles chargés
_cached_models_khalil = {}
class MLPredictionService:
    """Service to run all ML models for a client"""
    
    def __init__(self):
        # Path to ML models
        self.models_dir = os.path.join(settings.BASE_DIR, 'ml_models')
    def unwrap_estimator(
        self,
        obj,
        preferred_keys=("pipeline", "kmeans", "model", "estimator", "classifier", "regressor", "clf", "scaler")
    ):
        """Extract sklearn estimator from dict if needed"""
        if isinstance(obj, dict):
            print(f"⚠ Loaded dict with keys: {list(obj.keys())}")
            for k in preferred_keys:
                if k in obj:
                    return obj[k]

            # fallback: first sklearn-like object
            for v in obj.values():
                if hasattr(v, "predict") or hasattr(v, "transform"):
                    return v
        return obj
    def safe_float(self, v, default=0.0):
        """Convert to float, replacing None/NaN/inf with default."""
        try:
            x = float(v)
            if np.isnan(x) or np.isinf(x):
                return float(default)
            return x
        except Exception:
            return float(default)
    def load_pkl(self, path):
        """Load pickle file with multiple fallback methods"""
        import sklearn
        print(f"Loading {path} with sklearn {sklearn.__version__}")

        # Try pickle first
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Pickle loading failed: {e}")

        # Try joblib
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"Joblib loading failed: {e}")

        # Try with different pickle protocols
        for protocol in [pickle.HIGHEST_PROTOCOL, 4, 3, 2]:
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f, fix_imports=True, encoding='latin1')
            except Exception as e:
                print(f"Pickle protocol {protocol} failed: {e}")
                continue

        print(f"All loading methods failed for {path}")
        return None
    
    def load_joblib(self, path):
        """Load joblib file"""
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def run_habiba_models(self, client, prediction):
        """
        HABIBA - Customer Segmentation & CLV Prediction
        """
        print("🔵 Running HABIBA models...")
        encoded = client.get_encoded_values()
        
        try:
            # 1. Customer Segmentation (Clustering)
            # ==============================
            # 1. Load segmentation bundle
            # ==============================
            seg_bundle = self.load_pkl(
                os.path.join(self.models_dir, 'habiba', 'customer_segmentation_kmeans.pkl')
            )

            seg_scaler = seg_bundle.get("scaler")
            seg_kmeans = seg_bundle.get("kmeans")

            # ==============================
            # 2. Initial segmentation (CLV = 0)
            # ==============================
            seg_features = np.array([[
                0,  # CLV placeholder
                client.rating_value,
                client.salary,
                encoded['education_encoded'],
                encoded['marital_status_encoded'],
                encoded['class_encoded'],
                encoded['tier_encoded'],
                client.satisfaction,
            ]])

            X = seg_scaler.transform(seg_features) if seg_scaler else seg_features
            cluster = seg_kmeans.predict(X)[0]

            prediction.segmentation_cluster = int(cluster)
            prediction.segmentation_label = "Loyal Premium" if cluster == 0 else "Basic Economy"

            print(f"   ✓ Initial Segmentation: {prediction.segmentation_label}")

            # ==============================
            # 3. CLV prediction
            # ==============================
            clv_model = self.unwrap_estimator(
                self.load_pkl(
                    os.path.join(self.models_dir, 'habiba', 'clv_random_forest.pkl')
                )
            )

            clv_features = np.array([[
                client.rating_value,
                client.salary,
                encoded['education_encoded'],
                encoded['marital_status_encoded'],
                encoded['class_encoded'],
                encoded['tier_encoded'],
                client.satisfaction,
            ]])

            clv = clv_model.predict(clv_features)[0]
            prediction.clv_prediction = float(clv)

            print(f"   ✓ CLV: {clv:.2f}")

            # ==============================
            # 4. 🔁 RE-RUN segmentation WITH REAL CLV
            # ==============================
            seg_features[0][0] = clv   # ← THIS is the line you asked about

            X = seg_scaler.transform(seg_features) if seg_scaler else seg_features
            cluster = seg_kmeans.predict(X)[0]

            prediction.segmentation_cluster = int(cluster)
            prediction.segmentation_label = "Loyal Premium" if cluster == 0 else "Basic Economy"

            print(f"   ✓ Final Segmentation: {prediction.segmentation_label}")

            
            # 2. CLV Prediction (Regression)
            clv_model =self.load_pkl(
                    os.path.join(self.models_dir, 'habiba', 'clv_random_forest.pkl')
                )
            if clv_model:
                clv_features = np.array([[
                    client.rating_value,
                    client.salary,
                    encoded['education_encoded'],
                    encoded['marital_status_encoded'],
                    encoded['class_encoded'],
                    encoded['tier_encoded'],
                    client.satisfaction,
                ]])
                
                clv = clv_model.predict(clv_features)[0]
                prediction.clv_prediction = float(clv)
                print(f"   ✓ CLV: ${clv:,.2f}")
                
                # Re-run segmentation with actual CLV
                if seg_kmeans is not None:
                    seg_features[0][0] = clv
                    X = seg_scaler.transform(seg_features) if seg_scaler else seg_features
                    cluster = seg_kmeans.predict(X)[0]
                    prediction.segmentation_cluster = int(cluster)
                    prediction.segmentation_label = "Loyal Premium" if cluster == 0 else "Basic Economy"
        
        except Exception as e:
            print(f"   ❌ HABIBA error: {e}")
            
    
    def run_adhem_models(self, client, prediction):
        """
        ADHEM - Churn Classification & Regression
        """
        print("🔵 Running ADHEM models...")
        encoded = client.get_encoded_values()
        
        try:
            # 1. Churn Classification
            scaler = self.unwrap_estimator(
                self.load_pkl(
                    os.path.join(self.models_dir, 'adhem', 'scaler.joblib')
                )
            )

            classifier = self.unwrap_estimator(
                self.load_pkl(
                    os.path.join(self.models_dir, 'adhem', 'classifier.joblib')
                )
            )
            
            if scaler and classifier:
                clf_features = np.array([[
                    client.points_accumulated,
                    client.total_flights,
                    client.distance,
                    encoded['travel_type_encoded'],
                    client.satisfaction,
                    client.enrollment_duration,
                ]])
                
                clf_features_scaled = scaler.transform(clf_features)
                churn_pred = classifier.predict(clf_features_scaled)[0]
                
                prediction.churn_classification = int(churn_pred)
                prediction.churn_risk_label = "Will Churn" if churn_pred == 1 else "Won't Churn"
                prediction.risk_level = "High" if churn_pred == 1 else "Low"
                print(f"   ✓ Churn Classification: {prediction.churn_risk_label}")
            
            # 2. Churn Regression (Months until churn)
            scaler_reg = self.unwrap_estimator(
                self.load_pkl(
                    os.path.join(self.models_dir, 'adhem', 'scaler_regression.joblib')
                )
            )

            regressor = self.unwrap_estimator(
                self.load_pkl(
                    os.path.join(self.models_dir, 'adhem', 'regression.joblib')
                )
            )
            
            if scaler_reg and regressor:
                reg_features = np.array([[
                    client.total_flights,
                    client.distance,
                    encoded['travel_type_encoded'],
                    client.rating_value,
                    client.points_redeemed,
                    encoded['education_encoded'],
                    encoded['marital_status_encoded'],
                    client.salary,
                    encoded['gender_encoded'],
                    client.satisfaction,
                    encoded['enrollment_encoded'],
                ]])
                
                reg_features_scaled = scaler_reg.transform(reg_features)
                churn_months = regressor.predict(reg_features_scaled)[0]
                
                prediction.churn_months = float(churn_months)
                print(f"   ✓ Churn Timeline: {churn_months:.1f} months")
        
        except Exception as e:
            print(f"   ❌ ADHEM error: {e}")
    
    def run_asma_models(self, client, prediction):
        """
        ASMA - Enrollment Insights & Equity Segmentation
        """
        print("🔵 Running ASMA models...")
        encoded = client.get_encoded_values()
        
        try:
            # 1. Highly Engaged Classification
            engaged_model = self.load_joblib(os.path.join(self.models_dir, 'asma', 'random_forest_smote.joblib'))
            
            if engaged_model:
                engaged_features = np.array([[
                    encoded['class_encoded'],
                    encoded['province_encoded'],  # Use actual encoding
                    encoded['city_encoded'],     # Use actual encoding
                    client.salary,
                    encoded['education_encoded'],
                    encoded['marital_status_encoded'],
                    encoded['gender_encoded'],
                    encoded['tier_encoded'],
                    client.enrollment_year,
                    client.enrollment_month,
                    client.cancellation_year if client.cancellation_year else 0,
                    client.cancellation_month if client.cancellation_month else 0,
                    client.satisfaction,
                    encoded['travel_type_encoded'],
                ]])
                
                engaged_pred = engaged_model.predict(engaged_features)[0]
                prediction.highly_engaged = int(engaged_pred)
                prediction.engagement_label = "Highly Engaged" if engaged_pred == 1 else "Low Engagement"
                print(f"   ✓ Engagement: {prediction.engagement_label}")
            
            # 2. Equity Segmentation (Clustering)
            # 2. Equity Segmentation (Clustering)
            scaler = self.load_joblib(os.path.join(self.models_dir, 'asma', 'kmeans_equity_scaler.joblib'))
            equity_model = self.load_joblib(os.path.join(self.models_dir, 'asma', 'kmeans_equity_segment.joblib'))
            
            if equity_model:
                equity_features = np.array([[
                    encoded['province_encoded'],
                    client.salary,
                    encoded['education_encoded'],
                    encoded['marital_status_encoded'],
                    encoded['gender_encoded'],
                    encoded['tier_encoded'],
                    client.enrollment_month,
                    client.satisfaction,
                    encoded['travel_type_encoded'],
                ]])
                
                equity_features_scaled = scaler.transform(equity_features)
                cluster = equity_model.predict(equity_features_scaled)[0]
                prediction.equity_segment_cluster = int(cluster)
                if cluster == 1:
                    prediction.equity_segment_label = "Low Potential – Low Income & Education"
                elif cluster == 2:
                    prediction.equity_segment_label = "High Potential – High Income & Educated"
                else:  # cluster 0
                    prediction.equity_segment_label = "Moderate Potential – Regional Middle-Income"
                print(f"   ✓ Equity Segment: Group {cluster}")
        
        except Exception as e:
            print(f"   ❌ ASMA error: {e}")
            # Set default values
            prediction.highly_engaged = 0
            prediction.engagement_label = "Low Engagement"
            prediction.equity_segment_cluster = 1
            prediction.equity_segment_label = "Segment Group 1"
    
    def run_wajd_models(self, client, prediction):
        """
        WAJD - Loyalty Points Classification, Clustering & Forecasting
        """
        print("🔵 Running WAJD models...")
        encoded = client.get_encoded_values()
        
        try:
            # 1. Loyalty Points Classification
            scaler = self.load_pkl(os.path.join(self.models_dir, 'wajd', 'classification_scaler.pkl'))
            knn_model = self.load_pkl(os.path.join(self.models_dir, 'wajd', 'knn_model.pkl'))
            
            if scaler and knn_model:
                clf_features = np.array([[
                    client.points_accumulated,
                    client.enrollment_month,
                    client.enrollment_year,
                    client.cancellation_month if client.cancellation_month else 0,
                    client.total_flights,
                    client.distance,
                ]])
                
                clf_features_scaled = scaler.transform(clf_features)
                loyalty_pred = knn_model.predict(clf_features_scaled)[0]
                
                prediction.loyalty_points_classification_label = int(loyalty_pred)
                print(f"   ✓ Loyalty Classification: {loyalty_pred}")
            
            # 2. Loyalty Clustering
            cluster_scaler = self.load_pkl(os.path.join(self.models_dir, 'wajd', 'cluster_scaler.pkl'))
            kmeans = self.load_pkl(os.path.join(self.models_dir, 'wajd', 'kmeans_model.pkl'))
            
            if cluster_scaler and kmeans:
                cluster_features = np.array([[
                    client.satisfaction,
                    client.points_accumulated,
                    client.rating_value,
                    client.enrollment_month,
                    encoded['travel_type_encoded'],
                    client.enrollment_year,
                    client.cancellation_month if client.cancellation_month else 0,
                    client.cancellation_year if client.cancellation_year else 0,
                    client.points_redeemed,
                    client.total_flights,
                    client.distance,
                ]])
                
                cluster_features_scaled = cluster_scaler.transform(cluster_features)
                cluster = kmeans.predict(cluster_features_scaled)[0]
                
                prediction.loyalty_cluster = int(cluster)
                prediction.loyalty_cluster_label = "Active but low intensity flyers" if cluster == 0 else "Redemption Focused Members"
                print(f"   ✓ Loyalty Cluster: {prediction.loyalty_cluster_label}")
            
            # 3. SARIMA Forecasting (TODO: Implement time series forecasting)
            # sarima_model = self.load_pkl(os.path.join(self.models_dir, 'wajd', 'sarima_model.pkl'))
            # if sarima_model:
            #     forecast = sarima_model.forecast(steps=12)
            #     prediction.points_forecast = float(forecast[0])
        
        except Exception as e:
            print(f"   ❌ WAJD error: {e}")

    def run_khalil_models(self, client, prediction):
        """
        KHALIL - Customer Clustering + Preferred Travel Season + Similar Neighbors (optional)
        Models expected in: ml_models/khalil/
        - scaler.joblib
        - kmeans.joblib
        - season_knn.joblib
        - knn.joblib (optional)
        """
        print("🔵 Running KHALIL models...")
        encoded = client.get_encoded_values()

        try:
            # ------------------------------
            # 1) Load models (same style)
            # ------------------------------
            scaler = self.unwrap_estimator(
                self.load_pkl(os.path.join(self.models_dir, 'khalil', 'scaler.joblib'))
            )
            kmeans = self.unwrap_estimator(
                self.load_pkl(os.path.join(self.models_dir, 'khalil', 'kmeans.joblib'))
            )
            season_knn = self.unwrap_estimator(
                self.load_pkl(os.path.join(self.models_dir, 'khalil', 'season_knn.joblib'))
            )
            knn = self.unwrap_estimator(
                self.load_pkl(os.path.join(self.models_dir, 'khalil', 'knn.joblib'))
            )  # optional

            # ------------------------------
            # 2) Build the 8 features (order must match training!)
            # Khalil expects:
            # [Cancellation_Month, Cancellation_Year, Distance,
            #  Dollar_Cost_Points_Redeemed, Flight_Month, Points_Redeemed,
            #  Total_Flights, enrollment_encoded]
            # ------------------------------
            khalil_features = np.array([[
                self.safe_float(getattr(client, "cancellation_month", 0)),
                self.safe_float(getattr(client, "cancellation_year", 0)),
                self.safe_float(client.distance),
                self.safe_float(getattr(client, "dollar_cost_points_redeemed", 0)),
                self.safe_float(getattr(client, "flight_month", 1)),
                self.safe_float(client.points_redeemed),
                self.safe_float(client.total_flights),
                self.safe_float(encoded.get("enrollment_encoded", 0)),
            ]], dtype=float)
            
            

            # ------------------------------
            # 3) Predict cluster (requires scaler + kmeans)
            # ------------------------------
            if scaler and kmeans:
                X_scaled = scaler.transform(khalil_features)
                cluster_id = int(kmeans.predict(X_scaled)[0])

                cluster_names = {
                    0: "ACTIVE TRAVELERS",
                    1: "LOYALTY CHAMPIONS",
                    2: "UNSTABLE"
                }
                cluster_descriptions = {
                    0: "Frequent travelers with low loyalty – focus on engagement",
                    1: "Heavy loyalty points users – premium customers",
                    2: "Irregular travelers at churn risk – retention needed"
                }

                prediction.khalil_cluster_id = cluster_id
                prediction.khalil_cluster_name = cluster_names.get(cluster_id, "Unknown")
                prediction.khalil_cluster_description = cluster_descriptions.get(cluster_id, "")

                print(f"   ✓ Cluster: {prediction.khalil_cluster_name}")
            else:
                print("   ⚠️ Missing scaler or kmeans; skipping cluster prediction")
                return  # cluster is required for the rest

            # ------------------------------
            # 4) Predict preferred season (uses 5 features + cluster)
            # season_knn expects:
            # [Flight_Month, Distance, Points_Redeemed, Total_Flights, cluster]
            # ------------------------------
            season_mapping = {
                0: "Spring",
                1: "Summer",
                2: "Autumn",
                3: "Winter"
            }
            default_seasons = {0: "Summer", 1: "Spring", 2: "Winter"}

            if season_knn:
                season_features = np.array([[
                    khalil_features[0][4],   # Flight_Month
                    khalil_features[0][2],   # Distance
                    khalil_features[0][5],   # Points_Redeemed
                    khalil_features[0][6],   # Total_Flights
                    cluster_id               # cluster
                ]])

                season_pred = season_knn.predict(season_features)[0]
                if isinstance(season_pred, (int, np.integer)):
                    season_label = season_mapping.get(int(season_pred), "Autumn")
                else:
                    season_label = str(season_pred)

                prediction.khalil_preferred_season = season_label
                print(f"   ✓ Preferred Season: {season_label}")
            else:
                # fallback
                prediction.khalil_preferred_season = default_seasons.get(cluster_id, "Autumn")
                print(f"   ⚠️ season_knn missing; fallback season: {prediction.khalil_preferred_season}")

            # ------------------------------
            # 5) Optional: similar neighbors (knn)
            # ------------------------------
            if knn and scaler:
                distances, indices = knn.kneighbors(X_scaled, n_neighbors=5)
                neighbors = indices[0].tolist()
                prediction.khalil_similar_neighbors = neighbors  # store list (or stringify)
                print(f"   ✓ Similar Neighbors: {neighbors}")
            else:
                print("   ⚠️ knn missing; skipping neighbors")

        except Exception as e:
            print(f"   ❌ KHALIL error: {e}")

    def run_molka_models(self, client, prediction):
        """
        MOLKA - Loyalty Program & Tier Progression
        """
        print("🔵 Running MOLKA models...")
        encoded = client.get_encoded_values()
        
        try:
            # Load Random Forest model and transformers
            print("   📂 Loading models from:", os.path.join(self.models_dir, 'molka'))
            rf_model = self.load_pkl(os.path.join(self.models_dir, 'molka', 'random_forest_model.pkl'))
            print(f"   ✓ RF Model loaded: {rf_model is not None}")
            pt = self.load_pkl(os.path.join(self.models_dir, 'molka', 'power_transformer.pkl'))
            print(f"   ✓ PowerTransformer loaded: {pt is not None}")
            poly = self.load_pkl(os.path.join(self.models_dir, 'molka', 'polynomial_features.pkl'))
            print(f"   ✓ PolynomialFeatures loaded: {poly is not None}")
            scaler = self.load_pkl(os.path.join(self.models_dir, 'molka', 'standard_scaler.pkl'))
            print(f"   ✓ StandardScaler loaded: {scaler is not None}")
            
            if rf_model is not None and pt is not None and poly is not None and scaler is not None:
                # Get CLV from prediction (already calculated by HABIBA)
                clv = prediction.clv_prediction if prediction.clv_prediction else 0
                
                # Prepare 12 base features BEFORE transformation
                # Handle None values for cancellation_year and cancellation_month
                cancel_year = client.cancellation_year if client.cancellation_year is not None else 0
                cancel_month = client.cancellation_month if client.cancellation_month is not None else 0
                
                features_12 = np.array([[
                    client.satisfaction,                # satisfaction
                    clv,                                # CLV
                    client.rating_value,                # Rating_value
                    client.salary,                      # salary
                    encoded['marital_status_encoded'],  # marital_status_encoded
                    encoded['education_encoded'],       # education_encoded
                    encoded['class_encoded'],           # class_encoded
                    cancel_year,                        # Cancellation_Year
                    cancel_month,                       # Cancellation_Month
                    encoded['travel_type_encoded'],     # travel_type_encoded
                    encoded['city_encoded'],            # city_encoded
                    encoded['province_encoded'],        # province_encoded
                ]])
                
                # Apply transformation pipeline: PowerTransformer → PolynomialFeatures → StandardScaler
                X_transformed = pt.transform(features_12)
                X_poly = poly.transform(X_transformed)  # 12 features → 90 features
                X_scaled = scaler.transform(X_poly)
                
                # Predict tier (0=Star, 1=Nova, 2=Aurora)
                prediction_result = rf_model.predict(X_scaled)
                predicted_tier = int(prediction_result[0])
                
                # Tier mapping
                tier_mapping = {0: 'Star', 1: 'Nova', 2: 'Aurora'}
                
                # Get current tier
                current_tier_value = encoded['tier_encoded']  # 0=Star, 1=Nova, 2=Aurora
                current_tier_name = tier_mapping.get(current_tier_value, 'Star')
                
                # Store predicted tier as score
                prediction.tier_progression_score = float(predicted_tier)
                
                # Determine progression (only progression or remains, no downgrade)
                if predicted_tier > current_tier_value:
                    # Progress to higher tier
                    next_tier_name = tier_mapping.get(predicted_tier, 'Aurora')
                    prediction.tier_progression_label = f"Progression to {next_tier_name}"
                else:
                    # Stays at current tier (includes equal and lower predictions)
                    prediction.tier_progression_label = f"Remains at {current_tier_name}"
                
                print(f"   ✓ Current Tier: {current_tier_name} (value: {current_tier_value})")
                print(f"   ✓ Predicted Tier: {tier_mapping.get(predicted_tier)} (value: {predicted_tier})")
                print(f"   ✓ Status: {prediction.tier_progression_label}")
            else:
                print("   ❌ One or more models failed to load!")
                print(f"      RF Model: {rf_model is not None}")
                print(f"      PowerTransformer: {pt is not None}")
                print(f"      PolynomialFeatures: {poly is not None}")
                print(f"      StandardScaler: {scaler is not None}")
                prediction.tier_progression_score = 0.0
                prediction.tier_progression_label = "Unable to Load Models"
        
        except Exception as e:
            print(f"   ❌ MOLKA DS1 error: {e}")
            # Set default values on error
            prediction.tier_progression_score = 0.0
            prediction.tier_progression_label = "Unable to Calculate"
    
    def run_molka_ds2_models(self, client, prediction):
        """
        MOLKA DS2 - Identify Accelerators of Tier Upgrade
        Predicts if client will upgrade to next tier
        """
        print("🟣 Running MOLKA DS2 models (Tier Upgrade Accelerators)...")
        encoded = client.get_encoded_values()
        
        try:
            # Load DS2 model
            ds2_package = self.load_pkl(os.path.join(self.models_dir, 'molka', 'best_model_ds2.pkl'))
            
            if ds2_package is not None and 'best_model' in ds2_package:
                rf_model_ds2 = ds2_package['best_model']
                config = ds2_package.get('prediction_config', {})
                
                # Get CLV - if not available from HABIBA, calculate approximate CLV
                if prediction.clv_prediction and prediction.clv_prediction > 0:
                    clv = prediction.clv_prediction
                else:
                    # Calculate approximate CLV based on client data
                    # Simple formula: (Salary * 0.1) + (Points * 0.5) + (Distance * 2)
                    clv = (client.salary * 0.1) + (client.points_accumulated * 0.5) + (client.distance * 2)
                    print(f"   ⚠️  Using approximate CLV: ${clv:,.0f} (HABIBA model not available)")
                
                # Calculate feature engineering (4 derived features)
                avg_points_per_flight = client.points_accumulated / client.total_flights if client.total_flights > 0 else 0
                redemption_ratio = client.points_redeemed / client.points_accumulated if client.points_accumulated > 0 else 0
                clv_per_flight = clv / client.total_flights if client.total_flights > 0 else 0
                
                # Get median distance from config or use default
                median_distance = config.get('distance_median', 5000.0)
                long_distance_flag = 1 if client.distance > median_distance else 0
                
                # Prepare 11 features (7 base + 4 engineered)
                features_11 = np.array([[
                    encoded['class_encoded'],           # class_encoded
                    client.rating_value,                # Rating_value
                    client.satisfaction,                # satisfaction
                    clv,                                # CLV
                    client.salary,                      # salary
                    encoded['marital_status_encoded'],  # marital_status_encoded
                    encoded['education_encoded'],       # education_encoded
                    avg_points_per_flight,             # avg_points_per_flight
                    redemption_ratio,                  # redemption_ratio
                    clv_per_flight,                    # clv_per_flight
                    long_distance_flag,                # long_distance_flag
                ]])
                
                # Get feature importances and actual client values
                feature_names = [
                    "Class", "Rating", "Satisfaction", "CLV", "Salary",
                    "Marital Status", "Education", "Points/Flight",
                    "Redemption Ratio", "CLV/Flight", "Long Distance"
                ]
                
                importances = rf_model_ds2.feature_importances_
                client_values = features_11[0]
                
                # Create feature data with model importances (these are the real percentages)
                feature_data = []
                for i, name in enumerate(feature_names):
                    feature_data.append({
                        'name': name,
                        'value': client_values[i],
                        'importance_pct': importances[i] * 100
                    })
                
                # Sort by model importance to get the most important features globally
                feature_data.sort(key=lambda x: x['importance_pct'], reverse=True)
                
                # Format top 3 with model importance percentages and client values
                top_accelerators = []
                for feat in feature_data[:3]:
                    name = feat['name']
                    value = feat['value']
                    pct = feat['importance_pct']
                    
                    if name in ["Class", "Marital Status", "Education"]:
                        # Categorical - show importance percentage only
                        top_accelerators.append(f"{name} ({pct:.1f}%)")
                    elif name == "Long Distance":
                        status = "Yes" if value == 1 else "No"
                        top_accelerators.append(f"Long Distance: {status} ({pct:.1f}%)")
                    elif name in ["Redemption Ratio"]:
                        top_accelerators.append(f"{name}: {value:.1%} ({pct:.1f}%)")
                    elif name in ["CLV", "Salary"]:
                        top_accelerators.append(f"{name}: ${value:,.0f} ({pct:.1f}%)")
                    elif name in ["Points/Flight", "CLV/Flight"]:
                        top_accelerators.append(f"{name}: {value:.1f} ({pct:.1f}%)")
                    else:
                        top_accelerators.append(f"{name}: {value:.2f} ({pct:.1f}%)")
                
                # Predict upgrade probability
                upgrade_proba = rf_model_ds2.predict_proba(features_11)[0][1]
                
                # Store results
                prediction.tier_upgrade_probability = float(upgrade_proba)
                prediction.tier_upgrade_prediction = " • ".join(top_accelerators)
                
                print(f"   ✓ Upgrade Probability: {upgrade_proba:.2%}")
                print(f"   ✓ Top Accelerators: {', '.join(top_accelerators)}")
            else:
                print("   ❌ DS2 model or package structure invalid")
                prediction.tier_upgrade_probability = 0.0
                prediction.tier_upgrade_prediction = "Unable to Load Model"
        
        except Exception as e:
            print(f"   ❌ MOLKA DS2 error: {e}")
            import traceback
            traceback.print_exc()
            # Set default values on error
            prediction.tier_upgrade_probability = 0.0
            prediction.tier_upgrade_prediction = "Unable to Calculate"
            prediction.tier_progression_score = 0.0
            prediction.tier_progression_label = "Unable to Calculate"
    
    def determine_potential(self, prediction):
        """Determine overall potential based on all predictions"""
        try:
            # High potential criteria
            if (prediction.segmentation_label == "Loyal Premium" and 
                prediction.churn_classification == 0 and
                prediction.highly_engaged == 1):
                prediction.potential_category = 'High'
            # Low potential criteria
            elif (prediction.churn_classification == 1 or
                  prediction.engagement_label == "Low Engagement"):
                prediction.potential_category = 'Low'
            else:
                prediction.potential_category = 'Medium'
        except:
            prediction.potential_category = 'Medium'
    
    def run_all_predictions(self, client):
        """
        Run all ML models for a client
        This is called automatically when a client is created
        """
        print(f"\n{'='*60}")
        print(f"🚀 Running ML predictions for: {client.get_full_name()}")
        print(f"{'='*60}\n")
        
        # Get or create prediction record
        prediction, created = MLPrediction.objects.get_or_create(client=client)
        
        try:
            # Run all models
            self.run_habiba_models(client, prediction)
            self.run_adhem_models(client, prediction)
            self.run_asma_models(client, prediction)
            self.run_wajd_models(client, prediction)
            # TODO: self.run_khalil_models(client, prediction)
            self.run_khalil_models(client, prediction)
            self.run_molka_models(client, prediction)
            self.run_molka_ds2_models(client, prediction)
            # Determine overall potential
            self.determine_potential(prediction)
            
            # Mark as completed
            prediction.predictions_completed = True
            prediction.save()
            
            print(f"\n{'='*60}")
            print(f"✅ All predictions completed successfully!")
            print(f"{'='*60}\n")
            
            return prediction
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ Error running predictions: {str(e)}")
            print(f"{'='*60}\n")
            prediction.predictions_completed = False
            prediction.save()
            return None


# Singleton instance
ml_service = MLPredictionService()