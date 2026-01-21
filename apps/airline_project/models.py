from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from datetime import datetime


class Client(models.Model):
    """Customer/Client Information - Based on actual dataset"""
    
    # Basic Information
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    
    # Demographics
    gender = models.CharField(max_length=1, choices=[('M', 'Male'), ('F', 'Female')])
    education = models.CharField(max_length=50, choices=[
        ('High School or Below', 'High School or Below'),
        ('College', 'College'),
        ('Bachelor', 'Bachelor'),
        ('Master', 'Master'),
        ('Doctor', 'Doctor'),
    ])
    marital_status = models.CharField(max_length=1, choices=[
        ('S', 'Single'),
        ('M', 'Married'),
        ('D', 'Divorced'),
    ])
    salary = models.FloatField(validators=[MinValueValidator(0)])
    
    # Location
    province = models.CharField(max_length=100)
    city = models.CharField(max_length=100)
    
    # Loyalty Program
    tier = models.CharField(max_length=20, choices=[
        ('Star', 'Star'),
        ('Nova', 'Nova'),
        ('Aurora', 'Aurora'),
    ])
    enrollment_type = models.CharField(max_length=50, choices=[
        ('Standard', 'Standard'),
        ('2018 Promotion', '2018 Promotion'),
    ])
    enrollment_date = models.DateField()
    cancellation_date = models.DateField(null=True, blank=True)
    
    # Flight Data
    total_flights = models.IntegerField(default=0)
    distance = models.FloatField(default=0.0)  # Total distance traveled
    flight_class = models.CharField(max_length=20, choices=[
        ('Eco', 'Economy'),
        ('Eco Plus', 'Economy Plus'),
        ('Business', 'Business'),
    ])
    travel_type = models.CharField(max_length=20, choices=[
        ('Personal Travel', 'Personal Travel'),
        ('Business travel', 'Business Travel'),
    ])
    
    # Points & Rewards
    points_accumulated = models.FloatField(default=0.0)
    points_redeemed = models.FloatField(default=0.0)
    dollar_cost_points_redeemed = models.FloatField(default=0.0)
    
    # Satisfaction & Rating
    satisfaction = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        default=3
    )
    rating_value = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(5)],
        default=3.0
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Client'
        verbose_name_plural = 'Clients'
    
    def __str__(self):
        return f"{self.first_name} {self.last_name} ({self.email})"
    
    def get_full_name(self):
        return f"{self.first_name} {self.last_name}"
    
    @property
    def enrollment_duration(self):
        """Calculate enrollment duration in days"""
        end_date = self.cancellation_date if self.cancellation_date else datetime.now().date()
        return (end_date - self.enrollment_date).days
    
    @property
    def enrollment_year(self):
        return self.enrollment_date.year
    
    @property
    def enrollment_month(self):
        return self.enrollment_date.month
    
    @property
    def cancellation_year(self):
        return self.cancellation_date.year if self.cancellation_date else None
    
    @property
    def cancellation_month(self):
        return self.cancellation_date.month if self.cancellation_date else None
    
    def get_encoded_values(self):
        """Get all encoded values for ML models"""

        # Province Encoding
        province_encoding = {
            'Ontario': 0,
            'British Columbia': 1,
            'Quebec': 2,
            'Alberta': 3,
            'Manitoba': 4,
            'New Brunswick': 5,
            'Nova Scotia': 6,
            'Saskatchewan': 7,
            'Newfoundland': 8,
            'Yukon': 9,
            'Prince Edward Island': 10,
        }

        # City Encoding
        city_encoding = {
            'Toronto': 0,
            'Edmonton': 1,
            'Vancouver': 2,
            'Hull': 3,
            'Whitehorse': 4,
            'Trenton': 5,
            'Montreal': 6,
            'Dawson Creek': 7,
            'Quebec City': 8,
            'Fredericton': 9,
            'Ottawa': 10,
            'Tremblant': 11,
            'Calgary': 12,
            'Thunder Bay': 13,
            'Whistler': 14,
            'Peace River': 15,
            'Winnipeg': 16,
            'Sudbury': 17,
            'West Vancouver': 18,
            'Halifax': 19,
            'London': 20,
            'Regina': 21,
            'Kelowna': 22,
            "St. John's": 23,
            'Victoria': 24,
            'Kingston': 25,
            'Banff': 26,
            'Moncton': 27,
            'Charlottetown': 28,
        }

        # Encodings
        encodings = {
            # Province & City
            'province_encoded': province_encoding.get(self.province, 0),
            'city_encoded': city_encoding.get(self.city, 0),

            # Education (Ordinal)
            'education_encoded': {
                'High School or Below': 0,
                'College': 1,
                'Bachelor': 2,
                'Master': 3,
                'Doctor': 4,
            }.get(self.education, 0),

            # Gender (Binary)
            'gender_encoded': 1 if self.gender == 'M' else 0,

            # Marital Status (Label)
            'marital_status_encoded': {
                'D': 0,  # Divorced
                'M': 1,  # Married
                'S': 2,  # Single
            }.get(self.marital_status, 2),

            # Flight Class (Ordinal)
            'class_encoded': {
                'Eco': 0,
                'Eco Plus': 1,
                'Business': 2,
            }.get(self.flight_class, 0),

            # Travel Type (Binary)
            'travel_type_encoded': 1 if self.travel_type == 'Business travel' else 0,

            # Loyalty Tier (Ordinal)
            'tier_encoded': {
                'Star': 0,
                'Nova': 1,
                'Aurora': 2,
            }.get(self.tier, 0),

            # Enrollment Type (Binary)
            'enrollment_encoded': 1 if self.enrollment_type == '2018 Promotion' else 0,
        }

        return encodings



class MLPrediction(models.Model):
    """Store all ML model predictions for a client"""
    
    client = models.OneToOneField(Client, on_delete=models.CASCADE, related_name='predictions')
    
    # HABIBA - Segmentation & CLV
    segmentation_cluster = models.IntegerField(null=True, blank=True)  # 0 or 1
    segmentation_label = models.CharField(max_length=50, blank=True)  # "Loyal Premium" or "Basic Economy"
    clv_prediction = models.FloatField(null=True, blank=True)  # Predicted CLV value
    
    # ADHEM - Churn Predictions
    churn_classification = models.IntegerField(null=True, blank=True)  # 0 or 1
    churn_risk_label = models.CharField(max_length=50, blank=True)  # "Will Churn" or "Won't Churn"
    churn_months = models.FloatField(null=True, blank=True)  # Number of months until churn
    
    # ASMA - Enrollment Insights
    highly_engaged = models.IntegerField(null=True, blank=True)  # 0 or 1
    engagement_label = models.CharField(max_length=50, blank=True)
    equity_segment_cluster = models.IntegerField(null=True, blank=True)  # 1, 2, or 3
    equity_segment_label = models.CharField(max_length=100, blank=True)
    
    # KHALIL - Travel Behavior (TODO: Will add later)

    khalil_cluster_id = models.IntegerField(null=True, blank=True)
    khalil_cluster_name = models.CharField(max_length=50, null=True, blank=True)
    khalil_cluster_description = models.TextField(null=True, blank=True)
    khalil_preferred_season = models.CharField(max_length=20, null=True, blank=True)
    khalil_similar_neighbors = models.JSONField(null=True, blank=True)
    # WAJD - Loyalty Points
    loyalty_points_classification = models.IntegerField(null=True, blank=True)  # 0 or 1
    loyalty_cluster = models.IntegerField(null=True, blank=True)  # 0 or 1
    loyalty_cluster_label = models.CharField(max_length=100, blank=True)
    points_forecast = models.FloatField(null=True, blank=True)  # SARIMA forecast
    # MOLKA - Loyalty Program & Tier Progression (DS1)
    tier_progression_score = models.FloatField(null=True, blank=True)  # Score de progression tier
    tier_progression_label = models.CharField(max_length=50, blank=True)  # Label basé sur le score
    
    # MOLKA - Tier Upgrade Accelerators (DS2)
    tier_upgrade_probability = models.FloatField(null=True, blank=True)  # Probabilité de progression (0-1)
    tier_upgrade_prediction = models.CharField(max_length=50, blank=True)  # "Will Upgrade" ou "Will Not Upgrade"
    # Summary Fields
    risk_level = models.CharField(max_length=20, choices=[
        ('Low', 'Low Risk'),
        ('Medium', 'Medium Risk'),
        ('High', 'High Risk'),
    ], default='Low')
    
    potential_category = models.CharField(max_length=20, choices=[
        ('Low', 'Low Potential'),
        ('Medium', 'Medium Potential'),
        ('High', 'High Potential'),
    ], default='Medium')
    
    # Metadata
    predictions_completed = models.BooleanField(default=False)
    prediction_date = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)
    
    # Store raw prediction data as JSON
    raw_predictions = models.JSONField(default=dict, blank=True)
    
    class Meta:
        verbose_name = 'ML Prediction'
        verbose_name_plural = 'ML Predictions'
    
    def __str__(self):
        return f"Predictions for {self.client.get_full_name()}"
    
    def is_high_potential(self):
        """Check if client is high potential"""
        return self.potential_category == 'High'
    
    def is_high_risk(self):
        """Check if client is high churn risk"""
        return self.churn_classification == 1
    
    def get_summary(self):
        """Get a summary of all predictions"""
        return {
            'segmentation': self.segmentation_label or 'N/A',
            'clv': f"${self.clv_prediction:,.2f}" if self.clv_prediction else "N/A",
            'churn_risk': self.churn_risk_label or 'N/A',
            'churn_months': f"{self.churn_months:.1f} months" if self.churn_months else "N/A",
            'engagement': self.engagement_label or 'N/A',
            'equity_segment': f"Group {self.equity_segment_cluster}" if self.equity_segment_cluster else "N/A",
            'loyalty_cluster': self.loyalty_cluster_label or 'N/A',
            'potential': self.potential_category,
            'risk': self.risk_level,
        }