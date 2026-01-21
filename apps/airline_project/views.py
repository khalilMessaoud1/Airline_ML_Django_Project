from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Client, MLPrediction
from .ml_service import ml_service
import os, pickle
from django.conf import settings
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.views import View
from django.contrib.auth.decorators import login_required
from .utils_roles import redirect_by_role
# from .utils import redirect_by_role
# Power BI iframe URLs for each dashboard
POWERBI_IFRAMES = {
    'loyalty-points': 'https://app.powerbi.com/view?r=eyJrIjoiOTMzYjU3OTktOTY4ZC00YjFkLWJjOGItOTY4ZjczMGEyNGVkIiwidCI6ImRiZDY2NjRkLTRlYjktNDZlYi05OWQ4LTVjNDNiYTE1M2M2MSIsImMiOjl9',
    'churn-reduction': 'https://app.powerbi.com/view?r=eyJrIjoiYzY0M2UwNGMtM2NmMy00MjM0LWEzOTAtZDYzYmZkNmQ2M2I0IiwidCI6ImRiZDY2NjRkLTRlYjktNDZlYi05OWQ4LTVjNDNiYTE1M2M2MSIsImMiOjl9',
    'travel-behavior': 'https://app.powerbi.com/view?r=eyJrIjoiNTRkM2Y2ZDEtMDk0MC00ZDNkLTllNDItY2NiMjRlZTA2ZjU5IiwidCI6ImRiZDY2NjRkLTRlYjktNDZlYi05OWQ4LTVjNDNiYTE1M2M2MSIsImMiOjl9',
    'customer-lifetime': 'https://app.powerbi.com/view?r=eyJrIjoiM2VjZTkyNDItNTVmNi00ZGNkLWExMWUtZGM0MzhiMjkzNDRjIiwidCI6ImRiZDY2NjRkLTRlYjktNDZlYi05OWQ4LTVjNDNiYTE1M2M2MSIsImMiOjl9',
    'loyalty-program': 'https://app.powerbi.com/view?r=eyJrIjoiNmQyNWJmNTYtY2NjZS00ZTM3LWFlMDctMWViNjczZGEwMzEzIiwidCI6ImRiZDY2NjRkLTRlYjktNDZlYi05OWQ4LTVjNDNiYTE1M2M2MSIsImMiOjl9',
    'enrollment-insights': 'https://app.powerbi.com/view?r=eyJrIjoiYTA1NjU1OGItODhlYS00YmNmLWI3MjktZTQ1NGMzNzA0YjViIiwidCI6ImRiZDY2NjRkLTRlYjktNDZlYi05OWQ4LTVjNDNiYTE1M2M2MSIsImMiOjl9',
}

def after_login(request):
    return redirect_by_role(request.user)

def no_role(request):
    return render(request, "airline_project/no_role.html")
def create_client(request):
    """Create a new client and run all ML predictions"""

    if request.method == 'POST':
        try:
            # Parse date fields
            from datetime import datetime
            enrollment_date = datetime.strptime(request.POST.get('enrollment_date'), '%Y-%m-%d').date()
            cancellation_date = request.POST.get('cancellation_date')
            if cancellation_date:
                cancellation_date = datetime.strptime(cancellation_date, '%Y-%m-%d').date()
            else:
                cancellation_date = None
            
            # Create client
            client = Client.objects.create(
                first_name=request.POST.get('first_name'),
                last_name=request.POST.get('last_name'),
                email=request.POST.get('email'),
                gender=request.POST.get('gender'),
                education=request.POST.get('education'),
                marital_status=request.POST.get('marital_status'),
                salary=float(request.POST.get('salary')),
                province=request.POST.get('province'),
                city=request.POST.get('city'),
                tier=request.POST.get('tier'),
                enrollment_type=request.POST.get('enrollment_type'),
                enrollment_date=enrollment_date,
                cancellation_date=cancellation_date,
                total_flights=int(request.POST.get('total_flights')),
                distance=float(request.POST.get('distance')),
                flight_class=request.POST.get('flight_class'),
                travel_type=request.POST.get('travel_type'),
                points_accumulated=float(request.POST.get('points_accumulated')),
                points_redeemed=float(request.POST.get('points_redeemed')),
                dollar_cost_points_redeemed=float(request.POST.get('dollar_cost_points_redeemed', 0)),
                satisfaction=int(request.POST.get('satisfaction')),
                rating_value=float(request.POST.get('rating_value')),
            )
            
            messages.success(request, f'✅ Client {client.get_full_name()} created successfully! Running ML predictions...')
            
            # Redirect to client detail page
            return redirect('client-detail', client_id=client.id)
            
        except Exception as e:
            messages.error(request, f'❌ Error creating client: {str(e)}')
            return redirect('create-client')
    
    context = {
        'segment': 'create-client',
        'active_page': 'create-client',
        'page_title': 'Create New Client',
    }
    return render(request, 'airline_project/create_client.html', context)


def client_detail(request, client_id):
    """View client details and ML predictions"""
    try:
        client = Client.objects.get(id=client_id)
        
        # Get predictions (will be created by signal)
        try:
            predictions = client.predictions
        except MLPrediction.DoesNotExist:
            predictions = None
        
        context = {
            'segment': 'client-detail',
            'active_page': 'client-detail',
            'page_title': f'Client: {client.get_full_name()}',
            'client': client,
            'predictions': predictions,
        }
        return render(request, 'airline_project/client_detail.html', context)
    
    except Client.DoesNotExist:
        messages.error(request, 'Client not found')
        return redirect('create-client')


def clients_list(request):
    """List all clients"""
    clients = Client.objects.all()
    
    context = {
        'segment': 'clients-list',
        'active_page': 'clients-list',
        'page_title': 'All Clients',
        'clients': clients,
    }
    return render(request, 'airline_project/clients_list.html', context)


def loyalty_points(request):
    """Loyalty Points Utilization - Dashboard + ML Models"""
    context = {
    # Page meta (KEEP OUT of models array)
    "segment": "loyalty-points",
    "active_page": "loyalty-points",
    "page_title": "Loyalty Points Utilization",
    "powerbi_url": POWERBI_IFRAMES["loyalty-points"],

    # Dynamic ML models (LOOPED in template)
    "models": [
        {
            "title": "Redemption Likelihood",
            "icon": "tim-icons icon-chart-pie-36",
            "subtitle": "CUSTOMER REDEMPTION POTENTIAL",
            "description": "Assess each customer’s likelihood to redeem loyalty points, helping prioritize outreach efforts and personalize offers to maximize program effectiveness and return on investment.",
            "features": [
                "Prioritization of high-impact campaigns",
                "Personalized engagement opportunities",
                "Customers most likely to redeem soon or not",
            ],
            "url_name": "clients-list",
        },
        {
            "title": "Loyalty Points Forecasting",
            "icon": "tim-icons icon-chart-bar-32",
            "subtitle": "UTILIZATION TRENDS & SEASONALITY",
            "description": "Anticipate how customers will use loyalty points over time to uncover seasonal patterns and demand cycles. This insight supports smarter planning of promotions, rewards availability, and marketing campaigns.",
            "features": [
                "Seasonal peaks and low-usage periods",
                "Forward-looking utilization trends",
                "Better timing for promotional campaigns",
            ],
            "url_name": "sarima_forecast",
        },
        {
            "title": "Redemption Behavior Segmentation",
            "icon": "tim-icons icon-chart-bar-32",
            "subtitle": "CUSTOMER SEGMENTATION",
            "description": "Group customers based on how they redeem loyalty points to better understand usage patterns, preferences, and engagement levels. This enables tailored communication and differentiated reward strategies.",
            "features": [
                "Clear customer behavior groups:",
                "Active but Low-Intensity Flyers: Fly occasionally, accumulate points slowly but rarely redeem.",
                "Redemption-Focused Members: Low flight frequency but high number of points redeemed.",
            ],
            "url_name": "clients-list",
        },
    ],
}

    
    return render(request, 'airline_project/dashboard.html', context)


def churn_reduction(request):
    """Churn Reduction - Dashboard + ML Models"""
    context = {
        # Page meta (KEEP OUT of models array)
        "segment": "churn-reduction",
        "active_page": "churn-reduction",
        "page_title": "Churn Reduction",
        "powerbi_url": POWERBI_IFRAMES["churn-reduction"],

        # Dynamic ML models
        "models": [
            {
                "title": "Loyalty Membership Duration Forecast",
                "icon": "tim-icons icon-chart-pie-36",
                "subtitle": "EXPECTED CUSTOMER LIFETIME",
                "description": "Estimate how long new customers are likely to remain active in the loyalty program at the time of enrollment, and continuously refine this estimate as their engagement evolves. This insight enables proactive retention strategies and long-term value planning.",
                "features": [
                    "Expected loyalty membership duration",
                    "Early identification of short-term vs long-term members",
                    "Help you tailor retention strategies accordingly",
                ],
                "url_name": "clients-list",
            },
            {
                "title": "Cancellation Risk Assessment",
                "icon": "tim-icons icon-chart-bar-32",
                "subtitle": "LOYALTY PROGRAM RETENTION RISK",
                "description": "Evaluate the likelihood that active customers will cancel their loyalty membership in the future. This insight helps prioritize retention efforts, focus resources on at-risk customers, and reduce preventable churn.",
                "features": [
                    "Identification of customers at high risk of cancellation",
                    "Prioritized retention and outreach actions",
                    "Reduce churn through early intervention",
                ],
                "url_name": "clients-list",
            },
        ],
    }

    
    return render(request, 'airline_project/dashboard.html', context)


def travel_behavior(request):
    """Travel Behavior Analysis - Dashboard + ML Models"""
    context = {
        # Page meta (KEEP OUT of models array)
        "segment": "travel-behavior",
        "active_page": "travel-behavior",
        "page_title": "Travel Behavior Analysis",
        "powerbi_url": POWERBI_IFRAMES["travel-behavior"],

        # Dynamic ML models
        "models": [
            {
                "title": "Traveler Behavior Segmentation",
                "icon": "tim-icons icon-chart-pie-36",
                "subtitle": "FLIGHT ACTIVITY & TRAVEL PATTERNS",
                "description": "Group travelers based on how frequently they fly and the distances they travel in order to uncover distinct travel behavior profiles. These insights support more targeted offers, improved route planning, and personalized customer experiences.",
                "features": [
                    "Identification of frequent vs occasional travelers",
                    "To help improve personalization of offers and communications",
                    "Clear traveler profiles for targeted campaigns",
                ],
                "url_name": "clients-list",
            },
            {
                "title": "Seasonal Travel Preference Forecast",
                "icon": "tim-icons icon-chart-bar-32",
                "subtitle": "NEXT SEASON TRAVEL INSIGHTS",
                "description": "Anticipate the next travel season for each traveler group by analyzing historical travel patterns. This insight helps align marketing efforts, capacity planning, and promotional timing with expected demand.",
                "features": [
                    "Expected travel season per traveler group",
                    "Better timing of marketing and promotional campaigns",
                    "Support for capacity and route planning decisions",
                ],
                "url_name": "clients-list",
            },
        ],
    }

    return render(request, 'airline_project/dashboard.html', context)


def customer_lifetime(request):
    """Customer Lifetime Value - Dashboard + ML Models"""
    context = {
        # Page meta (KEEP OUT of models array)
        "segment": "customer-lifetime",
        "active_page": "customer-lifetime",
        "page_title": "Customer Lifetime Value",
        "powerbi_url": POWERBI_IFRAMES["customer-lifetime"],

        # Dynamic ML models
        "models": [
            {
                "title": "Customer Lifetime Value Forecast",
                "icon": "tim-icons icon-chart-pie-36",
                "subtitle": "FUTURE REVENUE POTENTIAL",
                "description": "Estimate the future value each customer is expected to generate based on their profile and travel behavior. This insight helps prioritize high-value customers, optimize marketing investments, and guide long-term growth strategies.",
                "features": [
                    "Expected long-term revenue per customer",
                    "Identification of high-value growth opportunities",
                    "Support for long-term strategic planning",
                ],
                "url_name": "clients-list",
            },
            {
                "title": "CLV-Based Customer Segmentation",
                "icon": "tim-icons icon-chart-bar-32",
                "subtitle": "VALUE-DRIVEN CUSTOMER GROUPS",
                "description": "Segment customers into value-based groups to clearly distinguish high, medium, and low lifetime value profiles. This enables differentiated engagement strategies and more effective use of resources.",
                "features": [
                    "Clear separation of high, medium, and low-value customers",
                    "Tailored engagement and retention strategies per segment",
                    "Improved return on customer acquisition and retention efforts",
                ],
                "url_name": "clients-list",
            },
        ],
    }

    return render(request, 'airline_project/dashboard.html', context)


def loyalty_program(request):
    """Loyalty Program - Dashboard + ML Models"""
    context = {
        # Page meta (KEEP OUT of models array)
        "segment": "loyalty-program",
        "active_page": "loyalty-program",
        "page_title": "Loyalty Program",
        "powerbi_url": POWERBI_IFRAMES["loyalty-program"],

        # Dynamic ML models
        "models": [
            {
                "title": "Classification",
                "icon": "tim-icons icon-chart-pie-36",
                "subtitle": "TIER CLASSIFICATION",
                "description": "Predict customer loyalty tier and upgrade probability.",
                "features": [
                    "Tier Prediction",
                    "Upgrade Probability",
                    "Member Engagement Score",
                ],
                "url_name": "clients-list",
            },
            {
                "title": "Regression",
                "icon": "tim-icons icon-chart-bar-32",
                "subtitle": "ENGAGEMENT SCORE PREDICTION",
                "description": "Estimate future member engagement and participation levels.",
                "features": [
                    "Engagement Metrics",
                    "Participation Rate",
                    "Benefit Utilization",
                ],
                "url_name": "clients-list",
            },
        ],
    }

    
    return render(request, 'airline_project/dashboard.html', context)


def enrollment_insights(request):
    """Enrollment Insights – Dashboard & ML Models"""

    context = {
        # Page meta (KEEP OUT of models array)
        "segment": "enrollment-insights",
        "active_page": "enrollment-insights",
        "page_title": "Enrollment Insights",
        "powerbi_url": POWERBI_IFRAMES["enrollment-insights"],

        # Dynamic ML models
        "models": [
            {
                "title": "Customer Segmentation",
                "icon": "tim-icons icon-chart-pie-36",
                "subtitle": "HIGH-POTENTIAL CUSTOMER SEGMENTATION",
                "description": (
                    "Segment customers based on region, income, and education level "
                    "to identify high-potential groups with a strong likelihood of "
                    "enrolling in the loyalty program."
                ),
                "features": [
                    "Region-Based Segmentation",
                    "Income and Education Profiling",
                    "High-Potential Customer Identification",
                    "Targeted Enrollment Strategy",
                ],
                "url_name": "clients-list",
            },
            {
                "title": "Engagement Classification",
                "icon": "tim-icons icon-chart-bar-32",
                "subtitle": "CUSTOMER ENGAGEMENT CLASSIFICATION",
                "description": (
                    "Classify customers as highly engaged or not engaged using "
                    "region, income, and education attributes to support "
                    "data-driven enrollment decision-making."
                ),
                "features": [
                    "Highly Engaged vs Not Engaged Classification",
                    "Socio-Demographic Analysis",
                    "Engagement Level Prediction",
                    "Enrollment Readiness Indicator",
                ],
                "url_name": "clients-list",
            },
        ],
    }

    return render(request, 'airline_project/dashboard.html', context)


def sarima_forecast(request):
    horizon = 3  # default

    if request.method == "POST":
        try:
            horizon = int(request.POST.get("horizon", 3))
        except ValueError:
            horizon = 3

    # path to your pkl (put it somewhere like: app_name/ml_models/sarima.pkl)
    model_path = os.path.join(settings.BASE_DIR, "ml_models", "wajd", "sarima_model.pkl")

    with open(model_path, "rb") as f:
        sarima_model = pickle.load(f)

    # Forecast
    # statsmodels SARIMAXResults typically supports: get_forecast(steps=n)
    pred = sarima_model.get_forecast(steps=horizon)
    mean = pred.predicted_mean
    ci = pred.conf_int()

    # Build a list of dicts to pass to template
    forecast = []
    for i in range(horizon):
        # mean index is often datetime-like; convert to string for template
        period = str(mean.index[i]) if hasattr(mean, "index") else f"t+{i+1}"

        # ci columns often named like 'lower y'/'upper y' or similar
        lower = float(ci.iloc[i, 0])
        upper = float(ci.iloc[i, 1])

        forecast.append({
            "period": period,
            "yhat": float(mean.iloc[i]),
            "lower": lower,
            "upper": upper,
        })
    # ----- Dynamic planning insight -----
    y = [float(v) for v in mean.values]
    idxs = list(mean.index)

    # Peak & low months
    peak_i = max(range(len(y)), key=lambda i: y[i])
    low_i  = min(range(len(y)), key=lambda i: y[i])

    peak_period = idxs[peak_i].strftime("%b %Y")
    low_period  = idxs[low_i].strftime("%b %Y")

    peak_value = y[peak_i]
    low_value  = y[low_i]

    # Overall trend
    first, last = y[0], y[-1]
    delta_pct = ((last - first) / first * 100) if first != 0 else 0

    if delta_pct > 5:
        trend_text = f"Overall utilization is increasing (~{delta_pct:.0f}% over the period)."
    elif delta_pct < -5:
        trend_text = f"Overall utilization is decreasing (~{abs(delta_pct):.0f}% over the period)."
    else:
        trend_text = "Overall utilization remains relatively stable."

    # Seasonality strength
    range_pct = ((peak_value - low_value) / low_value * 100) if low_value != 0 else 0

    if range_pct >= 40:
        seasonality_text = "Seasonality appears strong with significant variation between months."
    elif range_pct >= 20:
        seasonality_text = "Seasonality appears moderate with noticeable monthly variation."
    else:
        seasonality_text = "Seasonality appears mild with limited month-to-month variation."

    planning_insight = (
        f"Peak utilization is expected around {peak_period} "
        f"(≈ {peak_value:,.0f} points), while the lowest level is expected around {low_period} "
        f"(≈ {low_value:,.0f} points). "
        f"{trend_text} {seasonality_text}"
    )

    return render(request, "airline_project/sarima_forecast.html", {
        "horizon": horizon,
        "forecast": forecast,
        "planning_insight": planning_insight,
    })
# --------------------------------------------------
# class AuthSignin(View):
#     template_name = "airline_project/auth-signin.html"

#     def get(self, request):
#         # if already logged in, route them
#         if request.user.is_authenticated:
#             return redirect_by_role(request.user)
#         return render(request, self.template_name, {})

#     def post(self, request):
#         username = request.POST.get("username")
#         password = request.POST.get("password")

#         user = authenticate(request, username=username, password=password)
#         if user is None:
#             # keep same UI: just send an error message
#             return render(request, self.template_name, {"msg": "Invalid credentials"})

#         login(request, user)
#         return redirect_by_role(user)

