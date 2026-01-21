from django.urls import path
from . import views

urlpatterns = [
    path("after-login/", views.after_login, name="after_login"),

    path('loyalty-points/', views.loyalty_points, name='loyalty-points'),
    path('churn-reduction/', views.churn_reduction, name='churn-reduction'),
    path('travel-behavior/', views.travel_behavior, name='travel-behavior'),
    path('customer-lifetime/', views.customer_lifetime, name='customer-lifetime'),
    path('loyalty-program/', views.loyalty_program, name='loyalty-program'),
    path('enrollment-insights/', views.enrollment_insights, name='enrollment-insights'),
    path("sarima/", views.sarima_forecast, name="sarima_forecast"),
    # Client management
    path('create-client/', views.create_client, name='create-client'),
    path('client/<int:client_id>/', views.client_detail, name='client-detail'),
    path('clients/', views.clients_list, name='clients-list'),
    #customer signin role based redirection
    # path("accounts/auth-signin/", views.AuthSignin.as_view(), name="auth_signin"),
    path("accounts/no-role/", views.no_role, name="no_role"),
]