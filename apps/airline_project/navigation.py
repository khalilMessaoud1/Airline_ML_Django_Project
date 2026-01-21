# apps/airline_project/navigation.py

MENU_ITEMS = [
    {
        "key": "lpu",
        "label": "Loyalty Points Utilization",
        "url_name": "loyalty-points",
        "icon": "tim-icons icon-chart-pie-36",
        "active": "loyalty-points",
    },
    {
        "key": "cr",
        "label": "Churn Reduction",
        "url_name": "churn-reduction",
        "icon": "tim-icons icon-chart-bar-32",
        "active": "churn-reduction",
    },
    {
        "key": "tba",
        "label": "Travel Behavior Analysis",
        "url_name": "travel-behavior",
        "icon": "tim-icons icon-pin",
        "active": "travel-behavior",
    },
    {
        "key": "clv",
        "label": "Customer Lifetime Value",
        "url_name": "customer-lifetime",
        "icon": "tim-icons icon-coins",
        "active": "customer-lifetime",
    },
    {
        "key": "lp",
        "label": "Loyalty Program",
        "url_name": "loyalty-program",
        "icon": "tim-icons icon-trophy",
        "active": "loyalty-program",
    },
    {
        "key": "ei",
        "label": "Enrollment Insights",
        "url_name": "enrollment-insights",
        "icon": "tim-icons icon-single-02",
        "active": "enrollment-insights",
    },
]


ROLE_ALLOWED_KEYS = {
    "admin": {"lpu", "tba", "cr", "lp", "clv", "ei"},
    "marketing_lead": {"lpu", "tba", "cr", "lp", "clv"},
    "finance_lead": {"lpu", "clv", "cr"},
}
