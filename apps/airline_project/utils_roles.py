# apps/airline_project/utils_roles.py
from django.shortcuts import redirect
from .navigation import MENU_ITEMS, ROLE_ALLOWED_KEYS

def get_user_role(user):
    if not user.is_authenticated:
        return None
    if user.is_superuser:
        return "admin"
    for role in ("admin", "marketing_lead", "finance_lead"):
        if user.groups.filter(name=role).exists():
            return role
    return None

def get_allowed_menu(user):
    role = get_user_role(user)
    if role is None:
        return []
    allowed = ROLE_ALLOWED_KEYS.get(role, set())
    return [item for item in MENU_ITEMS if item["key"] in allowed]

def redirect_by_role(user):
    role = get_user_role(user)
    if role == "admin":
        return redirect("clients-list")
    if role == "marketing_lead":
        return redirect("clients-list")
    if role == "finance_lead":
        return redirect("clients-list")
    return redirect("airline:no_role")
