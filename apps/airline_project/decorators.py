# apps/airline_project/decorators.py
from functools import wraps
from django.http import HttpResponseForbidden
from .utils_roles import get_user_role
from .navigation import ROLE_ALLOWED_KEYS

def menu_key_required(menu_key):
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped(request, *args, **kwargs):
            role = get_user_role(request.user)
            if role == "admin":  # admin sees all
                return view_func(request, *args, **kwargs)

            allowed = ROLE_ALLOWED_KEYS.get(role, set())
            if menu_key in allowed:
                return view_func(request, *args, **kwargs)

            return HttpResponseForbidden("Access denied.")
        return _wrapped
    return decorator
