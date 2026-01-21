from .utils_roles import get_user_role, get_allowed_menu

def airline_role_menu(request):
    return {
        "user_role": get_user_role(request.user),
        "allowed_menu": get_allowed_menu(request.user),
    }
