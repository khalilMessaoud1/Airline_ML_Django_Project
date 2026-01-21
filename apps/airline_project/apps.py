from django.apps import AppConfig


class AirlineProjectConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.airline_project'

    def ready(self):
        import apps.airline_project.signals 
