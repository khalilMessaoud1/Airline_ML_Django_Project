from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Client
from .ml_service import ml_service


@receiver(post_save, sender=Client)
def run_ml_predictions(sender, instance, created, **kwargs):
    """
    Automatically run all 13 ML models when a new client is created
    """
    if created:  # Only run for new clients
        print(f"🚀 New client created: {instance.get_full_name()}")
        print(f"📊 Running all 13 ML models...")
        ml_service.run_all_predictions(instance)