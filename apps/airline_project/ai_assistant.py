import os
try:
    from groq import Groq
    _HAVE_GROQ = True
except Exception:
    Groq = None
    _HAVE_GROQ = False
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from .models import Client, MLPrediction

class AIAssistant:
    def __init__(self, client_id, objective=None):
        self.client_id = client_id
        self.objective = objective
        self.client = self._get_client()
        self.predictions = self._get_predictions()
        self.groq_client = self._initialize_groq()
        
    def _initialize_groq(self):
        """Initialize Groq client with API key from settings"""
        api_key = getattr(settings, 'GROQ_API_KEY', None)
        if not api_key:
            raise Exception("GROQ_API_KEY not found in settings. Please add it to your .env file.")
        if not _HAVE_GROQ:
            raise Exception(
                "groq Python package not installed. Install it in your environment, e.g. `pip install groq` "
                "or consult the project README for the correct package name."
            )
        return Groq(api_key=api_key)
    
    def _get_client(self):
        """Get Client object"""
        try:
            return Client.objects.get(id=self.client_id)
        except Client.DoesNotExist:
            raise Exception(f"Client with ID {self.client_id} not found in the database.")
    
    def _get_predictions(self):
        """Get MLPrediction object for this client"""
        try:
            return self.client.predictions
        except MLPrediction.DoesNotExist:
            return None
    
    def _get_objective_context(self):
        """Get context and focus areas based on objective"""
        contexts = {
            'churn': {
                'name': 'Churn Risk Analysis',
                'focus': 'Retention strategies and risk mitigation',
                'key_metrics': ['churn_classification', 'churn_risk_label', 'churn_months', 'risk_level']
            },
            'clv': {
                'name': 'Customer Lifetime Value Strategy',
                'focus': 'Value maximization and growth opportunities',
                'key_metrics': ['clv_prediction', 'segmentation_label', 'potential_category']
            },
            'loyalty': {
                'name': 'Loyalty Program Optimization',
                'focus': 'Points utilization and program engagement',
                'key_metrics': ['loyalty_points_classification', 'loyalty_cluster_label', 'points_forecast', 'tier_progression_score']
            },
            'travel': {
                'name': 'Travel Behavior Insights',
                'focus': 'Travel patterns and preferences optimization',
                'key_metrics': ['khalil_cluster_name', 'khalil_preferred_season', 'flight_class', 'travel_type']
            },
            'enrollment': {
                'name': 'Enrollment & Engagement Analysis',
                'focus': 'Program engagement and member satisfaction',
                'key_metrics': ['highly_engaged', 'engagement_label', 'equity_segment_label', 'tier_upgrade_probability']
            },
            None: {
                'name': 'Comprehensive Client Analysis',
                'focus': 'Holistic view of all key metrics and opportunities',
                'key_metrics': ['risk_level', 'potential_category', 'segmentation_label', 'clv_prediction']
            }
        }
        return contexts.get(self.objective, contexts[None])
    
    def _prepare_client_context(self):
        """Prepare client data and predictions for LLM context"""
        if not self.predictions:
            return {
                'client_info': self._get_client_info(),
                'predictions_available': False,
                'message': 'No ML predictions available for this client yet.'
            }
        
        context = {
            'client_info': self._get_client_info(),
            'predictions_available': True,
            'predictions': self._get_prediction_data(),
            'objective_context': self._get_objective_context()
        }
        return context
    
    def _get_client_info(self):
        """Get basic client information"""
        return {
            'name': self.client.get_full_name(),
            'email': self.client.email,
            'tier': self.client.tier,
            'enrollment_duration': f"{self.client.enrollment_duration} days",
            'total_flights': self.client.total_flights,
            'distance': f"{self.client.distance:,.0f} km",
            'points_accumulated': f"{self.client.points_accumulated:,.0f}",
            'points_redeemed': f"{self.client.points_redeemed:,.0f}"
        }
    
    def _get_prediction_data(self):
        """Get all relevant prediction data as a dictionary"""
        if not self.predictions:
            return {}
        
        return {
            # Habiba - Segmentation & CLV
            'segmentation_cluster': self.predictions.segmentation_cluster,
            'segmentation_label': self.predictions.segmentation_label,
            'clv_prediction': f"${self.predictions.clv_prediction:,.2f}" if self.predictions.clv_prediction else "N/A",
            
            # Adhem - Churn Predictions
            'churn_classification': self.predictions.churn_classification,
            'churn_risk_label': self.predictions.churn_risk_label,
            'churn_months': f"{self.predictions.churn_months:.1f} months" if self.predictions.churn_months else "N/A",
            'risk_level': self.predictions.risk_level,
            
            # Asma - Enrollment Insights
            'highly_engaged': self.predictions.highly_engaged,
            'engagement_label': self.predictions.engagement_label,
            'equity_segment_cluster': self.predictions.equity_segment_cluster,
            'equity_segment_label': self.predictions.equity_segment_label,
            
            # Khalil - Travel Behavior
            'khalil_cluster_id': self.predictions.khalil_cluster_id,
            'khalil_cluster_name': self.predictions.khalil_cluster_name,
            'khalil_preferred_season': self.predictions.khalil_preferred_season,
            
            # Wajd - Loyalty Points
            'loyalty_points_classification': self.predictions.loyalty_points_classification,
            'loyalty_cluster': self.predictions.loyalty_cluster,
            'loyalty_cluster_label': self.predictions.loyalty_cluster_label,
            'points_forecast': f"{self.predictions.points_forecast:,.0f}" if self.predictions.points_forecast else "N/A",
            
            # Molka - Tier Progression
            'tier_progression_score': f"{self.predictions.tier_progression_score:.1f}" if self.predictions.tier_progression_score else "N/A",
            'tier_progression_label': self.predictions.tier_progression_label,
            'tier_upgrade_probability': f"{self.predictions.tier_upgrade_probability * 100:.1f}%" if self.predictions.tier_upgrade_probability else "N/A",
            'tier_upgrade_prediction': self.predictions.tier_upgrade_prediction,
            
            # Summary fields
            'potential_category': self.predictions.potential_category,
        }
    
    def _format_prompt(self, context):
        """Format the prompt with actual client data"""
        objective_context = context['objective_context']
        client_info = context['client_info']
        
        # Format client info as readable string
        client_info_str = "\n".join([f"- {key.replace('_', ' ').title()}: {value}" for key, value in client_info.items()])
        
        # Format prediction data
        prediction_data = context.get('predictions', {})
        prediction_str = "\n".join([f"- {key.replace('_', ' ').title()}: {value}" for key, value in prediction_data.items()]) if prediction_data else "No prediction data available"
        
        # Base prompt that works well for chat
        base_prompt = f"""
You are an AI assistant for Flair Airlines specializing in customer analytics. You analyze customer data and ML predictions to provide actionable insights to airline staff.

Current Analysis Objective: {objective_context['name']}
Focus Area: {objective_context['focus']}

CLIENT INFORMATION:
{client_info_str}

ML PREDICTION RESULTS:
{prediction_str}

INSTRUCTIONS:
1. Provide concise, actionable insights based on the ML predictions
2. Focus on practical recommendations that airline staff can implement
3. Include specific numbers, timeframes, and concrete actions when possible
4. Keep responses professional but conversational (suitable for a chat interface)
5. Structure your response with a brief summary followed by key insights
6. Do not make up data that isn't provided in the context

Your response should be formatted for a chat interface - use concise paragraphs, bullet points when helpful, and emphasize key terms with asterisks for bold text.
"""
        
        return base_prompt
    
    def get_recommendation(self):
        """Get AI recommendation using Groq LLM"""
        try:
            context = self._prepare_client_context()
            
            if not context['predictions_available']:
                return {
                    'status': 'error',
                    'message': '⚠️ No ML predictions available for this client. Please run predictions first.',
                    'priority': 'low',
                    'metrics': {},
                    'full_analysis': 'Unable to generate recommendations without ML prediction data. Please ensure all ML models have been run for this client.'
                }
            
            prompt = self._format_prompt(context)
            
            # Call Groq API - using a faster model for chat interface
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional airline industry analyst providing strategic recommendations to airline executives. Be concise, data-driven, and action-oriented. Format responses for a chat interface with clear, scannable content."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.3-70b-versatile",  # Using Llama 3.1 70B model
                temperature=0.3,
                max_tokens=500  # Shorter responses for chat
            )
            
            analysis = chat_completion.choices[0].message.content
            
            # Determine priority based on risk and potential
            priority = self._determine_priority()
            
            # Extract key metrics for summary
            metrics = self._extract_key_metrics()
            
            # Generate a short summary message
            summary_message = self._generate_summary_message(priority)
            
            return {
                'status': 'success',
                'message': summary_message,
                'priority': priority,
                'metrics': metrics,
                'full_analysis': analysis
            }
            
        except Exception as e:
            error_message = str(e)
            if "Groq API key" in error_message or "authentication" in error_message.lower() or "invalid_api_key" in error_message.lower():
                error_message = "Groq API key not configured properly. Please check your .env file and restart the server."
            elif "rate limit" in error_message.lower():
                error_message = "Rate limit exceeded. Please try again in a few seconds."
            elif "network" in error_message.lower() or "timeout" in error_message.lower():
                error_message = "Network timeout. Please check your internet connection and try again."
            
            return {
                'status': 'error',
                'message': f'❌ Error generating recommendation: {error_message}',
                'priority': 'low',
                'metrics': {},
                'full_analysis': f'**Error Details:**\n{error_message}\n\n**Troubleshooting Steps:**\n1. Check your GROQ_API_KEY in .env file\n2. Ensure you have internet connectivity\n3. Verify Groq service is operational\n4. Check server logs for detailed error information'
            }
    
    def _determine_priority(self):
        """Determine priority level based on predictions"""
        if not self.predictions:
            return 'low'
        
        # High priority if high churn risk OR high potential value
        high_churn_risk = (
            self.predictions.churn_classification == 1 and 
            self.predictions.churn_months and 
            self.predictions.churn_months < 3
        )
        
        high_value = (
            self.predictions.potential_category == 'High' or
            (self.predictions.clv_prediction and self.predictions.clv_prediction > 5000)
        )
        
        if self.predictions.risk_level == 'High' or high_churn_risk or high_value:
            return 'high'
        
        # Medium priority if medium risk or medium potential
        medium_churn_risk = (
            self.predictions.churn_classification == 1 and 
            self.predictions.churn_months and 
            self.predictions.churn_months < 6
        )
        
        medium_value = (
            self.predictions.potential_category == 'Medium' or
            (self.predictions.clv_prediction and self.predictions.clv_prediction > 2000)
        )
        
        if self.predictions.risk_level == 'Medium' or medium_churn_risk or medium_value:
            return 'medium'
        
        return 'low'
    
    def _extract_key_metrics(self):
        """Extract key metrics relevant to the objective"""
        if not self.predictions:
            return {}
        
        obj_ctx = self._get_objective_context()
        metrics = {}
        
        # Always include these core metrics
        core_metrics = {
            'Current Tier': self.client.tier,
            'Enrollment Duration': f"{self.client.enrollment_duration} days",
            'Total Flights': self.client.total_flights
        }
        
        metrics.update(core_metrics)
        
        # Add objective-specific metrics
        for metric in obj_ctx['key_metrics']:
            if hasattr(self.predictions, metric):
                value = getattr(self.predictions, metric)
                # Format the metric name nicely
                display_name = metric.replace('_', ' ').title()
                metrics[display_name] = str(value) if value is not None else 'N/A'
        
        # Add CLV and risk level to all analyses for context
        if hasattr(self.predictions, 'clv_prediction') and self.predictions.clv_prediction is not None:
            metrics['Predicted CLV'] = f"${self.predictions.clv_prediction:,.2f}"
        
        if hasattr(self.predictions, 'risk_level'):
            metrics['Risk Level'] = str(self.predictions.risk_level)
        
        return metrics
    
    def _generate_summary_message(self, priority):
        """Generate a short summary message based on priority"""
        if priority == 'high':
            return "🚨 High Priority: Requires immediate attention"
        elif priority == 'medium':
            return "⚠️ Medium Priority: Important insights for consideration"
        else:
            return "✅ Low Priority: Good standing with optimization opportunities"

def get_assistant(client_id, objective=None):
    """Factory function to get AI assistant for a client"""
    return AIAssistant(client_id, objective)