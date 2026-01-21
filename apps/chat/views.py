from django.shortcuts import render
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
import json
from django.utils import timezone
from ..airline_project.models import Client
from ..airline_project.ai_assistant import get_assistant
from .models import ChatSession, ChatMessage

@login_required
def chat_widget(request):
    """Render the chat widget"""
    clients = Client.objects.all().order_by('last_name')[:20]  # Limit for performance
    return render(request, 'chat/chat_widget.html', {'clients': clients})

@csrf_exempt
@login_required
def chat_message(request):
    """Handle chat messages"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        message = data.get('message', '').strip()
        client_id = data.get('client_id', None)
        session_id = data.get('session_id', None)
        
        if not message:
            return JsonResponse({'error': 'Message cannot be empty'}, status=400)
        
        # Get or create session
        if session_id:
            try:
                session = ChatSession.objects.get(id=session_id, user=request.user)
            except ChatSession.DoesNotExist:
                session = ChatSession.objects.create(user=request.user, client_id=client_id)
        else:
            session = ChatSession.objects.create(user=request.user, client_id=client_id)
        
        # Save user message
        ChatMessage.objects.create(
            session=session,
            message=message,
            is_user=True
        )
        
        # Process the message and get AI response
        response = process_ai_message(request.user, message, client_id)
        
        # Save AI response
        ChatMessage.objects.create(
            session=session,
            message=message,
            response=response['response'],
            is_user=False
        )
        
        return JsonResponse({
            'session_id': session.id,
            'response': response['response'],
            'client_id': client_id,
            'timestamp': timezone.now().isoformat()
        })
        
    except Exception as e:
        return JsonResponse({
            'error': f'An error occurred: {str(e)}'
        }, status=500)

def process_ai_message(user, message, client_id=None):
    """Process message and generate AI response using Groq"""
    try:
        # Default response if no client selected
        if not client_id:
            return {
                'response': "I see you haven't selected a client yet. Please select a client from the dropdown menu to analyze their data and get personalized insights."
            }
        
        # Get the client object
        try:
            client = Client.objects.get(id=client_id)
        except Client.DoesNotExist:
            return {
                'response': f"I couldn't find a client with ID {client_id}. Please select a valid client from the dropdown."
            }
        
        # Determine objective based on message content
        objective = determine_objective(message)
        
        # Get AI assistant and generate recommendation
        assistant = get_assistant(client.id, objective)
        recommendation = assistant.get_recommendation()
        
        # Format response for chat
        if recommendation['status'] == 'error':
            return {
                'response': f"❌ **Error**: {recommendation['message']}\n\nPlease try again with a different request or check if all ML models have been run for this client."
            }
        
        # Create formatted response
        response_text = f"🤖 **AI Analysis for {client.get_full_name()}**\n\n"
        
        # Add priority-based message
        response_text += f"**Priority**: {recommendation['message']}\n\n"
        
        # Add key metrics
        if recommendation.get('metrics'):
            response_text += "**Key Metrics**:\n"
            for key, value in recommendation['metrics'].items():
                response_text += f"- {key}: {value}\n"
            response_text += "\n"
        
        # Add first paragraph of analysis
        response_text += "**Insight**:\n"
        first_paragraph = recommendation['full_analysis'].split('\n')[0]
        response_text += first_paragraph + "\n\n"
        
        response_text += "💡 *You can ask for more details or specific recommendations by continuing the conversation!*"
        
        return {
            'response': response_text,
            'objective': objective
        }
        
    except Exception as e:
        return {
            'response': f"❌ **Error generating response**: {str(e)}\n\nThis might be due to:\n- Missing API key configuration\n- Network connectivity issues\n- Unavailable ML predictions for this client\n\nPlease try again or contact your system administrator."
        }

def determine_objective(message):
    """Determine the analysis objective based on message content"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ['churn', 'leave', 'cancel', 'stop', 'risk']):
        return 'churn'
    elif any(word in message_lower for word in ['value', 'clv', 'lifetime', 'profit']):
        return 'clv'
    elif any(word in message_lower for word in ['loyalty', 'points', 'tier', 'rewards']):
        return 'loyalty'
    elif any(word in message_lower for word in ['travel', 'flight', 'journey', 'route', 'trip']):
        return 'travel'
    elif any(word in message_lower for word in ['engage', 'enrollment', 'join', 'program', 'participate']):
        return 'enrollment'
    else:
        return None  # General overview