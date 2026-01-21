class AirlineAIChat {
    constructor() {
        this.sessionId = null;
        this.clientId = null;
        this.typingTimeout = null;
        
        this.initElements();
        this.bindEvents();
        this.initChat();
    }
    
    initElements() {
        this.chatWidget = document.getElementById('chat-widget');
        this.chatIcon = document.getElementById('chat-toggle-btn');
        this.chatCloseBtn = document.getElementById('chat-close-btn');
        this.clientSelect = document.getElementById('client-select');
        this.chatInput = document.getElementById('chat-input');
        this.sendMessageBtn = document.getElementById('send-message-btn');
        this.chatMessages = document.getElementById('chat-messages');
        this.typingIndicator = document.getElementById('typing-indicator');
    }
    
    bindEvents() {
        this.chatIcon.addEventListener('click', () => this.toggleChat());
        this.chatCloseBtn.addEventListener('click', () => this.toggleChat());
        this.clientSelect.addEventListener('change', () => this.handleClientChange());
        this.sendMessageBtn.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Close chat if clicking outside
        document.addEventListener('click', (e) => {
            if (this.chatWidget.style.display === 'block' && 
                !this.chatWidget.contains(e.target) && 
                e.target !== this.chatIcon) {
                this.toggleChat(false);
            }
        });
    }
    
    initChat() {
        // Check if we have a stored session
        const storedSession = localStorage.getItem('airline_ai_session');
        if (storedSession) {
            try {
                const session = JSON.parse(storedSession);
                this.sessionId = session.id;
                this.clientId = session.client_id;
                if (this.clientSelect) {
                    this.clientSelect.value = this.clientId || '';
                }
            } catch (e) {
                console.error('Error parsing stored session:', e);
            }
        }
        
        // Show notification badge if there are unread messages
        const unreadCount = localStorage.getItem('airline_ai_unread') || 0;
        if (parseInt(unreadCount) > 0) {
            this.showNotificationBadge(unreadCount);
        }
    }
    
    toggleChat(show = null) {
        const isVisible = this.chatWidget.style.display === 'block';
        const shouldShow = show === null ? !isVisible : show;
        
        if (shouldShow) {
            this.chatWidget.style.display = 'block';
            this.chatWidget.style.transform = 'translateY(0)';
            this.chatInput.focus();
            
            // Clear notification badge
            this.hideNotificationBadge();
            localStorage.setItem('airline_ai_unread', '0');
        } else {
            this.chatWidget.style.transform = 'translateY(20px)';
            setTimeout(() => {
                this.chatWidget.style.display = 'none';
            }, 300);
        }
        
        return shouldShow;
    }
    
    handleClientChange() {
        this.clientId = this.clientSelect.value || null;
        
        if (this.clientId) {
            // Save to session storage
            localStorage.setItem('airline_ai_session', JSON.stringify({
                id: this.sessionId,
                client_id: this.clientId,
                timestamp: new Date().toISOString()
            }));
            
            // Show welcome message for the selected client
            if (this.chatMessages) {
                const clientName = this.clientSelect.options[this.clientSelect.selectedIndex].text;
                this.addMessage(`Great! I'm ready to analyze ${clientName}. What would you like to know about them? You can ask about churn risk, loyalty points, travel behavior, or any other insights.`, 'ai');
            }
        }
    }
    
    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        this.chatInput.value = '';
        
        // Show typing indicator
        this.showTypingIndicator();
        
        // Clear any previous typing timeout
        if (this.typingTimeout) {
            clearTimeout(this.typingTimeout);
        }
        
        try {
            const response = await fetch('/airline/chat/message/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCookie('csrftoken')
                },
                body: JSON.stringify({
                    message: message,
                    client_id: this.clientId,
                    session_id: this.sessionId
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Update session ID if new one was created
                if (data.session_id) {
                    this.sessionId = data.session_id;
                    localStorage.setItem('airline_ai_session', JSON.stringify({
                        id: this.sessionId,
                        client_id: this.clientId,
                        timestamp: new Date().toISOString()
                    }));
                }
                
                // Hide typing indicator and show response
                this.hideTypingIndicator();
                this.addMessage(data.response, 'ai');
            } else {
                this.hideTypingIndicator();
                this.addMessage(`Error: ${data.error || 'Failed to get response'}`, 'ai');
            }
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage(`Connection error: ${error.message}. Please check your internet connection and try again.`, 'ai');
            console.error('Chat error:', error);
        }
    }
    
    addMessage(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message mb-3`;
        
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        
        const avatarContent = document.createElement('div');
        if (sender === 'ai') {
            avatarContent.className = 'rounded-circle bg-primary text-white d-flex align-items-center justify-content-center';
            avatarContent.style.width = '32px';
            avatarContent.style.height = '32px';
            avatarContent.style.fontSize = '0.9rem';
            avatarContent.textContent = 'AI';
        } else {
            avatarContent.className = 'rounded-circle bg-info text-white d-flex align-items-center justify-content-center';
            avatarContent.style.width = '32px';
            avatarContent.style.height = '32px';
            avatarContent.style.fontSize = '0.9rem';
            avatarContent.innerHTML = '<i class="tim-icons icon-single-02"></i>';
        }
        
        avatar.appendChild(avatarContent);
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content ml-3';
        
        const messageText = document.createElement('p');
        messageText.className = 'mb-1';
        messageText.style.fontSize = '0.9rem';
        messageText.innerHTML = this.formatMessage(content);
        
        const timestamp = document.createElement('small');
        timestamp.className = 'text-muted';
        timestamp.style.fontSize = '0.75rem';
        timestamp.textContent = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        messageContent.appendChild(messageText);
        messageContent.appendChild(timestamp);
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        
        this.chatMessages.appendChild(messageDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    formatMessage(text) {
        // Convert markdown-like syntax to HTML
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/❌/g, '<span class="text-danger">❌</span>')
            .replace(/💡/g, '<span class="text-info">💡</span>')
            .replace(/🤖/g, '<span class="text-primary">🤖</span>')
            .replace(/\n/g, '<br>');
    }
    
    showTypingIndicator() {
        this.typingIndicator.style.display = 'block';
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    hideTypingIndicator() {
        this.typingIndicator.style.display = 'none';
    }
    
    showNotificationBadge(count = 1) {
        const badge = document.getElementById('notification-badge');
        if (badge) {
            badge.textContent = count > 9 ? '9+' : count;
            badge.style.display = 'inline-block';
        }
    }
    
    hideNotificationBadge() {
        const badge = document.getElementById('notification-badge');
        if (badge) {
            badge.style.display = 'none';
        }
    }
    
    getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
}

