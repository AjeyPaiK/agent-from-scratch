"""Typography and layout for the chat UI.

Notes
-----
``CHAT_THEME_CSS`` is injected into the Streamlit page via ``st.markdown`` in
``chat_app.py``. It styles fonts, chat messages, avatars, and hides default
Streamlit chrome.
"""

CHAT_THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:ital,wght@0,400;0,500;0,600;1,400&display=swap');

html, body, [class*="css"] {
  font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stApp {
  background: linear-gradient(180deg, #faf9f7 0%, #f5f4f1 100%);
}
.stApp[data-theme="dark"] {
  background: linear-gradient(180deg, #1c1917 0%, #141210 100%);
}

.block-container {
  max-width: 46rem;
  padding-top: 0.75rem;
  padding-bottom: 5rem;
}

.chat-header {
  text-align: center;
  padding: 1.25rem 0 1.75rem;
  margin-bottom: 0.5rem;
}
.chat-header h1 {
  font-family: 'Source Sans 3', sans-serif;
  font-size: 1.35rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  margin: 0;
  color: #1c1917;
}
.stApp[data-theme="dark"] .chat-header h1 { color: #fafaf9; }
.chat-header p {
  margin: 0.35rem 0 0;
  font-size: 0.82rem;
  color: #78716c;
  font-weight: 400;
}

.welcome-card {
  border-radius: 14px;
  padding: 1.25rem 1.35rem;
  margin: 0.5rem 0 1.5rem;
  line-height: 1.55;
  font-size: 0.92rem;
}
.stApp:not([data-theme="dark"]) .welcome-card {
  background: #ffffff;
  border: 1px solid #e7e5e4;
  color: #44403c;
  box-shadow: 0 1px 2px rgba(28, 25, 23, 0.04);
}
.stApp[data-theme="dark"] .welcome-card {
  background: #292524;
  border: 1px solid #44403c;
  color: #d6d3d1;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
  font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  font-size: 0.98rem !important;
  line-height: 1.58 !important;
  letter-spacing: -0.005em !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) li {
  color: #292524 !important;
}
.stApp[data-theme="dark"] [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) p,
.stApp[data-theme="dark"] [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) li {
  color: #e7e5e4 !important;
}

[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 0.5rem 0 !important;
  align-items: flex-start !important;
  gap: 0.65rem !important;
}

[data-testid="stChatMessage"] > div {
  padding-top: 0.1rem !important;
}

[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
  width: 1.75rem !important;
  height: 1.75rem !important;
  min-width: 1.75rem !important;
  min-height: 1.75rem !important;
  margin-top: 0.12rem !important;
  flex-shrink: 0 !important;
  align-self: flex-start !important;
}

[data-testid="stChatMessageAvatarUser"] {
  background: #e7e5e4 !important;
}
.stApp[data-theme="dark"] [data-testid="stChatMessageAvatarUser"] {
  background: #44403c !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
  background: #d6d3d1 !important;
}
.stApp[data-theme="dark"] [data-testid="stChatMessageAvatarAssistant"] {
  background: #57534e !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
  color: #1c1917 !important;
}
.stApp[data-theme="dark"] [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
  color: #fafaf9 !important;
}

[data-testid="stChatInput"] textarea {
  font-family: 'Source Sans 3', sans-serif !important;
  font-size: 0.95rem !important;
  border-radius: 12px !important;
}
.stApp:not([data-theme="dark"]) [data-testid="stChatInput"] {
  background: #ffffff;
  border: 1px solid #e7e5e4;
  border-radius: 14px;
  box-shadow: 0 4px 24px rgba(28, 25, 23, 0.06);
}

.stream-cursor {
  display: inline-block;
  width: 2px;
  height: 1.05em;
  background: #a8a29e;
  margin-left: 1px;
  vertical-align: text-bottom;
  animation: cursor-blink 1.05s step-end infinite;
}
@keyframes cursor-blink {
  50% { opacity: 0; }
}

[data-testid="stDeployButton"], [data-testid="stToolbar"],
.stApp > header[data-testid="stHeader"] { display: none !important; }
#MainMenu, footer { visibility: hidden; height: 0; }
"""
