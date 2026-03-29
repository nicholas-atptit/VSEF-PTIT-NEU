import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, Bot, Sparkles } from 'lucide-react';

export default function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Xin chào! Tôi là AI Agent của Hệ thống Algo Trading. Bạn muốn phân tích mã chứng khoán nào hôm nay?' }
  ]);
  const [inputStr, setInputStr] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const endRef = useRef(null);

  const scrollToBottom = () => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!inputStr.trim() || isTyping) return;

    const userMsg = inputStr.trim();
    setInputStr('');
    
    // Add user message to UI
    const updatedMessages = [...messages, { role: 'user', content: userMsg }];
    setMessages(updatedMessages);
    setIsTyping(true);

    try {
      const resp = await fetch('http://127.0.0.1:8000/api/v2/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMsg,
          history: messages.filter(m => m.role !== 'system')
        })
      });

      if (!resp.ok) {
        throw new Error(`Connection error: ${resp.status}`);
      }

      const data = await resp.json();
      setMessages([...updatedMessages, { role: 'assistant', content: data.response }]);
    } catch (err) {
      setMessages([
        ...updatedMessages, 
        { role: 'assistant', content: `❌ **Lỗi Kết Nối**: Không thể kết nối tới local API Server. Vui lòng kiểm tra xem bạn đã chạy \`uvicorn src.api.main:app\` chưa.\n\nChi tiết: ${err.message}` }
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="chat-container">
      {/* Header */}
      <div className="chat-header">
        <Sparkles size={24} color="#8b5cf6" />
        <h1>Agentic Analysis Terminal</h1>
        <div style={{flex: 1}}></div>
        <div style={{display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', color: '#a0a0b0'}}>
          <div className="status-indicator"></div>
          [LIVE]
        </div>
      </div>

      {/* History */}
      <div className="chat-history">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role === 'assistant' ? 'ai' : 'user'}`}>
            <ReactMarkdown>{msg.content}</ReactMarkdown>
          </div>
        ))}
        {isTyping && (
          <div className="typing-indicator">
            <div className="typing-dot"></div>
            <div className="typing-dot"></div>
            <div className="typing-dot"></div>
          </div>
        )}
        <div ref={endRef}></div>
      </div>

      {/* Input */}
      <form onSubmit={handleSend} className="chat-input-area">
        <input 
          type="text" 
          className="chat-input"
          placeholder="Nhập yêu cầu phân tích (VD: Nhận định MSN, VGI...)"
          value={inputStr}
          onChange={(e) => setInputStr(e.target.value)}
          disabled={isTyping}
          autoFocus
        />
        <button type="submit" className="send-btn" disabled={!inputStr.trim() || isTyping}>
          <Send size={20} />
        </button>
      </form>
    </div>
  );
}
