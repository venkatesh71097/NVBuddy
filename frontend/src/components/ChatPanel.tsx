import { useState, useEffect, useRef } from 'react';
import { Send, Cpu } from 'lucide-react';
import { marked } from 'marked';
import { sendMessage } from '../services/api';
import type { ChatMessage as ApiChatMessage } from '../services/api';

interface ChatBubble extends ApiChatMessage {
    id: string;
    isError?: boolean;
}

const ChatPanel: React.FC = () => {
    const [messages, setMessages] = useState<ChatBubble[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [activeThoughts, setActiveThoughts] = useState<string[]>([]);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, activeThoughts]);

    // Initial greeting
    useEffect(() => {
        setMessages([
            {
                id: 'init',
                role: 'assistant',
                content: 'Hi! I am your Dual-Brain Agent. You can ask me about NVIDIA technologies, general system design, sizing calculations, or deployment optimizations. What would you like to practice today?'
            }
        ]);
    }, []);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMsg: ChatBubble = { id: Date.now().toString(), role: 'user', content: input.trim() };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsLoading(true);
        setActiveThoughts(['Connecting to Dual-Brain...']);

        try {
            // We pass the history to the backend
            const historyForApi: ApiChatMessage[] = messages.filter(m => !m.isError).map(m => ({
                role: m.role,
                content: m.content
            }));

            const response = await sendMessage(userMsg.content, historyForApi);

            setMessages(prev => [
                ...prev,
                {
                    id: (Date.now() + 1).toString(),
                    role: 'assistant',
                    content: response.answer
                }
            ]);
            setActiveThoughts([]);
        } catch (error) {
            setMessages(prev => [
                ...prev,
                {
                    id: (Date.now() + 1).toString(),
                    role: 'assistant',
                    content: 'Sorry, I encountered an error communicating with the agent backend. Ensure the FastAPI server is running.',
                    isError: true
                }
            ]);
            setActiveThoughts([]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="chat-container">
            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '24px', paddingBottom: '20px' }}>
                {messages.map((msg) => (
                    <div key={msg.id} className={`message ${msg.role}`}>
                        {msg.role === 'assistant' && (
                            <div style={{ marginRight: '16px', color: 'var(--nv-green)' }}>
                                <Cpu size={24} />
                            </div>
                        )}
                        <div
                            className={msg.isError ? 'error-text' : ''}
                            style={{ lineHeight: '1.6', fontSize: '15px' }}
                            dangerouslySetInnerHTML={{ __html: marked.parse(msg.content) as string }}
                        />
                    </div>
                ))}

                {isLoading && (
                    <div className="message assistant glass-panel" style={{ alignItems: 'center' }}>
                        <div style={{ marginRight: '16px', color: 'var(--nv-green)' }}>
                            <Cpu size={24} className="pulsing" />
                        </div>
                        <div>
                            <div className="thinking-log">
                                {activeThoughts[activeThoughts.length - 1] || 'Agent is thinking...'}
                            </div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <div className="input-area">
                <div className="input-wrapper">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Ask a question (e.g., 'Calculate GPU sizing for Llama 3 70B')"
                        disabled={isLoading}
                    />
                    <button
                        className="send-btn"
                        onClick={handleSend}
                        disabled={!input.trim() || isLoading}
                    >
                        <Send size={20} />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ChatPanel;
