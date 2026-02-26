import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

export interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
}

export interface ChatResponse {
    answer: string;
    node: string;
    thoughts: string[];
}

export interface SystemStatus {
    status: string;
    api_key_set: boolean;
    faiss_index_exists: boolean;
    resume_index_exists: boolean;
}

export const checkStatus = async (): Promise<SystemStatus> => {
    try {
        const res = await axios.get(`${API_URL}/status`);
        return res.data;
    } catch (err) {
        console.error("Failed to fetch status", err);
        return {
            status: "offline",
            api_key_set: false,
            faiss_index_exists: false,
            resume_index_exists: false
        };
    }
};

export const sendMessage = async (question: string, history: ChatMessage[]): Promise<ChatResponse> => {
    const res = await axios.post(`${API_URL}/chat`, {
        question,
        chat_history: history
    });
    return res.data;
};
