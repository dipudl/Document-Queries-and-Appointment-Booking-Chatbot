import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'

export default function ChatWindow({ messages, loading }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  return (
    <div className="chat-window">
      {messages.length === 0 && (
        <div style={{ textAlign: 'center', color: '#9ca3af', marginTop: '40px' }}>
          <p style={{ fontSize: '1.1rem', marginBottom: '8px' }}>Welcome to Document Queries & Appointment Booking Chatbot!</p>
          <p style={{ fontSize: '0.9rem' }}>Upload a document and ask questions, or book an appointment.</p>
        </div>
      )}
      {messages.map((msg, i) => (
        <MessageBubble key={i} role={msg.role} content={msg.content} />
      ))}
      {loading && <div className="typing-indicator">Thinking...</div>}
      <div ref={bottomRef} />
    </div>
  )
}
