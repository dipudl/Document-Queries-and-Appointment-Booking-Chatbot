import { useState, useRef } from 'react'
import ChatWindow from './components/ChatWindow'
import FileUpload from './components/FileUpload'

function generateSessionId() {
  return 'session_' + Math.random().toString(36).substring(2, 10)
}

const SESSION_KEY = 'metacloud_session_id'

function getSessionId() {
  let id = sessionStorage.getItem(SESSION_KEY)
  if (!id) {
    id = generateSessionId()
    sessionStorage.setItem(SESSION_KEY, id)
  }
  return id
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const sessionId = useRef(getSessionId())

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || loading) return

    const userMsg = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId.current, message: text }),
      })
      const data = await res.json()
      const botMsg = { role: 'bot', content: data.response }
      setMessages(prev => [...prev, botMsg])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'bot', content: 'Error: Could not reach the server.' }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleUpload = async (file) => {
    const formData = new FormData()
    formData.append('session_id', sessionId.current)
    formData.append('file', file)

    setMessages(prev => [...prev, { role: 'user', content: `Uploading: ${file.name}` }])
    setLoading(true)

    try {
      const res = await fetch('/upload', { method: 'POST', body: formData })
      const data = await res.json()
      setMessages(prev => [
        ...prev,
        { role: 'bot', content: `Document "${data.filename}" uploaded and processed (${data.chunks} chunks). You can now ask questions about it!` },
      ])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'bot', content: 'Error: Failed to upload document.' }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Document Queries & Appointment Booking Chatbot</h1>
        <FileUpload onUpload={handleUpload} disabled={loading} />
      </header>
      <ChatWindow messages={messages} loading={loading} />
      <div className="input-bar">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  )
}
