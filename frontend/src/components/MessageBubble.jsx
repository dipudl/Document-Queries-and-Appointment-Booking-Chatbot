export default function MessageBubble({ role, content }) {
  return <div className={`message ${role}`}>{content}</div>
}
