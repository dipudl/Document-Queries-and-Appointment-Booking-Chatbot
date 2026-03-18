export default function FileUpload({ onUpload, disabled }) {
  const handleChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      onUpload(file)
      e.target.value = ''
    }
  }

  return (
    <div className="file-upload">
      <label>
        Upload Doc
        <input
          type="file"
          accept=".pdf,.txt,.md"
          onChange={handleChange}
          disabled={disabled}
        />
      </label>
    </div>
  )
}
