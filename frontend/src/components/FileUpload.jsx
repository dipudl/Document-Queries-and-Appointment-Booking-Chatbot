export default function FileUpload({ onUpload, disabled }) {
  const ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.md']

  const handleChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase()
      if (!ALLOWED_EXTENSIONS.includes(ext)) {
        alert('Only PDF, TXT and MD files are supported.')
        e.target.value = ''
        return
      }
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
