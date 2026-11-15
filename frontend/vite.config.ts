import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The FastAPI demo service runs on 127.0.0.1:8010 (8000 is often taken, e.g. by Splunk);
// /api is proxied so no CORS setup is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8010' },
  },
})
