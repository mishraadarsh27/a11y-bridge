import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      // Proxy REST endpoints to the FastAPI backend
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/sign': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/translate': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      // Proxy WebSocket to FastAPI
      '/ws': {
        target: 'http://127.0.0.1:8000',
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
