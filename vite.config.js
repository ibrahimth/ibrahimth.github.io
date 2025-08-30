import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// Final config for GitHub Pages deployment
export default defineConfig({
  base: '/', // ensures assets load from this subpath
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 5173,
    open: true,
  },
})
