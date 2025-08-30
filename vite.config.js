import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// Final config for GitHub Pages deployment
export default defineConfig({
  base: '/app/', // deploy to /app/ subdirectory
  plugins: [react()],
  build: {
    outDir: 'app', // build to app directory instead of dist
  },
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
