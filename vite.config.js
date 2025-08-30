import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// Build the React app into /app with relative asset URLs
export default defineConfig({
  base: './',            // IMPORTANT: make assets relative for subfolder hosting
  plugins: [react()],
  build: {
    outDir: '../app',    // put the built site in /app (relative to src)
    emptyOutDir: true,
  },
  root: 'src',           // use src as the project root
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
