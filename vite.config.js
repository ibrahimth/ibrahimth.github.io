import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// Build the React app for GitHub Pages
export default defineConfig({
  base: '/',            // Base path for GitHub Pages user site (ibrahimth.github.io)
  plugins: [react()],
  build: {
    outDir: 'dist',    // put the built site in /dist
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html'),
        app: path.resolve(__dirname, 'app/index.html'),
      },
    },
  },
  // root: 'src',           // REMOVED: use project root
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
