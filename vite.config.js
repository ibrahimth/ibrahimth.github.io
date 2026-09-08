import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  base: '/',
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html'),
        app: path.resolve(__dirname, 'app/index.html'),
        coe526: path.resolve(__dirname, 'COE526_Data_Privacy/index.html'),
        coe526Foundations: path.resolve(
          __dirname,
          'COE526_Data_Privacy/interactive-01-privacy-foundations-and-terms.html',
        ),
        coe526Laws: path.resolve(
          __dirname,
          'COE526_Data_Privacy/interactive-02-privacy-laws.html',
        ),
        coe526Assessment: path.resolve(
          __dirname,
          'COE526_Data_Privacy/COE526-Interactive-02-Privacy-Assessment-Lab.html',
        ),
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 5173,
  },
})
