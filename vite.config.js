import { defineConfig } from 'vite';

export default defineConfig({
  base: '/robodog-3d/',
  publicDir: 'public',
  build: {
    outDir: 'dist',
    assetsInlineLimit: 0,
  },
});
