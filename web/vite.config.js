import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

// The built app is committed inside the Python package, so `pip install` is all a user
// needs. File names are fixed (no content hashes) to keep those commits small.
export default defineConfig({
  plugins: [svelte()],
  base: "./",
  build: {
    outDir: "../src/semantic_librarian/web",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        entryFileNames: "assets/app.js",
        chunkFileNames: "assets/[name].js",
        assetFileNames: "assets/app[extname]",
      },
    },
  },
  server: {
    // `npm run dev` reads data from a bundle served by `sl serve <library>` on port 8000
    proxy: { "/data": "http://127.0.0.1:8000" },
  },
});
