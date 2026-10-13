import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy dev : /api -> FastAPI (port 8001)
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8001",
    },
  },
});
