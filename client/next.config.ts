/** @type {import('next').NextConfig} */
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*", // Перенаправляем все запросы к /api на бэкенд
        destination: "http://localhost:8000/api/:path*", // Бэкенд на
      },
    ];
  },
};

export default nextConfig;
