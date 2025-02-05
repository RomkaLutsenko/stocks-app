"use client";

import { Header } from "@/features/Header";
import { Macro } from "@/features/Macro";
import Image from "next/image";
import { useForecastStore } from "./store/zustand";

export default function Home() {
  const { imageUrl, isLoading } = useForecastStore();

  return (
    <div className="p-8">
      <Header />

      {/* Main Content */}
      <div className="grid grid-cols-4 gap-4">
        {/* Chart Section */}
        <div className="col-span-3 border rounded-md p-4 h-[800px] flex items-center justify-center">
          {isLoading ? (
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 border-4 border-gray-300 border-t-blue-500 rounded-full animate-spin"></div>
              <p className="mt-4 text-gray-600">Загрузка...</p>
            </div>
          ) : imageUrl ? (
            <Image
              src={imageUrl}
              width="1000"
              height="1000"
              alt="Прогноз"
              unoptimized
            />
          ) : (
            <p className="text-gray-500">Нет данных</p>
          )}
        </div>

        {/* Macro Section */}
        <Macro />
      </div>
    </div>
  );
}
