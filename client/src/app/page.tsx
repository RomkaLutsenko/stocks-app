"use client";

import { Header } from "@/features/Header";
import { Macro } from "@/features/Macro";
import { useForecastStore } from "./store/zustand";

export default function Home() {
  const { imageUrl } = useForecastStore();

  return (
    <div className="p-8">
      <Header />

      {/* Main Content */}
      <div className="grid grid-cols-4 gap-4">
        {/* Chart Section */}
        <div className="col-span-3 border rounded-md p-4 h-[800px] flex items-center justify-center">
          {imageUrl && <img src={imageUrl} alt="Прогноз" />}
        </div>

        {/* Macro Section */}
        <Macro />
      </div>
    </div>
  );
}
