"use client";

import { Header } from "@/features/Header";
import { Macro } from "@/features/Macro";
import Image from "next/image";
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
          {imageUrl && (
            <Image
              src={imageUrl}
              width="1000"
              height="1000"
              alt="Прогноз"
              unoptimized
            />
          )}
        </div>

        {/* Macro Section */}
        <Macro />
      </div>
    </div>
  );
}
