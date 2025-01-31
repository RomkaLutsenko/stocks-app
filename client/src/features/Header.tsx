"use client";

import { useForecastStore } from "@/app/store/zustand";
import SearchDropdown from "@/features/SearchDropdown";
import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";
import { useMutation } from "@tanstack/react-query";

export const Header: React.FC = () => {
  const {
    dateRange,
    forecastDays,
    stock,
    selectedMacro,
    //setDateRange,
    setForecastDays,
    setImageUrl,
  } = useForecastStore();

  const handleForecast = () => {
    localStorage.removeItem("forecastData");

    // Сохраняем данные в localStorage
    const forecastData = {
      dateRange,
      forecastDays,
      stock,
      selectedMacro,
    };
    localStorage.setItem("forecastData", JSON.stringify(forecastData));

    // Отправляем запрос с этими данными
    mutation.mutate(forecastData);
  };

  const mutation = useMutation({
    mutationFn: async (data: {
      dateRange: { start: string; end: string };
      forecastDays: string;
      stock: string;
      selectedMacro: string[];
    }) => {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error("Ошибка при запросе прогноза");
      }

      // Получаем изображение в виде blob
      const blob = await response.blob();
      return URL.createObjectURL(blob); // Создаем URL для отображения изображения
    },
    onSuccess: (imageURL) => {
      setImageUrl(imageURL); // Сохраняем URL в zustand
    },
    onError: (error) => {
      console.error("Ошибка:", error);
      alert("Произошла ошибка при выполнении запроса.");
    },
  });

  return (
    <div className="flex items-center justify-between mb-4">
      {/* <DatePickerWithRange
        className="mr-4"
        onDateChange={(start, end) => {
          setDateRange(start, end);
        }}
      /> */}
      <div>
        Пока что период(на котором обучается модель) один: с 2016-01-04 по
        2021-07-12
      </div>
      <Input
        placeholder="На сколько дней вперед прогноз"
        className="mr-4 w-60"
        value={forecastDays}
        onChange={(e) => setForecastDays(e.target.value)}
      />
      <SearchDropdown />
      <Button onClick={handleForecast}>Сделать прогноз</Button>
    </div>
  );
};
