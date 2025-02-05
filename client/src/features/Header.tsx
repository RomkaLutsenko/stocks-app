"use client";

import { MacroIndicator, useForecastStore } from "@/app/store/zustand";
import SearchDropdown from "@/features/SearchDropdown";
import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";
import { useMutation } from "@tanstack/react-query";
import { DatePickerWithRange } from "./DatePickerWithRange";

export const Header: React.FC = () => {
  const {
    dateRange,
    stock,
    selectedMacro,
    daysRange,
    setImageUrl,
    setDaysRange,
  } = useForecastStore();

  const handleForecast = () => {
    localStorage.removeItem("forecastData");

    // Сохраняем данные в localStorage
    const forecastData = {
      dateRange,
      stock,
      selectedMacro,
      daysRange,
    };
    localStorage.setItem("forecastData", JSON.stringify(forecastData));

    // Отправляем запрос с этими данными
    mutation.mutate(forecastData);
  };

  const mutation = useMutation({
    mutationFn: async (data: {
      dateRange: { start: string; end: string };
      stock: { symbol: string; name: string };
      selectedMacro: MacroIndicator[];
      daysRange: string;
    }) => {
      const response = await fetch("http://185.41.163.126:8000/predict", {
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
      console.log("blob");
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
      <div className="mr-4">
        Период обучения: с 2016-01-04 до первой даты, которую вы указали в
        календаре
      </div>
      {<DatePickerWithRange className="mr-4" />}
      <div className="mr-4">
        Сколько дополнительных дней слева и справа отображать:
      </div>
      <Input
        placeholder="Сколько дополнительных дней справа и слева отображать"
        className="mr-4 w-11"
        value={daysRange}
        onChange={(e) => setDaysRange(e.target.value)}
      />
      <SearchDropdown />
      <Button onClick={handleForecast}>Сделать прогноз</Button>
    </div>
  );
};
