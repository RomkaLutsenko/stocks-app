import { create } from "zustand";

interface ForecastState {
  dateRange: { start: string; end: string };
  forecastDays: string;
  stock: string;
  selectedMacro: string[];

  imageUrl: string | null;
  setImageUrl: (url: string) => void;

  setDateRange: (start: string, end: string) => void;
  setForecastDays: (days: string) => void;
  setStock: (chosenStock: string) => void;
  setSelectedMacro: (checkboxes: string[]) => void;
}

export const useForecastStore = create<ForecastState>((set) => ({
  dateRange: { start: "2016-01-04", end: "2021-07-12" },
  forecastDays: "7",
  stock: "SBER.ME",
  selectedMacro: [],

  imageUrl: null,
  setImageUrl: (url) => set({ imageUrl: url }),

  setDateRange: (start, end) => set({ dateRange: { start, end } }),
  setForecastDays: (days) => set({ forecastDays: days }),
  setStock: (chosenStock) => set({ stock: chosenStock }),
  setSelectedMacro: (checkboxes) => set({ selectedMacro: checkboxes }),
}));
