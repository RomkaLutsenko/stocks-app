import { create } from "zustand";

interface ForecastState {
  dateRange: { start: string; end: string };
  stock: { symbol: string; name: string };
  selectedMacro: string[];
  daysRange: string;

  imageUrl: string | null;
  setImageUrl: (url: string) => void;

  setDateRange: (start: string, end: string) => void;
  setStock: (chosenStock: { symbol: string; name: string }) => void;
  setSelectedMacro: (checkboxes: string[]) => void;
  setDaysRange: (daysRange: string) => void;
}

export const useForecastStore = create<ForecastState>((set) => ({
  dateRange: { start: "2016-01-04", end: "2021-07-13" },
  stock: { symbol: "BBG004730N88", name: "Сбербанк" },
  selectedMacro: [],
  daysRange: "30",

  imageUrl: null,
  setImageUrl: (url) => set({ imageUrl: url }),

  setDateRange: (start, end) => set({ dateRange: { start, end } }),
  setStock: (chosenStock) => set({ stock: chosenStock }),
  setSelectedMacro: (checkboxes) => set({ selectedMacro: checkboxes }),
  setDaysRange: (daysRange) => set({ daysRange }),
}));
