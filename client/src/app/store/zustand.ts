import { addDays, format } from "date-fns";
import { create } from "zustand";

export interface MacroIndicator {
  symbol: string;
  name: string;
}

interface ForecastState {
  dateRange: { start: string; end: string };
  stock: { symbol: string; name: string };
  selectedMacro: MacroIndicator[];
  daysRange: string;

  imageUrl: string | null;
  setImageUrl: (url: string) => void;

  setDateRange: (start: string, end: string) => void;
  setStock: (chosenStock: { symbol: string; name: string }) => void;
  setSelectedMacro: (checkboxes: MacroIndicator[]) => void;
  setDaysRange: (daysRange: string) => void;
}

const defaultStart = format(addDays(new Date(), -7), "yyyy-MM-dd");
const defaultEnd = format(new Date(), "yyyy-MM-dd");

export const useForecastStore = create<ForecastState>((set) => ({
  dateRange: { start: defaultStart, end: defaultEnd },
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
