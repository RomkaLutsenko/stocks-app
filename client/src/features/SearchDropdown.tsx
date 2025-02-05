import { useForecastStore } from "@/app/store/zustand";
import { tickers } from "@/shared/utils/Arrays";
import React, { useEffect, useRef, useState } from "react";

const SearchDropdown = () => {
  const { stock, setStock } = useForecastStore();

  const [filteredTickers, setFilteredTickers] = useState(tickers); // Состояние для фильтрованных тикеров
  const [isDropdownVisible, setDropdownVisible] = useState(false); // Видимость выпадающего списка
  const dropdownRef = useRef<HTMLDivElement>(null); // Ссылка на выпадающий список

  // Обработчик ввода
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value.toLowerCase();
    setStock(stock);

    // Фильтрация тикеров
    const filtered = tickers.filter(
      (ticker) =>
        ticker.symbol.toLowerCase().includes(value) ||
        (ticker.name && ticker.name.toLowerCase().includes(value)),
    );
    setFilteredTickers(filtered);

    setDropdownVisible(true); // Показать выпадающий список
  };

  // Обработчик выбора элемента
  const handleSelect = (ticker: { symbol: string; name: string }) => {
    setStock(ticker); // Устанавливаем выбранный тикер в поле ввода
    setDropdownVisible(false); // Скрываем выпадающий список
  };

  // Обработчик кликов вне компонента
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setDropdownVisible(false); // Скрываем выпадающий список
      }
    };

    document.addEventListener("mousedown", handleClickOutside);

    return () => {
      document.removeEventListener("mousedown", handleClickOutside); // Чистим слушатель
    };
  }, []);

  return (
    <div className="relative w-80 mr-4" ref={dropdownRef}>
      {/* Поле ввода */}
      <input
        type="text"
        placeholder="Поиск по доступным акциям"
        className="w-full p-2 border rounded"
        value={tickers.find((t) => t.symbol === stock.symbol)?.name || ""}
        onChange={handleInputChange}
        onFocus={() => setDropdownVisible(true)} // Показать список при фокусе
      />

      {/* Выпадающий список */}
      {isDropdownVisible && (
        <ul className="absolute z-10 w-full bg-white border border-gray-300 rounded shadow-lg mt-1 max-h-60 overflow-y-auto">
          {filteredTickers.length > 0 ? (
            filteredTickers.map((ticker) => (
              <li
                key={ticker.symbol}
                className="p-2 hover:bg-gray-100 cursor-pointer"
                onClick={() => handleSelect(ticker)}
              >
                <strong>{ticker.symbol}</strong>:{" "}
                {ticker.name || "Нет названия"}
              </li>
            ))
          ) : (
            <li className="p-2 text-gray-500">Ничего не найдено</li>
          )}
        </ul>
      )}
    </div>
  );
};

export default SearchDropdown;
