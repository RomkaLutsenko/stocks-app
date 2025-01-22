"use client";

import { useForecastStore } from "@/app/store/zustand";
import { Checkbox } from "@/shared/ui/checkbox";
import { macro } from "@/shared/utils/Arrays";

export const Macro: React.FC = () => {
  const { selectedMacro, setSelectedMacro } = useForecastStore();

  // Обработчик изменения чекбоксов
  const handleCheckboxChange = (factor: string) => {
    const updatedMacro = selectedMacro.includes(factor)
      ? selectedMacro.filter((item) => item !== factor)
      : [...selectedMacro, factor];
    setSelectedMacro(updatedMacro);
  };

  const handleSelectAllChange = () => {
    if (selectedMacro.length === macro.length) {
      // Если все выбраны, снимаем все
      setSelectedMacro([]);
    } else {
      // Если не все выбраны, выбираем все
      setSelectedMacro(macro.map((factor) => factor.symbol));
    }
  };

  return (
    <div className="border rounded-md p-4">
      <h2 className="text-lg font-semibold mb-2 flex justify-between items-center space-x-2">
        <span>Макрофакторы</span>
        {/* Чекбокс "Выбрать все" */}
        <div>
          <Checkbox
            id="select-all"
            onCheckedChange={handleSelectAllChange}
            checked={selectedMacro.length === macro.length}
          />
          <label htmlFor="select-all" className="text-sm ml-2">
            Выбрать все
          </label>
        </div>
      </h2>
      {macro.map((factor) => (
        <div key={factor.symbol} className="flex items-center space-x-2 mb-2">
          <Checkbox
            id={factor.symbol}
            onCheckedChange={() => handleCheckboxChange(factor.symbol)}
            checked={selectedMacro.includes(factor.symbol)}
          />
          <label htmlFor={factor.symbol} className="text-sm">
            {factor.name}
          </label>
        </div>
      ))}
    </div>
  );
};
