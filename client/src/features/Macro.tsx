"use client";

import { useForecastStore } from "@/app/store/zustand";
import { Checkbox } from "@/shared/ui/checkbox";
import { macro } from "@/shared/utils/Arrays";

export const Macro: React.FC = () => {
  const { selectedMacro, setSelectedMacro } = useForecastStore();

  // Обработчик изменения чекбоксов
  const handleCheckboxChange = (factor: { symbol: string; name: string }) => {
    const exists = selectedMacro.some((item) => item.symbol === factor.symbol);
    let updatedMacro;
    if (exists) {
      // Если фактор уже выбран, удаляем его
      updatedMacro = selectedMacro.filter(
        (item) => item.symbol !== factor.symbol,
      );
    } else {
      // Если не выбран — добавляем его (как объект)
      updatedMacro = [...selectedMacro, factor];
    }
    setSelectedMacro(updatedMacro);
  };

  const handleSelectAllChange = () => {
    if (selectedMacro.length === macro.length) {
      // Если все выбраны, снимаем выбор
      setSelectedMacro([]);
    } else {
      // Иначе выбираем все факторы (копируем массив macro)
      setSelectedMacro(macro);
    }
  };

  return (
    <div className="border rounded-md p-4">
      <h2 className="text-lg font-semibold mb-2 flex justify-between items-center">
        <span>Макрофакторы</span>
        {/* Чекбокс "Выбрать все" */}
        <div className="ml-4">
          <Checkbox
            className="mr-2"
            id="select-all"
            onCheckedChange={handleSelectAllChange}
            checked={selectedMacro.length === macro.length}
          />
          <label htmlFor="select-all" className="text-sm">
            Выбрать все
          </label>
        </div>
      </h2>
      {macro.map((factor) => (
        <div key={factor.symbol} className="flex items-center space-x-2 mb-2">
          <Checkbox
            id={factor.symbol}
            onCheckedChange={() => handleCheckboxChange(factor)}
            checked={selectedMacro.some(
              (item) => item.symbol === factor.symbol,
            )}
          />
          <label htmlFor={factor.symbol} className="text-sm">
            {factor.name}
          </label>
        </div>
      ))}
    </div>
  );
};
