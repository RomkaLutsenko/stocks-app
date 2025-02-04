"use client";

import { addDays, format } from "date-fns";
import { CalendarIcon } from "lucide-react";
import * as React from "react";
import { DateRange } from "react-day-picker";

import { useForecastStore } from "@/app/store/zustand";
import { cn } from "@/shared/lib/utils";
import { Button } from "@/shared/ui/button";
import { Calendar } from "@/shared/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/shared/ui/popover";

export function DatePickerWithRange({
  className,
}: React.HTMLAttributes<HTMLDivElement>) {
  const { setDateRange } = useForecastStore();

  const [date, setDate] = React.useState<DateRange | undefined>({
    from: new Date(2025, 0, 21),
    to: addDays(new Date(2025, 0, 21), 7),
  });

  const handleSelect = (selectedDate: DateRange | undefined) => {
    if (selectedDate?.from && selectedDate?.to) {
      setDate(selectedDate);
      setDateRange(
        format(selectedDate.from, "yyyy-MM-dd"),
        format(selectedDate.to, "yyyy-MM-dd"),
      );
    }
  };

  React.useEffect(() => {
    if (date?.from && date?.to) {
      setDateRange(
        format(date.from, "yyyy-MM-dd"),
        format(date.to, "yyyy-MM-dd"),
      );
    }
  }, []);

  return (
    <div className={cn("grid gap-2", className)}>
      <Popover>
        <PopoverTrigger asChild>
          <Button
            id="date"
            variant={"outline"}
            className={cn(
              "w-[300px] justify-start text-left font-normal",
              !date && "text-muted-foreground",
            )}
          >
            <CalendarIcon />
            {date?.from ? (
              date.to ? (
                <>
                  {format(date.from, "LLL dd, y")} -{" "}
                  {format(date.to, "LLL dd, y")}
                </>
              ) : (
                format(date.from, "LLL dd, y")
              )
            ) : (
              <span>Pick a date</span>
            )}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0" align="start">
          <Calendar
            initialFocus
            mode="range"
            defaultMonth={date?.from}
            selected={date}
            onSelect={handleSelect}
            numberOfMonths={2}
          />
        </PopoverContent>
      </Popover>
    </div>
  );
}
