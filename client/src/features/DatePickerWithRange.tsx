"use client";

import { format } from "date-fns";
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
  const { dateRange, setDateRange } = useForecastStore();

  const handleSelect = (selectedDate: DateRange | undefined) => {
    if (selectedDate?.from && selectedDate?.to) {
      setDateRange(
        format(selectedDate.from, "yyyy-MM-dd"),
        format(selectedDate.to, "yyyy-MM-dd"),
      );
    }
  };

  return (
    <div className={cn("grid gap-2", className)}>
      <Popover>
        <PopoverTrigger asChild>
          <Button
            id="date"
            variant={"outline"}
            className={cn(
              "w-[300px] justify-start text-left font-normal",
              !dateRange && "text-muted-foreground",
            )}
          >
            <CalendarIcon />
            {dateRange?.start ? (
              dateRange.end ? (
                <>
                  {format(dateRange.start, "LLL dd, y")} -{" "}
                  {format(dateRange.end, "LLL dd, y")}
                </>
              ) : (
                format(dateRange.start, "LLL dd, y")
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
            defaultMonth={
              dateRange?.start ? new Date(dateRange.start) : undefined
            }
            selected={
              dateRange?.start && dateRange?.end
                ? {
                    from: new Date(dateRange.start),
                    to: new Date(dateRange.end),
                  }
                : undefined
            }
            onSelect={handleSelect}
            numberOfMonths={2}
          />
        </PopoverContent>
      </Popover>
    </div>
  );
}
