"use client";

import { useEffect, useRef, useState } from "react";
import useSWR from "swr";
import { Boxes, Check, ChevronDown } from "lucide-react";
import { apiFetch } from "@/lib/auth";
import { cn } from "@/lib/utils";

interface Container {
  id: string;
  name: string;
}

const fetcher = async (): Promise<Container[]> => {
  const res = await apiFetch("/api/containers");
  if (!res.ok) return [];
  return res.json();
};

interface ContainerPickerProps {
  value: string | null;
  onChange: (id: string | null) => void;
}

export function ContainerPicker({ value, onChange }: ContainerPickerProps) {
  const { data: containers } = useSWR("containers-list", fetcher, {
    revalidateOnFocus: false,
  });
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const selected = containers?.find((c) => c.id === value);
  const label = selected?.name ?? "All containers";

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((p) => !p)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-surface border border-border text-sm text-foreground hover:bg-surface-raised transition-colors"
        title="Scope the chat to a specific container"
      >
        <Boxes className="w-4 h-4 text-muted-foreground" />
        <span className="max-w-[160px] truncate">{label}</span>
        <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />
      </button>

      {open && (
        <div className="absolute z-30 mt-1 right-0 w-56 rounded-md border border-border bg-surface shadow-lg overflow-hidden">
          <button
            type="button"
            onClick={() => {
              onChange(null);
              setOpen(false);
            }}
            className={cn(
              "w-full flex items-center justify-between px-3 py-2 text-sm hover:bg-surface-raised transition-colors",
              value === null && "text-primary"
            )}
          >
            <span>All containers</span>
            {value === null && <Check className="w-4 h-4" />}
          </button>
          {(containers ?? []).map((c) => (
            <button
              key={c.id}
              type="button"
              onClick={() => {
                onChange(c.id);
                setOpen(false);
              }}
              className={cn(
                "w-full flex items-center justify-between px-3 py-2 text-sm hover:bg-surface-raised transition-colors",
                value === c.id && "text-primary"
              )}
            >
              <span className="truncate">{c.name}</span>
              {value === c.id && <Check className="w-4 h-4 shrink-0" />}
            </button>
          ))}
          {(!containers || containers.length === 0) && (
            <div className="px-3 py-2 text-xs text-muted-foreground">
              No containers available
            </div>
          )}
        </div>
      )}
    </div>
  );
}
