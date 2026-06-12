"use client";

import { useEffect, useRef, useState } from "react";
import useSWR from "swr";
import { Check, ChevronDown, FolderOpen } from "lucide-react";
import { apiFetch } from "@/lib/auth";
import { cn } from "@/lib/utils";

interface DomainFolder {
  id: string;
  name: string;
  domain_tag: string | null;
}

interface DomainPickerProps {
  containerId: string | null;
  value: string | null;
  onChange: (id: string | null) => void;
}

export function DomainPicker({ containerId, value, onChange }: DomainPickerProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const swrKey = containerId ? `/api/folders?container_id=${containerId}` : null;
  const { data: folders } = useSWR<DomainFolder[]>(
    swrKey,
    async (url: string) => {
      const res = await apiFetch(url);
      if (!res.ok) return [];
      return res.json();
    },
    { revalidateOnFocus: false },
  );

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  // Reset selection when container changes
  useEffect(() => {
    onChange(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [containerId]);

  const items = folders ?? [];
  if (items.length === 0) return null;

  const selected = items.find((f) => f.id === value);
  const label = selected?.name ?? "All domains";

  return (
    <div className="relative min-w-0" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((p) => !p)}
        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-[#f4f4f4] border border-[#e5e5e5] text-[13px] text-[#0a0a0a] hover:bg-[#ebebeb] transition-colors max-w-[200px] sm:max-w-[240px]"
        title="Scope the chat to a specific domain"
      >
        <FolderOpen className="w-3.5 h-3.5 text-[#a3a3a3] shrink-0" />
        <span className="truncate">{label}</span>
        <ChevronDown className="w-3 h-3 text-[#a3a3a3] shrink-0 ml-0.5" />
      </button>

      {open && (
        <div className="absolute z-30 bottom-full mb-1.5 left-0 w-56 rounded-xl border border-[#e5e5e5] bg-white shadow-[0_4px_20px_rgba(0,0,0,0.1)] overflow-hidden">
          <button
            type="button"
            onClick={() => { onChange(null); setOpen(false); }}
            className={cn(
              "w-full flex items-center justify-between px-3 py-2 text-[13px] hover:bg-[#f4f4f4] transition-colors",
              value === null ? "text-[#0a0a0a] font-medium" : "text-[#737373]"
            )}
          >
            <span>All domains</span>
            {value === null && <Check className="w-3.5 h-3.5 shrink-0" />}
          </button>

          {items.map((f) => (
            <button
              key={f.id}
              type="button"
              onClick={() => { onChange(f.id); setOpen(false); }}
              className={cn(
                "w-full flex items-center justify-between px-3 py-2 text-[13px] hover:bg-[#f4f4f4] transition-colors",
                value === f.id ? "text-[#0a0a0a] font-medium" : "text-[#737373]"
              )}
            >
              <span className="truncate">{f.name}</span>
              {value === f.id && <Check className="w-3.5 h-3.5 shrink-0" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
