"use client";

import { useEffect, useRef, useState } from "react";
import useSWR from "swr";
import { Boxes, Check, ChevronDown } from "lucide-react";
import { useAuth } from "@/components/auth-provider";
import { apiFetch } from "@/lib/auth";
import { cn } from "@/lib/utils";

interface Container {
  id: string;
  name: string;
}

/** GET /api/access-requests/my-grants -> [{organization_id, organization_name, container_ids}] */
interface AccessGrant {
  organization_id: string;
  organization_name: string;
  container_ids: string[];
}

const fetcher = async (): Promise<Container[]> => {
  const res = await apiFetch("/api/containers");
  if (!res.ok) return [];
  return res.json();
};

const grantsFetcher = async (): Promise<AccessGrant[]> => {
  const res = await apiFetch("/api/access-requests/my-grants");
  if (!res.ok) return [];
  return res.json();
};

interface ContainerPickerProps {
  value: string | null;
  onChange: (id: string | null) => void;
}

export function ContainerPicker({ value, onChange }: ContainerPickerProps) {
  const { user } = useAuth();
  const isPlatformAdmin = !!user?.is_admin;

  const { data: containers } = useSWR("containers-list", fetcher, {
    revalidateOnFocus: false,
  });
  // Only platform admins see granted-org containers alongside their own.
  const { data: grants } = useSWR(
    isPlatformAdmin ? "access-my-grants" : null,
    grantsFetcher,
    { revalidateOnFocus: false },
  );

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

  const own = containers ?? [];
  // Map own container ids → names so granted ids can borrow a friendly label.
  const nameById = new Map(own.map((c) => [c.id, c.name] as const));

  // Build the granted-org groups, skipping any container already in "own"
  // (avoid showing the same container twice).
  const ownIds = new Set(own.map((c) => c.id));
  const grantedGroups = (grants ?? [])
    .map((g) => ({
      org: g.organization_name,
      containers: g.container_ids
        .filter((id) => !ownIds.has(id))
        .map((id) => ({ id, name: nameById.get(id) ?? id })),
    }))
    .filter((g) => g.containers.length > 0);

  // Resolve label for the currently-selected id across own + granted.
  const grantedSelected = grantedGroups
    .flatMap((g) => g.containers)
    .find((c) => c.id === value);
  const selected = own.find((c) => c.id === value);
  const label = selected?.name ?? grantedSelected?.name ?? "All containers";

  const hasAny = own.length > 0 || grantedGroups.length > 0;

  return (
    <div className="relative min-w-0" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((p) => !p)}
        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-[#f4f4f4] border border-[#e5e5e5] text-[13px] text-[#0a0a0a] hover:bg-[#ebebeb] transition-colors max-w-[200px] sm:max-w-[240px]"
        title="Scope the chat to a specific container"
      >
        <Boxes className="w-3.5 h-3.5 text-[#a3a3a3] shrink-0" />
        <span className="truncate">{label}</span>
        <ChevronDown className="w-3 h-3 text-[#a3a3a3] shrink-0 ml-0.5" />
      </button>

      {open && (
        <div className="absolute z-30 mt-1.5 left-0 w-56 rounded-xl border border-[#e5e5e5] bg-white shadow-[0_4px_20px_rgba(0,0,0,0.1)] overflow-hidden max-h-[60vh] overflow-y-auto">
          <button
            type="button"
            onClick={() => { onChange(null); setOpen(false); }}
            className={cn(
              "w-full flex items-center justify-between px-3 py-2 text-[13px] hover:bg-[#f4f4f4] transition-colors",
              value === null ? "text-[#0a0a0a] font-medium" : "text-[#737373]"
            )}
          >
            <span>All containers</span>
            {value === null && <Check className="w-3.5 h-3.5 shrink-0" />}
          </button>

          {own.map((c) => (
            <button
              key={c.id}
              type="button"
              onClick={() => { onChange(c.id); setOpen(false); }}
              className={cn(
                "w-full flex items-center justify-between px-3 py-2 text-[13px] hover:bg-[#f4f4f4] transition-colors",
                value === c.id ? "text-[#0a0a0a] font-medium" : "text-[#737373]"
              )}
            >
              <span className="truncate">{c.name}</span>
              {value === c.id && <Check className="w-3.5 h-3.5 shrink-0" />}
            </button>
          ))}

          {/* Granted-org containers, grouped/labeled by organization. */}
          {grantedGroups.map((g) => (
            <div key={g.org} className="border-t border-[#e5e5e5]">
              <p className="px-3 pt-2 pb-1 text-[10px] font-semibold text-[#a3a3a3] uppercase tracking-widest truncate">
                {g.org}
              </p>
              {g.containers.map((c) => (
                <button
                  key={c.id}
                  type="button"
                  onClick={() => { onChange(c.id); setOpen(false); }}
                  className={cn(
                    "w-full flex items-center justify-between px-3 py-2 text-[13px] hover:bg-[#f4f4f4] transition-colors",
                    value === c.id ? "text-[#0a0a0a] font-medium" : "text-[#737373]"
                  )}
                >
                  <span className="truncate">{c.name}</span>
                  {value === c.id && <Check className="w-3.5 h-3.5 shrink-0" />}
                </button>
              ))}
            </div>
          ))}

          {!hasAny && (
            <div className="px-3 py-2.5 text-[12px] text-[#a3a3a3]">
              No containers available
            </div>
          )}
        </div>
      )}
    </div>
  );
}
