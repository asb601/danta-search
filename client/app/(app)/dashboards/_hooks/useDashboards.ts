"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "@/lib/auth";
import { DashboardSummary, DashboardFolder } from "@/components/analytics-catalog/types";

export function useDashboards() {
  const [dashboards, setDashboards] = useState<DashboardSummary[]>([]);
  const [folders, setFolders] = useState<DashboardFolder[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [activeFolder, setActiveFolder] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ limit: "200" });
      if (search.trim()) params.set("search", search.trim());
      if (activeFolder) params.set("folder_id", activeFolder);
      const [dRes, fRes] = await Promise.all([
        apiFetch(`/api/dashboards?${params}`),
        apiFetch(`/api/dashboards/folders`),
      ]);
      if (dRes.ok) setDashboards((await dRes.json()).dashboards || []);
      if (fRes.ok) setFolders((await fRes.json()).folders || []);
    } finally {
      setLoading(false);
    }
  }, [search, activeFolder]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const createDashboard = useCallback(async (title: string) => {
    const res = await apiFetch(`/api/dashboards`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, folder_id: activeFolder }),
    });
    return res.ok ? res.json() : null;
  }, [activeFolder]);

  const createFolder = useCallback(async (name: string) => {
    const res = await apiFetch(`/api/dashboards/folders`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (res.ok) await refresh();
  }, [refresh]);

  const updateDashboard = useCallback(async (id: string, patch: Record<string, unknown>) => {
    await apiFetch(`/api/dashboards/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
    await refresh();
  }, [refresh]);

  const deleteDashboard = useCallback(async (id: string) => {
    await apiFetch(`/api/dashboards/${id}`, { method: "DELETE" });
    await refresh();
  }, [refresh]);

  const duplicateDashboard = useCallback(async (id: string) => {
    await apiFetch(`/api/dashboards/${id}/duplicate`, { method: "POST" });
    await refresh();
  }, [refresh]);

  return {
    dashboards, folders, loading, search, setSearch, activeFolder, setActiveFolder,
    refresh, createDashboard, createFolder, updateDashboard, deleteDashboard, duplicateDashboard,
  };
}
