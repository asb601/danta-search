"use client";

import { useState, useCallback } from "react";
import useSWR, { mutate as globalMutate } from "swr";
import FileManagerView, {
  type FileItem,
  type FileStatus,
  type FileType,
  type UploadProgressItem,
} from "@/components/file-manager/FileManagerView";
import { apiFetch } from "@/lib/auth";
import { useAuth } from "@/components/auth-provider";
import { uploadFileQueue, UploadController, type UploadProgress } from "@/lib/upload";

/* ── types for container picker ──────────────────────────────────────────── */

interface Container {
  id: string;
  name: string;
}

const containersFetcher = async (): Promise<Container[]> => {
  const res = await apiFetch("/api/containers");
  if (!res.ok) return [];
  return res.json();
};

function mapFolder(f: { id: string; name: string; created_at: string; domain_tag?: string | null }): FileItem {
  return {
    id: f.id,
    name: f.name,
    type: "folder",
    size: 0,
    status: "not_ingested",
    lastModified: new Date(f.created_at),
    domainTag: f.domain_tag ?? null,
  };
}

function mapFile(f: {
  id: string;
  name: string;
  content_type: string;
  size: number;
  created_at: string;
  container_id?: string | null;
  ingest_status?: string;
  uploaded_by_name?: string | null;
}): FileItem {
  const ext = f.name.split(".").pop()?.toLowerCase() ?? "";
  let type: FileType = "txt";
  if (ext === "csv") type = "csv";
  else if (ext === "xlsx" || ext === "xls") type = "xlsx";
  else if (ext === "pdf") type = "pdf";

  return {
    id: f.id,
    name: f.name,
    type,
    size: f.size,
    status: (
      f.ingest_status === "ingested" ? "indexed" :
      f.ingest_status === "running"  ? "pending" :
      f.ingest_status === "pending"  ? "pending" :
      f.ingest_status === "failed"   ? "failed"  :
      "not_ingested"
    ) as FileStatus,
    lastModified: new Date(f.created_at),
    synced: !!f.container_id,
    uploadedBy: f.uploaded_by_name || undefined,
  };
}

const contentsFetcher = async (key: string): Promise<FileItem[]> => {
  // key format: contents:<folderId>:<containerId|all>
  const parts = key.split(":");
  const folderId = parts[1] ?? "root";
  const containerScope = parts[2] ?? "all";
  const qs = containerScope !== "all" ? `?container_id=${encodeURIComponent(containerScope)}` : "";
  const url = `/api/folders/${folderId}/contents${qs}`;
  const res = await apiFetch(url);
  if (!res.ok) return [];
  const data = await res.json();
  return [
    ...(data.folders ?? []).map(mapFolder),
    ...(data.files ?? [])
      .filter((f: { name: string }) => !f.name.toLowerCase().endsWith(".parquet"))
      .map(mapFile),
  ];
};

interface FolderBreadcrumb {
  id: string;
  name: string;
}

export default function FoldersPage() {
  const { user } = useAuth();
  const isAdmin = user?.is_admin ?? false;
  const canWrite = isAdmin || user?.role === "developer";

  const [folderStack, setFolderStack] = useState<FolderBreadcrumb[]>([]);
  const [selectedContainerId, setSelectedContainerId] = useState<string>("");

  const currentFolderId =
    folderStack.length > 0 ? folderStack[folderStack.length - 1].id : null;
  const folderName =
    folderStack.length > 0 ? folderStack[folderStack.length - 1].name : undefined;
  const swrKey = `contents:${currentFolderId ?? "root"}:${selectedContainerId || "all"}`;

  const { data: items, isLoading, mutate } = useSWR(swrKey, contentsFetcher, {
    revalidateOnFocus: false,
  });

  // Fetch containers for the picker (available to all authenticated users)
  const { data: containers } = useSWR(
    "containers-list",
    containersFetcher,
    { revalidateOnFocus: false }
  );

  const handleFolderOpen = useCallback(
    (id: string) => {
      const folder = items?.find((i) => i.id === id && i.type === "folder");
      if (folder) {
        setFolderStack((prev) => [...prev, { id: folder.id, name: folder.name }]);
      }
    },
    [items]
  );

  const handleFolderHover = useCallback((id: string) => {
    const prefetchKey = `contents:${id}:${selectedContainerId || "all"}`;
    globalMutate(prefetchKey, contentsFetcher(prefetchKey), false);
  }, [selectedContainerId]);

  const handleBack = useCallback(() => {
    setFolderStack((prev) => prev.slice(0, -1));
  }, []);

  const [uploadProgress, setUploadProgress] = useState<UploadProgressItem[]>([]);
  const [uploadCtrl, setUploadCtrl] = useState<UploadController | null>(null);

  const handleUpload = useCallback(
    async (files: File[]) => {
      if (!selectedContainerId) return;

      const ctrl = new UploadController();
      setUploadCtrl(ctrl);

      // Init progress state — all queued
      setUploadProgress(
        files.map((f) => ({ fileName: f.name, percent: 0, speedMBps: 0, remainingSecs: 0, phase: "queued" as const }))
      );

      const onProgress = (p: UploadProgress) => {
        setUploadProgress((prev) =>
          prev.map((item, i) =>
            i === p.fileIndex
              ? { fileName: p.fileName, percent: p.percent, speedMBps: p.speedMBps, remainingSecs: p.remainingSecs, phase: p.phase }
              : item
          )
        );
      };

      // Queue-based upload: 2 files at a time
      await uploadFileQueue(files, currentFolderId, selectedContainerId, onProgress, ctrl, 2);

      await mutate();
      ctrl.cleanup();
      setUploadCtrl(null);

      // Clear progress after a short delay
      setTimeout(() => setUploadProgress([]), 3000);
    },
    [currentFolderId, mutate, selectedContainerId]
  );

  const handleCancelUpload = useCallback((fileIndex: number) => {
    uploadCtrl?.cancel(fileIndex);
  }, [uploadCtrl]);

  const handleCancelAll = useCallback(() => {
    uploadCtrl?.cancelAll();
  }, [uploadCtrl]);

  const [deletingIds, setDeletingIds] = useState<Set<string>>(new Set());

  const handleDelete = useCallback(
    async (id: string) => {
      const item = items?.find((i) => i.id === id);
      if (!item) return;
      // Guard against re-firing the same delete. A folder delete cascades through
      // every file inside it and can take a while; without this, a second click
      // (or a re-render) launches a concurrent delete and the two deadlock in the DB.
      if (deletingIds.has(id)) return;
      setDeletingIds((prev) => new Set(prev).add(id));
      const endpoint =
        item.type === "folder" ? "/api/folders/" + id : "/api/files/" + id;
      try {
        await apiFetch(endpoint, { method: "DELETE" });
        await mutate();
      } catch (err) {
        // Surface the failure instead of silently swallowing it (previously the
        // folder just reappeared after mutate with no indication anything broke).
        await mutate();
        alert(
          `Could not delete "${item.name ?? id}". It may still be processing a large delete — wait a moment and try again, one folder at a time.`
        );
      } finally {
        setDeletingIds((prev) => {
          const next = new Set(prev);
          next.delete(id);
          return next;
        });
      }
    },
    [items, mutate, deletingIds]
  );

  const handleIngest = useCallback(async (id: string) => {
    try {
      const res = await apiFetch("/api/chat/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_ids: [id] }),
      });
      if (!res.ok) return;
      // Optimistic: mark as pending in local data
      await mutate();
      // Poll until status changes from pending
      const poll = setInterval(async () => {
        const statusRes = await apiFetch(`/api/chat/ingest-status/${id}`);
        if (statusRes.ok) {
          const data = await statusRes.json();
          if (data.ingest_status !== "pending") {
            clearInterval(poll);
            await mutate();
          }
        }
      }, 3000);
      // Safety: stop polling after 2 minutes
      setTimeout(() => { clearInterval(poll); mutate(); }, 120_000);
    } catch {
      // silently ignore
    }
  }, [mutate]);

  const handleIngestForce = useCallback(async (id: string) => {
    try {
      const res = await apiFetch("/api/chat/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_ids: [id], force_preprocess: true }),
      });
      if (!res.ok) return;
      await mutate();
      const poll = setInterval(async () => {
        const statusRes = await apiFetch(`/api/chat/ingest-status/${id}`);
        if (statusRes.ok) {
          const data = await statusRes.json();
          if (data.ingest_status !== "pending") {
            clearInterval(poll);
            await mutate();
          }
        }
      }, 3000);
      setTimeout(() => { clearInterval(poll); mutate(); }, 120_000);
    } catch {
      // silently ignore
    }
  }, [mutate]);

  const handleOpenFile = useCallback(async (id: string) => {
    try {
      const res = await apiFetch(`/api/files/${id}/signed-url`);
      if (!res.ok) return;
      const { signed_url } = await res.json();
      window.open(signed_url, "_blank");
    } catch {
      // signed-url fetch failed — silently ignore
    }
  }, []);

  const [reingestLoading, setReingestLoading] = useState(false);
  const [reingestQuickLoading, setReingestQuickLoading] = useState(false);
  const [reingestOrgWideLoading, setReingestOrgWideLoading] = useState(false);

  // One parameterized reingest-all runner. `forcePreprocess` toggles the
  // clean/convert step; `allContainers` fans out org-wide. `setLoading` is the
  // per-scope loading setter so each menu option shows its own spinner.
  const runReingestAll = useCallback(
    async (opts: {
      forcePreprocess: boolean;
      allContainers?: boolean;
      setLoading: (v: boolean) => void;
    }) => {
      const { forcePreprocess, allContainers, setLoading } = opts;
      setLoading(true);
      try {
        const params = new URLSearchParams({
          force_preprocess: String(forcePreprocess),
        });
        if (allContainers) params.set("all_containers", "true");
        const res = await apiFetch(`/api/admin/reingest-all?${params.toString()}`, {
          method: "POST",
        });
        if (!res.ok) {
          setLoading(false);
          return;
        }
        // Poll until all files in the current view are done re-ingesting.
        const poll = setInterval(async () => {
          await mutate();
          const freshItems = await contentsFetcher(swrKey);
          const stillPending = freshItems.some(
            (i) => i.type !== "folder" && (i.status === "pending" || i.status === "not_ingested")
          );
          if (!stillPending) {
            clearInterval(poll);
            setLoading(false);
            await mutate();
          }
        }, 5000);
        // Safety: stop after 5 minutes
        setTimeout(() => { clearInterval(poll); setLoading(false); mutate(); }, 300_000);
      } catch {
        setLoading(false);
      }
    },
    [mutate, swrKey]
  );

  const handleReingestAll = useCallback(
    () => runReingestAll({ forcePreprocess: true, setLoading: setReingestLoading }),
    [runReingestAll]
  );

  const handleReingestAllQuick = useCallback(
    () => runReingestAll({ forcePreprocess: false, setLoading: setReingestQuickLoading }),
    [runReingestAll]
  );

  const handleReingestAllOrgWide = useCallback(
    () =>
      runReingestAll({
        forcePreprocess: false,
        allContainers: true,
        setLoading: setReingestOrgWideLoading,
      }),
    [runReingestAll]
  );

  const handleCreateFolder = useCallback(
    async (name: string) => {
      const payload = {
        name,
        parent_id: currentFolderId ?? undefined,
      };
      try {
        await apiFetch("/api/folders", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        await mutate();
      } catch {
        await mutate();
      }
    },
    [currentFolderId, swrKey, mutate]
  );

  const handleRename = useCallback(
    async (id: string, newName: string) => {
      const item = items?.find((i) => i.id === id);
      if (!item) return;
      const endpoint =
        item.type === "folder" ? "/api/folders/" + id : "/api/files/" + id;
      try {
        await apiFetch(endpoint, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: newName }),
        });
        await mutate();
      } catch {
        await mutate();
      }
    },
    [items, swrKey, mutate]
  );

  const handleMove = useCallback(
    async (fileId: string, targetFolderId: string | null) => {
      try {
        await apiFetch(`/api/files/${fileId}/move`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ folder_id: targetFolderId }),
        });
        await mutate();
      } catch {
        await mutate();
      }
    },
    [mutate]
  );

  return (
    <FileManagerView
      files={items ?? []}
      folderName={folderName}
      loading={isLoading}
      readOnly={!canWrite}
      uploadProgress={uploadProgress}
      containers={containers ?? []}
      selectedContainerId={selectedContainerId}
      onContainerChange={setSelectedContainerId}
      onUpload={handleUpload}
      onIngest={handleIngest}
      onIngestForce={canWrite ? handleIngestForce : undefined}
      onDelete={handleDelete}
      onRename={handleRename}
      onCreateFolder={handleCreateFolder}
      onFolderOpen={handleFolderOpen}
      onOpenFile={handleOpenFile}
      onFolderHover={handleFolderHover}
      onBack={folderStack.length > 0 ? handleBack : undefined}
      onCancelUpload={handleCancelUpload}
      onCancelAllUploads={handleCancelAll}
      onReingestAll={canWrite ? handleReingestAll : undefined}
      reingestLoading={reingestLoading}
      onReingestAllQuick={canWrite ? handleReingestAllQuick : undefined}
      reingestQuickLoading={reingestQuickLoading}
      onReingestAllOrgWide={canWrite ? handleReingestAllOrgWide : undefined}
      reingestOrgWideLoading={reingestOrgWideLoading}
      onMove={canWrite ? handleMove : undefined}
    />
  );
}