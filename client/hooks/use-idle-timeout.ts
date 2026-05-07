"use client";

import { useEffect, useRef, useCallback } from "react";

// Events that count as "user is active".
// touchstart covers mobile; scroll covers passive reading.
const ACTIVITY_EVENTS = [
  "mousemove",
  "keydown",
  "click",
  "scroll",
  "touchstart",
] as const;

/**
 * Idle timeout hook. Call this once at the authenticated layout level.
 *
 * How it works
 * ─────────────
 * 1. On mount, we start a setTimeout for `timeoutMs`.
 * 2. On every user-activity event we:
 *      a. clearTimeout the old timer   ← cancel the pending logout
 *      b. setTimeout a fresh one       ← restart the 30-minute clock
 * 3. If the timer fires without being reset, we call `onTimeout`.
 * 4. On unmount we cancel the timer and remove all listeners (cleanup).
 *
 * Why useRef for the timer id?
 * ─────────────────────────────
 * useState would cause a re-render every time we reset the timer —
 * that's potentially hundreds of renders per scroll. useRef is a
 * mutable box React does NOT watch, so writes to it are free.
 *
 * Why useCallback for resetTimer?
 * ─────────────────────────────────
 * The event listener is attached inside a useEffect. If resetTimer were
 * a plain inline function, it would be recreated on every render, making
 * removeEventListener fail (it needs the exact same function reference
 * to remove the right listener). useCallback with [] deps gives a stable
 * reference across renders.
 */
export function useIdleTimeout({
  timeoutMs,
  onTimeout,
}: {
  timeoutMs: number;
  onTimeout: () => void;
}) {
  // useRef, not useState — mutations here do NOT re-render the component.
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // useCallback with [] so this function reference is stable across renders.
  // If we didn't do this, removeEventListener below would silently fail
  // because browser compares by reference identity — different function
  // objects even with identical bodies are NOT the same listener.
  const resetTimer = useCallback(() => {
    // Step 1: cancel whatever is pending right now.
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
    }
    // Step 2: start a fresh countdown.
    timerRef.current = setTimeout(onTimeout, timeoutMs);
  }, [onTimeout, timeoutMs]);
  // ^ onTimeout and timeoutMs in deps: if the parent passes a new logout
  // function or changes the timeout duration, resetTimer rebuilds correctly.

  useEffect(() => {
    // Start the initial timer when the hook mounts (user just logged in).
    resetTimer();

    // Attach the same resetTimer function to every activity event.
    // { passive: true } is a browser performance hint: we promise we won't
    // call preventDefault() inside the listener, so the browser doesn't
    // have to pause scrolling to check. Always set this on scroll/touch.
    const opts: AddEventListenerOptions = { passive: true };
    for (const event of ACTIVITY_EVENTS) {
      window.addEventListener(event, resetTimer, opts);
    }

    // Cleanup: runs on unmount (user navigates away, logs out, etc.)
    // This MUST mirror add exactly — same event names, same function ref,
    // same options object shape. Without this, listeners pile up across
    // navigations and you get memory leaks.
    return () => {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current);
      }
      for (const event of ACTIVITY_EVENTS) {
        window.removeEventListener(event, resetTimer, opts);
      }
    };
  }, [resetTimer]);
  // ^ resetTimer in deps: when resetTimer rebuilds (parent changes onTimeout),
  // the effect re-runs: removes old listeners, attaches new ones. Correct.
}
