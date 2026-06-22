import { useCallback, useRef, useState } from 'react';

/**
 * useTypingAnalytics — Captures real-time typing dynamics for CERAS.
 *
 * CHANGES from previous version:
 * - firstInputDelay: time (ms) between hook mount / textarea focus and
 *   the first real keystroke OR real paste. Always measurable, never
 *   null/0 in practice.
 * - copyPasteEvents: incremented ONLY by registerPaste() — called from
 *   PromptInput's existing onPaste handler (handlePaste -> onPaste prop
 *   -> here) when the user actually pastes into the textarea. This is
 *   kept separate from simulateFromExample(), which is for clicking a
 *   pre-filled example card (not a real paste action, so it must NOT
 *   count toward copy_paste_events).
 */

export default function useTypingAnalytics() {
  const [analytics, setAnalytics] = useState({
    totalKeystrokes: 0,
    deletions: 0,
    deletionRatio: 0,
    wpm: 0,
    hesitations: 0,
    longestPause: 0,
    currentPause: 0,
    isHesitating: false,
    sessionDuration: 0,
    avgKeystrokeInterval: 0,
    firstInputDelay: 0,
    copyPasteEvents: 0,
    interKeyDelayStd: 0,
    burstCount: 0,
    burstTypingRatio: 0,
  });

  const stateRef = useRef({
    totalKeys: 0,
    delKeys: 0,
    wordCount: 0,
    timestamps: [],
    lastKeystrokeTime: null,
    sessionStart: null,
    hesitationCount: 0,
    longestPause: 0,
    pauseTimerId: null,
    isHesitating: false,
    focusStart: Date.now(),
    firstInputDelay: 0,
    firstInputCaptured: false,
    copyPasteEvents: 0,
  });

  const _captureFirstInputIfNeeded = (now) => {
    const s = stateRef.current;
    if (!s.firstInputCaptured) {
      s.firstInputDelay = now - s.focusStart;
      s.firstInputCaptured = true;
    }
    if (!s.sessionStart) s.sessionStart = now;
  };

  // Call this on every keydown in the textarea
  const onKeyDown = useCallback((e) => {
    const now = Date.now();
    const s = stateRef.current;

    _captureFirstInputIfNeeded(now);

    s.timestamps.push(now);
    if (s.timestamps.length > 30) s.timestamps.shift();

    const isDeletion = e.key === 'Backspace' || e.key === 'Delete';
    s.totalKeys++;
    if (isDeletion) s.delKeys++;

    if (s.lastKeystrokeTime) {
      const gap = now - s.lastKeystrokeTime;
      if (gap > s.longestPause) s.longestPause = gap;
      if (gap > 2000) s.hesitationCount++;
    }
    s.lastKeystrokeTime = now;
    s.isHesitating = false;

    if (s.pauseTimerId) clearTimeout(s.pauseTimerId);
    s.pauseTimerId = setTimeout(() => {
      s.isHesitating = true;
      updateState();
    }, 2000);

    updateState();
  }, []);

  // Call this when the user genuinely pastes into the textarea
  // (wired via PromptInput's handlePaste -> onPaste prop -> here).
  // Does NOT simulate typing speed — just records the paste event
  // and lets normal onKeyDown/onChange handle the resulting text.
  const registerPaste = useCallback(() => {
    const now = Date.now();
    const s = stateRef.current;
    _captureFirstInputIfNeeded(now);
    s.copyPasteEvents += 1;
    updateState();
  }, []);

  const updateState = useCallback(() => {
    const s = stateRef.current;
    const now = Date.now();
    const sessionSec = s.sessionStart ? (now - s.sessionStart) / 1000 : 0;

    let wpm = 0;
    if (s.timestamps.length >= 2) {
      const span = (s.timestamps[s.timestamps.length - 1] - s.timestamps[0]) / 1000 / 60;
      if (span > 0) wpm = Math.round((s.timestamps.length / 5) / span);
    }

    let avgInterval = 0;
    let interKeyDelayStd = 0;
    let burstCount = 0;
    let burstTypingRatio = 0;

    if (s.timestamps.length >= 2) {
      const intervals = [];
      for (let i = 1; i < s.timestamps.length; i++) {
        intervals.push(s.timestamps[i] - s.timestamps[i - 1]);
      }
      avgInterval = Math.round(intervals.reduce((a, b) => a + b, 0) / intervals.length);

      // Standard deviation of inter-keystroke intervals.
      // Used for both latency_std and inter_key_delay_std in
      // ml_training_data — same underlying statistic.
      const variance = intervals.reduce((sum, v) => sum + (v - avgInterval) ** 2, 0) / intervals.length;
      interKeyDelayStd = Math.round(Math.sqrt(variance));

      // Burst detection: a "burst" is a run of consecutive keystrokes
      // where the gap between each is below BURST_THRESHOLD_MS (fast,
      // fluent typing rhythm). burst_count = number of distinct bursts
      // (length >= 2) found in the recent keystroke window.
      // burst_typing_ratio = proportion of keystrokes that fall inside
      // any burst, vs. isolated/slow keystrokes.
      const BURST_THRESHOLD_MS = 150;
      let inBurst = false;
      let burstKeyTally = 0;
      for (let i = 0; i < intervals.length; i++) {
        if (intervals[i] < BURST_THRESHOLD_MS) {
          if (!inBurst) {
            burstCount += 1;
            inBurst = true;
          }
          burstKeyTally += 1;
        } else {
          inBurst = false;
        }
      }
      burstTypingRatio = s.timestamps.length > 0
        ? Math.round((burstKeyTally / s.timestamps.length) * 1000) / 1000
        : 0;
    }

    const currentPause = s.lastKeystrokeTime ? now - s.lastKeystrokeTime : 0;

    setAnalytics({
      totalKeystrokes: s.totalKeys,
      deletions: s.delKeys,
      deletionRatio: s.totalKeys > 0 ? s.delKeys / s.totalKeys : 0,
      wpm: Math.min(wpm, 200),
      hesitations: s.hesitationCount,
      longestPause: s.longestPause,
      currentPause,
      isHesitating: s.isHesitating,
      sessionDuration: Math.round(sessionSec),
      avgKeystrokeInterval: avgInterval,
      firstInputDelay: s.firstInputDelay,
      copyPasteEvents: s.copyPasteEvents,
      interKeyDelayStd,
      burstCount,
      burstTypingRatio,
    });
  }, []);

  // Simulate analytics for clicking a pre-filled EXAMPLE card.
  // This is NOT a real paste action — copyPasteEvents stays untouched.
  const simulateFromExample = useCallback((text) => {
    const words = text.trim().split(/\s+/).filter(Boolean);
    const wordCount = words.length;
    const charCount = text.length;
    const AVG_WPM = 50;
    const AVG_INTERVAL = Math.round(60000 / (AVG_WPM * 5));
    const sessionSec = wordCount > 0 ? Math.round((wordCount / AVG_WPM) * 60) : 0;

    const s = stateRef.current;
    if (s.pauseTimerId) clearTimeout(s.pauseTimerId);

    const now = Date.now();
    _captureFirstInputIfNeeded(now);

    stateRef.current = {
      ...s,
      totalKeys: charCount,
      delKeys: 0,
      wordCount,
      timestamps: [now - AVG_INTERVAL, now],
      lastKeystrokeTime: now,
      sessionStart: now - sessionSec * 1000,
      hesitationCount: 0,
      longestPause: 500,
      pauseTimerId: null,
      isHesitating: false,
    };

    updateState();
  }, []);

  const reset = useCallback(() => {
    const s = stateRef.current;
    if (s.pauseTimerId) clearTimeout(s.pauseTimerId);
    const now = Date.now();
    stateRef.current = {
      totalKeys: 0,
      delKeys: 0,
      wordCount: 0,
      timestamps: [],
      lastKeystrokeTime: null,
      sessionStart: null,
      hesitationCount: 0,
      longestPause: 0,
      pauseTimerId: null,
      isHesitating: false,
      focusStart: now,
      firstInputDelay: 0,
      firstInputCaptured: false,
      copyPasteEvents: 0,
    };
    setAnalytics({
      totalKeystrokes: 0,
      deletions: 0,
      deletionRatio: 0,
      wpm: 0,
      hesitations: 0,
      longestPause: 0,
      currentPause: 0,
      isHesitating: false,
      sessionDuration: 0,
      avgKeystrokeInterval: 0,
      firstInputDelay: 0,
      copyPasteEvents: 0,
      interKeyDelayStd: 0,
      burstCount: 0,
      burstTypingRatio: 0,
    });
  }, []);

  return {
    analytics,
    onKeyDown,
    registerPaste,
    simulateFromExample,
    reset,
  };
}