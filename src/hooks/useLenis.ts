import { useEffect, useRef } from 'react';
import Lenis from 'lenis';

const useLenis = () => {
  const lenisRef = useRef<Lenis | null>(null);
  const rafIdRef = useRef<number | null>(null);

  useEffect(() => {
    // Prevent multiple initializations
    if (lenisRef.current) {
      return;
    }

    // Check if already initialized globally
    if (typeof window !== 'undefined' && (window as any).__LENIS_INSTANCE) {
      lenisRef.current = (window as any).__LENIS_INSTANCE;
      return;
    }

    // Initialize Lenis
    lenisRef.current = new Lenis({
      duration: 1.2,
      easing: (t: number) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      touchMultiplier: 2,
      smoothWheel: true,
    });

    // Store globally to prevent duplicates
    (window as any).__LENIS_INSTANCE = lenisRef.current;

    // Animation frame for smooth scrolling
    const raf = (time: number) => {
      if (lenisRef.current) {
        lenisRef.current.raf(time);
      }
      rafIdRef.current = requestAnimationFrame(raf);
    };

    rafIdRef.current = requestAnimationFrame(raf);

    // Cleanup function
    return () => {
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
      if (lenisRef.current) {
        lenisRef.current.destroy();
        lenisRef.current = null;
      }
      delete (window as any).__LENIS_INSTANCE;
    };
  }, []);

  return lenisRef.current;
};

export { useLenis };