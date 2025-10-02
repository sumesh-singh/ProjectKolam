import { useMotionValue, useTransform } from 'framer-motion';
import { useRef, useEffect } from 'react';

export const useParallax = (speed: number = 0.5) => {
  const ref = useRef<HTMLDivElement>(null);
  const scrollY = useMotionValue(0);
  const elementMetrics = useRef({ top: 0, height: 0, windowHeight: 0 });

  useEffect(() => {
    const updateMetrics = () => {
      if (ref.current) {
        const rect = ref.current.getBoundingClientRect();
        elementMetrics.current = {
          top: rect.top + window.scrollY,
          height: rect.height,
          windowHeight: window.innerHeight
        };
      }
    };

    // Capture metrics on mount
    updateMetrics();

    const handleScroll = () => {
      scrollY.set(window.scrollY);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    window.addEventListener('resize', updateMetrics);

    return () => {
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('resize', updateMetrics);
    };
  }, [scrollY]);

  // Calculate parallax offset based on scroll position
  const y = useTransform(scrollY, (value: number) => {
    const { top: elementTop, height: elementHeight, windowHeight } = elementMetrics.current;

    if (elementTop || elementHeight) {
      // Calculate progress based on element position
      const start = elementTop - windowHeight;
      const end = elementTop + elementHeight;
      const progress = Math.max(0, Math.min(1, (value - start) / (end - start)));

      return -progress * speed * 100;
    }

    return 0;
  });

  return { ref, y };
};