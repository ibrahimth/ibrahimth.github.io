import { useEffect, useRef } from 'react';

interface TouchDragOptions {
  onMove: (clientX: number, clientY: number) => void;
  onStart?: () => void;
  onEnd?: () => void;
}

export function useTouchDrag(
  elementRef: React.RefObject<SVGElement>,
  isActive: boolean,
  options: TouchDragOptions
) {
  const lastPosition = useRef<{ x: number; y: number } | null>(null);
  const isDragging = useRef(false);

  useEffect(() => {
    if (!isActive || !elementRef.current) return;

    const element = elementRef.current;

    function handleMouseMove(e: MouseEvent) {
      if (!isDragging.current) return;
      e.preventDefault();
      options.onMove(e.clientX, e.clientY);
    }

    function handleTouchMove(e: TouchEvent) {
      if (!isDragging.current || e.touches.length === 0) return;
      e.preventDefault();
      const touch = e.touches[0];
      options.onMove(touch.clientX, touch.clientY);
    }

    function handleMouseUp() {
      if (!isDragging.current) return;
      isDragging.current = false;
      lastPosition.current = null;
      options.onEnd?.();
    }

    function handleTouchEnd() {
      if (!isDragging.current) return;
      isDragging.current = false;
      lastPosition.current = null;
      options.onEnd?.();
    }

    function handleMouseDown(e: MouseEvent) {
      e.preventDefault();
      isDragging.current = true;
      lastPosition.current = { x: e.clientX, y: e.clientY };
      options.onStart?.();
    }

    function handleTouchStart(e: TouchEvent) {
      if (e.touches.length !== 1) return;
      e.preventDefault();
      e.stopPropagation();
      const touch = e.touches[0];
      isDragging.current = true;
      lastPosition.current = { x: touch.clientX, y: touch.clientY };
      options.onStart?.();
    }

    // Add event listeners
    element.addEventListener('mousedown', handleMouseDown);
    element.addEventListener('touchstart', handleTouchStart, { passive: false });

    // Global move and up listeners
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('touchmove', handleTouchMove, { passive: false });
    document.addEventListener('touchend', handleTouchEnd);

    return () => {
      element.removeEventListener('mousedown', handleMouseDown);
      element.removeEventListener('touchstart', handleTouchStart);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('touchmove', handleTouchMove);
      document.removeEventListener('touchend', handleTouchEnd);
    };
  }, [isActive, options]);

  return isDragging.current;
}