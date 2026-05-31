import { useEffect, useRef } from 'react'

/**
 * Adds .visible to any element with class .reveal/.reveal-left/.reveal-right/.reveal-scale
 * when it enters the viewport. Supports stagger delays via .stagger-N classes.
 */
export function useScrollReveal(rootMargin = '-60px') {
  const containerRef = useRef<HTMLElement | null>(null)

  useEffect(() => {
    const targets = document.querySelectorAll<HTMLElement>(
      '.reveal, .reveal-left, .reveal-right, .reveal-scale'
    )

    const obs = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            ;(entry.target as HTMLElement).classList.add('visible')
            obs.unobserve(entry.target)
          }
        })
      },
      { rootMargin, threshold: 0.1 }
    )

    targets.forEach(el => obs.observe(el))
    return () => obs.disconnect()
  }, [rootMargin])

  return containerRef
}
