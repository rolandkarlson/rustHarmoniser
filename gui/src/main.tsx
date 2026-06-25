import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

declare global {
  interface Number {
    mod(n: number): number;
  }
  interface Array<T> {
    /** Element at `i` wrapped by length (handles negatives); like Rust's get_wrapped. */
    get(i: number): T;
  }
  // eslint-disable-next-line no-var
  var range: (x: number) => number[];
  // eslint-disable-next-line no-var
  var mod: (x: number, n: number) => number;
}

(Number.prototype as any).mod = function (n: number) {
  "use strict";
  return ((this as number % n) + n) % n;
};

(Array.prototype as any).get = function (i: number) {
  "use strict";
  const len = this.length;
  if (len === 0) return undefined;
  return this[((Math.trunc(i) % len) + len) % len];
};

(window as any).range = (x: number) => Array.from({ length: x }, (_, i) => i);

(window as any).mod = (x: number, n: number) => ((x % n) + n) % n;

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
