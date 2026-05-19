import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

declare global {
  interface Number {
    mod(n: number): number;
  }
  // eslint-disable-next-line no-var
  var range: (x: number) => number[];
}

(Number.prototype as any).mod = function (n: number) {
  "use strict";
  return ((this as number % n) + n) % n;
};

(window as any).range = (x: number) => Array.from({ length: x }, (_, i) => i);

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
