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

console.log("var a = (i, n, s, v) => {\n" +
    "var x = (i - s)% 8;\n" +
    "if(n.indexOf(x) > -1){\n" +
    " return v.get(i);\n" +
    "}\n" +
    " return 1;\n" +
    "}\n" +
    "\n" +
    "range(5).map((n) => config.voice_rhythm_contour[n] = range(128).map(i=>a(i, [0,3], n, [[3,0.25,1,2,1], [0.5, 0.75,1,2,3], [1,2],[1,2],[1,2],[1,2]].get(n))));\n" +
    "config.mode_contour = range(128).map(i=>(Math.floor(i/16)*4)%7);\n" +
    "config.schillinger_sequence = range(128).map(i=>mod(i*-2,7));");

(Number.prototype as any).mod = function (n: number) {
  "use strict";
  return ((this as number % n) + n) % n;
};

(Number.prototype as any).step = function (n: number) {
  "use strict";
  return Math.floor((this as number / n)) ;
};
(Number.prototype as any).a = function (n: (x: unknown) => unknown) {
  "use strict";
  return n(this) ;
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
