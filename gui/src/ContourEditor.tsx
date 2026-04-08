import React, { useState, useRef } from 'react';

interface ContourEditorProps {
  label: string;
  data: number[];
  onChange: (newData: number[]) => void;
  yMin: number;
  yMax: number;
  xMax: number;
  resolution: number;
  pl: number;
  snaps?: number[];
  color?: string;
}

export const ContourEditor: React.FC<ContourEditorProps> = ({
  label,
  data,
  onChange,
  yMin,
  yMax,
  xMax,
  resolution,
  pl,
  snaps,
  color = '#22d3ee',
}) => {
  const innerContainerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [lastIdx, setLastIdx] = useState<number | null>(null);
  const [zoom, setZoom] = useState(1);

  // Explicitly calculate unified ticks mirroring both Snap targets and Graph visuals
  const range = yMax - yMin;
  const validYTicks: number[] = [];
  if (snaps && snaps.length > 0) {
    validYTicks.push(...[...snaps].sort((a,b) => a-b));
  } else {
    const yStep = range <= 1.0 ? 0.1 : (range > 15 ? 2 : 1);
    for (let val = yMin; val <= yMax + 0.001; val += yStep) {
      validYTicks.push(parseFloat(val.toFixed(2)));
    }
  }

  const handlePointerEvent = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!innerContainerRef.current) return;
    const rect = innerContainerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const relX = Math.max(0, Math.min(1, x / rect.width));
    const relY = Math.max(0, Math.min(1, 1 - y / rect.height));

    const dataX = relX * xMax;
    let dataY = yMin + relY * (yMax - yMin);

    // Strictly snap input universally parsing `validYTicks` mappings perfectly!
    if (validYTicks.length > 0) {
      let closest = validYTicks[0];
      let minDiff = Math.abs(dataY - closest);
      for (const s of validYTicks) {
        const diff = Math.abs(dataY - s);
        if (diff < minDiff) {
          minDiff = diff;
          closest = s;
        }
      }
      dataY = closest;
    }

    const totalSteps = xMax / resolution;
    const idx = Math.min(totalSteps - 1, Math.floor(dataX / resolution));
    const newData = [...data];

    // Ensure array is large enough
    while (newData.length <= idx) {
      newData.push(yMin); // Fallback init
    }

    if (isDragging && lastIdx !== null) {
      const start = Math.min(lastIdx, idx);
      const end = Math.max(lastIdx, idx);
      const prevVal = data[lastIdx] ?? dataY;

      for (let i = start; i <= end; i++) {
        if (i >= newData.length) newData.push(yMin);
        const t = end > start ? (i - start) / (end - start) : 0;
        let val = idx > lastIdx ? prevVal + t * (dataY - prevVal) : dataY + t * (prevVal - dataY);
        
        if (validYTicks.length > 0) {
          let closest = validYTicks[0];
          let minDiff = Math.abs(val - closest);
          for (const s of validYTicks) {
            const diff = Math.abs(val - s);
            if (diff < minDiff) { minDiff = diff; closest = s; }
          }
          val = closest;
        }
        newData[i] = val;
      }
    } else {
      newData[idx] = dataY;
    }

    setLastIdx(idx);
    onChange(newData);
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    setIsDragging(true);
    innerContainerRef.current?.setPointerCapture(e.pointerId);
    handlePointerEvent(e);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (isDragging) handlePointerEvent(e);
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    setIsDragging(false);
    setLastIdx(null);
    if (innerContainerRef.current?.hasPointerCapture(e.pointerId)) {
        innerContainerRef.current?.releasePointerCapture(e.pointerId);
    }
  };

  // Generate SVG sequence boxes mapping discrete boundaries
  const boxWidth = (resolution / xMax) * 100;
  
  const contourBoxes = data.map((val, i) => {
    const px = (i * resolution / xMax) * 100;
    
    // Find matching track index dynamically matching Y Grid graphics
    let tickIdx = validYTicks.indexOf(val);
    if (tickIdx === -1) {
      let closest = 0;
      let minDiff = Infinity;
      for (let j = 0; j < validYTicks.length; j++) {
         const diff = Math.abs(val - validYTicks[j]);
         if (diff < minDiff) { minDiff = diff; closest = j; }
      }
      tickIdx = closest;
    }
    
    const tickVal = validYTicks[tickIdx];
    const prevVal = tickIdx > 0 ? validYTicks[tickIdx-1] : tickVal;
    const nextVal = tickIdx < validYTicks.length - 1 ? validYTicks[tickIdx+1] : tickVal;

    const vTop = tickIdx === validYTicks.length - 1 ? tickVal + (tickVal - prevVal)/2 : tickVal + (nextVal - tickVal)/2;
    const vBottom = tickIdx === 0 ? tickVal - (nextVal - tickVal)/2 : tickVal - (tickVal - prevVal)/2;
    
    const pyTopEdge = 100 - ((vTop - yMin) / range) * 100;
    const pyBottomEdge = 100 - ((vBottom - yMin) / range) * 100;
    
    const boundedTop = Math.max(0, Math.min(100, pyTopEdge));
    const boundedBottom = Math.max(0, Math.min(100, pyBottomEdge));
    const cellHeight = Math.max(0.2, boundedBottom - boundedTop); // minimum fallback

    return (
      <rect 
        key={`box-${i}`}
        x={px} 
        y={boundedTop} 
        width={boxWidth} 
        height={cellHeight} 
        fill={color}
        stroke="rgba(0,0,0,0.5)"
        strokeWidth="0.5"
        vectorEffect="non-scaling-stroke"
        className="transition-colors duration-75"
      />
    );
  });

  // Generate grid logic plotting boundaries perfectly matching `pl` (Phrase Length)
  const totalSteps = xMax / resolution;
  const gridLines = [];
  const xLabels = [];
  
  for (let i = 0; i <= totalSteps; i++) {
    const px = (i * resolution / xMax) * 100;
    const isPhrase = i % pl === 0;

    if (isPhrase) {
      if (i > 0) {
        gridLines.push(
          <line key={`grid-x-phrase-${i}`} x1={px} y1="0" x2={px} y2="100" stroke="#475569" strokeWidth="1" vectorEffect="non-scaling-stroke" />
        );
      }
      xLabels.push(
        <div key={`x-label-${i}`} className={`absolute bottom-0 -translate-x-1/2 text-[10px] whitespace-nowrap select-none ${isPhrase ? 'text-slate-300 font-bold' : 'text-slate-500'}`} style={{ left: `${px}%` }}>
          {i + 1}.1
        </div>
      );
    } else if (zoom > 2) {
      gridLines.push(
        <line key={`grid-x-bar-${i}`} x1={px} y1="0" x2={px} y2="100" stroke="#334155" strokeWidth="0.5" strokeDasharray="2 2" vectorEffect="non-scaling-stroke" />
      );
      if (zoom > 4) {
        xLabels.push(
          <div key={`x-label-${i}`} className="absolute bottom-0 -translate-x-1/2 text-[10px] text-slate-500 whitespace-nowrap select-none" style={{ left: `${px}%` }}>
            {i + 1}.1
          </div>
        );
      }
    }
  }

  // Find standard line (Zero line) if it spans negative values
  if (yMin < 0 && yMax > 0) {
    const zeroY = 100 - ((0 - yMin) / (yMax - yMin)) * 100;
    gridLines.push(<line key="grid-y-zero" x1="0" y1={zeroY} x2="100" y2={zeroY} stroke="#475569" strokeWidth="1" vectorEffect="non-scaling-stroke" />);
  }

  // Generate piano-roll style checkered Y-Axis grid overlays parsing dynamic snaps precisely
  const yLabelsDOM = [];
  let yBgCount = 0;

  for (let i = 0; i < validYTicks.length; i++) {
     const val = validYTicks[i];
     const prevVal = i > 0 ? validYTicks[i-1] : val;
     const nextVal = i < validYTicks.length - 1 ? validYTicks[i+1] : val;

     // Extrapolate bounding rows precisely filling uneven graphical Snap intervals
     const vTop = i === validYTicks.length - 1 ? val + (val - prevVal)/2 : val + (nextVal - val)/2;
     const vBottom = i === 0 ? val - (nextVal - val)/2 : val - (val - prevVal)/2;
     
     const pyTopEdge = 100 - ((vTop - yMin) / range) * 100;
     const pyBottomEdge = 100 - ((vBottom - yMin) / range) * 100;
     
     const boundedTop = Math.max(0, Math.min(100, pyTopEdge));
     const boundedBottom = Math.max(0, Math.min(100, pyBottomEdge));
     const h = Math.max(0, boundedBottom - boundedTop);

     if (h > 0) {
       const bg = yBgCount % 2 === 0 ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.15)";
       gridLines.unshift(
         <rect key={`y-bg-${yBgCount}`} x="0" y={boundedTop} width="100" height={h} fill={bg} />
       );
     }
     
     const pyCenter = 100 - ((val - yMin) / range) * 100;
     gridLines.push(
       <line key={`grid-y-line-${yBgCount}`} x1="0" y1={pyCenter} x2="100" y2={pyCenter} stroke="rgba(255,255,255,0.05)" strokeWidth="0.5" vectorEffect="non-scaling-stroke" />
     );

     yLabelsDOM.push(
       <div 
         key={`y-label-${yBgCount}`} 
         className="absolute right-2 -translate-y-1/2 whitespace-nowrap text-[10px]" 
         style={{ top: `${Math.max(0, Math.min(100, pyCenter))}%` }}
       >
         {Number.isInteger(val) ? val.toString() : val.toFixed(2)}
       </div>
     );

     yBgCount++;
  }

  const stepString = snaps && snaps.length > 0 
    ? `Snaps: ${snaps.join(', ')}` 
    : `Step Size: ${range <= 1.0 ? 0.1 : (range > 15 ? 2 : 1)}`;

  return (
    <div className="flex flex-col gap-2 w-full h-full select-none min-h-0">
      <div className="flex justify-between items-center text-sm font-semibold text-slate-300 shrink-0">
        <span>{label}</span>
        <span className="text-xs text-slate-500 font-normal bg-slate-900 border border-slate-700 px-2 py-0.5 rounded">
          {stepString}
        </span>
      </div>
      
      <div className="flex flex-row w-full flex-1 gap-2 min-h-0">
        {/* Y Axis with Absolute Absolute Labels */}
        <div className="relative text-[10px] text-slate-500 w-10 border-r border-slate-700 pr-2 h-full shrink-0">
          {yLabelsDOM}
        </div>

        {/* Graph Area Container with Native Scroll/Zoom */}
        <div className="flex-1 flex flex-col min-w-0 h-full relative">
          
          {/* Zoom Overlay Control */}
          <div className="absolute top-2 right-4 z-10 flex gap-2 items-center bg-slate-950/80 px-3 py-1 rounded shadow border border-slate-700 backdrop-blur-sm">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Zoom</span>
            <input type="range" min="1" max="15" step="0.1" value={zoom} onChange={e => setZoom(parseFloat(e.target.value))} className="w-24 accent-cyan-500" />
            <span className="text-[10px] text-slate-500 font-mono w-4">{zoom.toFixed(1)}x</span>
          </div>

          <div 
            className="flex-1 overflow-x-auto overflow-y-hidden w-full h-full bg-slate-900 border border-slate-700 rounded-lg relative"
            onWheel={(e) => {
              if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                setZoom(z => Math.max(1, Math.min(15, z - e.deltaY * 0.01)));
              }
            }}
          >
            {/* The scaled inner drawing canvas tracking mouse naturally */}
            <div 
              ref={innerContainerRef}
              style={{ width: `${zoom * 100}%` }}
              className="h-full relative cursor-crosshair touch-none overflow-hidden group"
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerUp}
            >
              <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
                {gridLines}
                {contourBoxes}
              </svg>
              
              {/* X Axis DAW formatting overlaying bottom grid */}
              <div className="absolute bottom-0 w-full h-5 pointer-events-none opacity-50 group-hover:opacity-100 transition-opacity">
                {xLabels}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
