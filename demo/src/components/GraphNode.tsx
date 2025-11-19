import { motion } from "motion/react";
import { useState, useEffect } from "react";

interface GraphNodeProps {
  delay: number;
}

export function GraphNode({ delay }: GraphNodeProps) {
  const [picValue, setPicValue] = useState("8.2");

  useEffect(() => {
    fetch('/data.json')
      .then(res => res.json())
      .then(data => {
        if (data.prediction) setPicValue(data.prediction);
      })
      .catch(err => console.log("Using default prediction"));
  }, []);

  // Data points for the graph
  const dataPoints = [
    { x: 20, y: 140 },
    { x: 40, y: 120 },
    { x: 60, y: 130 },
    { x: 80, y: 100 },
    { x: 100, y: 90 },
    { x: 120, y: 70 },
    { x: 140, y: 60 },
    { x: 160, y: 50 },
    { x: 180, y: 40 },
  ];

  // Generate path for line graph
  const linePath = dataPoints
    .map((point, i) => `${i === 0 ? "M" : "L"} ${point.x} ${point.y}`)
    .join(" ");

  // Generate path for area under curve
  const areaPath = `${linePath} L 180 160 L 20 160 Z`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: delay * 0.3, duration: 0.5 }}
      className="relative flex-1"
    >
      <div className="relative bg-white border border-gray-200 rounded-lg p-8 shadow-2xl">
        {/* Label */}
        <div className="mb-6">
          <h3 className="text-black mb-1">Predicted Affinity (pIC50)</h3>
          <p className="text-gray-500">Binding affinity score</p>
        </div>

        {/* Graph visualization */}
        <div className="relative h-48">
          <svg
            width="100%"
            height="100%"
            viewBox="0 0 200 180"
            className="overflow-visible"
          >
            {/* Grid lines */}
            {[40, 80, 120, 160].map((y, i) => (
              <motion.line
                key={`h-${i}`}
                x1="20"
                y1={y}
                x2="180"
                y2={y}
                stroke="#E5E7EB"
                strokeWidth="1"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ delay: delay * 0.3 + 0.5 + i * 0.05, duration: 0.3 }}
              />
            ))}
            {[60, 100, 140].map((x, i) => (
              <motion.line
                key={`v-${i}`}
                x1={x}
                y1="20"
                x2={x}
                y2="160"
                stroke="#E5E7EB"
                strokeWidth="1"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ delay: delay * 0.3 + 0.5 + i * 0.05, duration: 0.3 }}
              />
            ))}

            {/* Axes */}
            <motion.line
              x1="20"
              y1="160"
              x2="180"
              y2="160"
              stroke="#000"
              strokeWidth="2"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ delay: delay * 0.3 + 0.7, duration: 0.4 }}
            />
            <motion.line
              x1="20"
              y1="20"
              x2="20"
              y2="160"
              stroke="#000"
              strokeWidth="2"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ delay: delay * 0.3 + 0.7, duration: 0.4 }}
            />

            {/* Area under curve */}
            <motion.path
              d={areaPath}
              fill="url(#gradient)"
              opacity="0.2"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.2 }}
              transition={{ 
                delay: delay * 0.3 + 0.9, 
                duration: 1,
                repeat: Infinity,
                repeatType: "loop",
                repeatDelay: 2
              }}
            />

            {/* Line graph */}
            <motion.path
              d={linePath}
              stroke="#000"
              strokeWidth="2"
              fill="none"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ 
                delay: delay * 0.3 + 1.0, 
                duration: 1, 
                ease: "easeOut",
                repeat: Infinity,
                repeatType: "loop",
                repeatDelay: 2
              }}
            />

            {/* Data points */}
            {dataPoints.map((point, i) => (
              <motion.circle
                key={i}
                cx={point.x}
                cy={point.y}
                r="3"
                fill="#000"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ 
                  delay: delay * 0.3 + 1.0 + i * 0.08, 
                  duration: 0.2,
                  repeat: Infinity,
                  repeatType: "loop",
                  repeatDelay: 2
                }}
              />
            ))}

            {/* Highlight last point */}
            <motion.circle
              cx={dataPoints[dataPoints.length - 1].x}
              cy={dataPoints[dataPoints.length - 1].y}
              r="5"
              fill="none"
              stroke="#000"
              strokeWidth="2"
              initial={{ scale: 0, opacity: 0 }}
              animate={{ 
                scale: [1, 1.3, 1],
                opacity: 1 
              }}
              transition={{
                scale: {
                  duration: 2,
                  repeat: Infinity,
                  delay: delay * 1.2,
                  ease: "easeInOut",
                },
                opacity: {
                  duration: 0.3,
                  delay: delay * 0.3 + 1.6,
                }
              }}
            />

            {/* Value label */}
            <motion.g
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: delay * 0.3 + 1.7, duration: 0.3 }}
            >
              <rect
                x="145"
                y="20"
                width="50"
                height="26"
                fill="#fff"
                stroke="#000"
                strokeWidth="1"
                rx="4"
              />
              <foreignObject x="147" y="22" width="46" height="22">
                <input
                  type="text"
                  value={picValue}
                  onChange={(e) => setPicValue(e.target.value)}
                  className="w-full h-full text-center bg-transparent border-none outline-none text-xs"
                  style={{ font: 'inherit' }}
                />
              </foreignObject>
            </motion.g>

            {/* Gradient definition */}
            <defs>
              <linearGradient id="gradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#000" stopOpacity="0.3" />
                <stop offset="100%" stopColor="#000" stopOpacity="0" />
              </linearGradient>
            </defs>
          </svg>
        </div>

        {/* Progress indicator */}
        <motion.div
          className="mt-6 h-0.5 bg-gray-200 rounded-full overflow-hidden"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: delay * 0.3 + 0.5 }}
        >
          <motion.div
            className="h-full bg-black"
            animate={{
              x: ["-100%", "100%"],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              delay: delay * 1.2,
              ease: "linear",
            }}
          />
        </motion.div>
      </div>
    </motion.div>
  );
}