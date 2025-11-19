import { motion } from "motion/react";

interface ProcessingNodeProps {
  delay: number;
}

export function ProcessingNode({ delay }: ProcessingNodeProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: delay * 0.3, duration: 0.5 }}
      className="relative flex-1"
    >
      <div className="relative bg-white border-2 border-gray-800 rounded-lg p-8 shadow-2xl">
        {/* Label */}
        <div className="mb-6">
          <h3 className="text-black mb-1">ML Model</h3>
          <p className="text-gray-500">Neural network processing</p>
        </div>

        {/* Loading animation */}
        <div className="relative h-48 flex items-center justify-center">
          {/* Circular loader */}
          <div className="relative w-32 h-32">
            {/* Outer ring */}
            <motion.div
              className="absolute inset-0 rounded-full border-2 border-gray-700"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: delay * 0.3 + 0.5 }}
            />

            {/* Animated ring segments */}
            {[0, 120, 240].map((rotation, i) => (
              <motion.div
                key={i}
                className="absolute inset-0"
                initial={{ rotate: rotation, opacity: 0 }}
                animate={{ 
                  rotate: rotation + 360,
                  opacity: 1 
                }}
                transition={{
                  rotate: {
                    duration: 3,
                    repeat: Infinity,
                    ease: "linear",
                    delay: delay * 1.2,
                  },
                  opacity: {
                    duration: 0.3,
                    delay: delay * 0.3 + 0.5 + i * 0.1,
                  }
                }}
              >
                <div className="w-full h-full rounded-full border-2 border-transparent border-t-black" />
              </motion.div>
            ))}

            {/* Center pulsing dot */}
            <motion.div
              className="absolute inset-0 flex items-center justify-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: delay * 0.3 + 0.8 }}
            >
              <motion.div
                className="w-3 h-3 bg-black rounded-full"
                animate={{
                  scale: [1, 1.5, 1],
                  opacity: [1, 0.5, 1],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  delay: delay * 1.2,
                  ease: "easeInOut",
                }}
              />
            </motion.div>

            {/* Orbiting particles */}
            {[0, 60, 120, 180, 240, 300].map((angle, i) => {
              const radius = 50;
              const x = Math.cos((angle * Math.PI) / 180) * radius;
              const y = Math.sin((angle * Math.PI) / 180) * radius;
              
              return (
                <motion.div
                  key={i}
                  className="absolute top-1/2 left-1/2 w-1.5 h-1.5 bg-gray-400 rounded-full"
                  initial={{ 
                    x: -0.75,
                    y: -0.75,
                    opacity: 0 
                  }}
                  animate={{
                    x: [x, -x, x],
                    y: [y, -y, y],
                    opacity: [0, 1, 0],
                  }}
                  transition={{
                    duration: 4,
                    repeat: Infinity,
                    delay: delay * 1.2 + i * 0.2,
                    ease: "easeInOut",
                  }}
                />
              );
            })}
          </div>

          {/* Processing text */}
          <motion.div
            className="absolute bottom-0 text-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{
              duration: 2,
              repeat: Infinity,
              delay: delay * 1.2,
              ease: "easeInOut",
            }}
          >
            <span className="text-gray-400">Processing</span>
            <motion.span
              className="text-gray-400"
              animate={{ opacity: [0, 1, 0] }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                delay: delay * 1.2,
                ease: "easeInOut",
              }}
            >
              ...
            </motion.span>
          </motion.div>
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