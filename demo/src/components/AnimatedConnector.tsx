import { motion } from "motion/react";

interface AnimatedConnectorProps {
  delay: number;
}

export function AnimatedConnector({ delay }: AnimatedConnectorProps) {
  return (
    <div className="relative flex items-center justify-center w-20">
      {/* Arrow line */}
      <motion.div
        className="h-0.5 w-full bg-gray-800 relative overflow-hidden"
        initial={{ scaleX: 0 }}
        animate={{ scaleX: 1 }}
        transition={{ delay: delay * 0.3, duration: 0.5 }}
      >
        {/* Flowing particles */}
        {[0, 0.4, 0.8].map((particleDelay, index) => (
          <motion.div
            key={index}
            className="absolute top-0 left-0 w-2 h-full bg-white rounded-full shadow-lg"
            animate={{
              x: ["-8px", "88px"],
              opacity: [0, 1, 1, 0],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              delay: delay * 1.2 + particleDelay,
              ease: "linear"
            }}
          />
        ))}
      </motion.div>

      {/* Arrow head */}
      <motion.div
        className="absolute right-0"
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: delay * 0.3 + 0.3, duration: 0.3 }}
      >
        <motion.div
          animate={{
            x: [0, 3, 0],
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            delay: delay * 1.2,
            ease: "easeInOut"
          }}
        >
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path
              d="M3 2 L10 6 L3 10 Z"
              fill="#000"
              stroke="#000"
              strokeWidth="1"
            />
          </svg>
        </motion.div>
      </motion.div>
    </div>
  );
}