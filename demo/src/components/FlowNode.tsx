import { motion } from "motion/react";
import { LucideIcon } from "lucide-react";

interface FlowNodeProps {
  icon: LucideIcon;
  title: string;
  description: string;
  delay: number;
  color: string;
}

export function FlowNode({ icon: Icon, title, description, delay, color }: FlowNodeProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: delay * 0.3, duration: 0.5 }}
      className="relative flex-1"
    >
      {/* Pulsing glow effect */}
      <motion.div
        className={`absolute inset-0 bg-gradient-to-br ${color} rounded-2xl blur-xl opacity-50`}
        animate={{
          opacity: [0.3, 0.6, 0.3],
          scale: [1, 1.05, 1],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          delay: delay * 1.2,
          ease: "easeInOut"
        }}
      />

      {/* Card */}
      <motion.div
        className="relative bg-slate-800/90 backdrop-blur-sm border border-slate-700 rounded-2xl p-6 shadow-xl"
        animate={{
          borderColor: ["rgba(148, 163, 184, 0.3)", "rgba(168, 85, 247, 0.5)", "rgba(148, 163, 184, 0.3)"],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          delay: delay * 1.2,
          ease: "easeInOut"
        }}
      >
        {/* Icon container */}
        <motion.div
          className={`w-16 h-16 mx-auto mb-4 rounded-xl bg-gradient-to-br ${color} flex items-center justify-center shadow-lg`}
          animate={{
            rotate: [0, 5, -5, 0],
            scale: [1, 1.1, 1],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            delay: delay * 1.2,
            ease: "easeInOut"
          }}
        >
          <Icon className="size-8 text-white" />
        </motion.div>

        {/* Text content */}
        <h3 className="text-white text-center mb-2">{title}</h3>
        <p className="text-slate-400 text-center">{description}</p>

        {/* Data flow indicator */}
        <motion.div
          className="mt-4 h-1 bg-slate-700 rounded-full overflow-hidden"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: delay * 0.3 + 0.5 }}
        >
          <motion.div
            className={`h-full bg-gradient-to-r ${color}`}
            animate={{
              x: ["-100%", "100%"],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              delay: delay * 1.2,
              ease: "linear"
            }}
          />
        </motion.div>
      </motion.div>
    </motion.div>
  );
}
