import { motion } from "motion/react";
import { ArrowRight } from "lucide-react";
import { MoleculeNode } from "./MoleculeNode";
import { ProcessingNode } from "./ProcessingNode";
import { GraphNode } from "./GraphNode";
import { AnimatedConnector } from "./AnimatedConnector";

export function MLFlowchart() {
  return (
    <div className="w-full max-w-6xl">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center mb-16"
      >
        <h1 className="text-white mb-2">Machine Learning Prediction Process</h1>
        <p className="text-gray-400">Molecular affinity prediction workflow</p>
      </motion.div>

      <div className="relative flex items-center justify-between gap-4">
        {/* Step 1: Input Molecule */}
        <MoleculeNode delay={0} />

        {/* Connector 1 */}
        <AnimatedConnector delay={1} />

        {/* Step 2: ML Model */}
        <ProcessingNode delay={2} />

        {/* Connector 2 */}
        <AnimatedConnector delay={3} />

        {/* Step 3: Predicted Affinity */}
        <GraphNode delay={4} />
      </div>

      {/* Loop indicator */}
      <motion.div
        className="mt-16 flex items-center justify-center gap-2 text-gray-500"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 5, duration: 0.5 }}
      >
        <motion.div
          animate={{ 
            x: [0, 8, 0],
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        >
          <ArrowRight className="size-4" />
        </motion.div>
        <span>Continuous prediction cycle</span>
      </motion.div>
    </div>
  );
}