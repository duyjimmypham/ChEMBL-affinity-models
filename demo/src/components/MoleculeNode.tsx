import { motion } from "motion/react";

interface MoleculeNodeProps {
  delay: number;
}

export function MoleculeNode({ delay }: MoleculeNodeProps) {
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
          <h3 className="text-black mb-1">Input Molecule</h3>
          <p className="text-gray-500">Chemical structure</p>
        </div>

        {/* Molecule image placeholder */}
        <div className="relative h-48 flex items-center justify-center">
          <motion.div
            className="w-full h-full border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center bg-gray-50 overflow-hidden"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: delay * 0.3 + 0.5, duration: 0.5 }}
          >
            <img
              src="/molecule.png"
              alt="Input Molecule"
              className="w-full h-full object-contain p-4"
              onError={(e) => {
                e.currentTarget.style.display = "none";
                e.currentTarget.nextElementSibling?.classList.remove("hidden");
              }}
            />
            <div className="hidden text-center absolute inset-0 flex flex-col items-center justify-center">
              {/* Upload prompt removed */}
            </div>
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
