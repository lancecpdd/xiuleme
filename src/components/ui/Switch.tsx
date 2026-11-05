import React, { useId } from 'react';
import clsx from 'clsx';
import { motion } from 'framer-motion';
import Text from './Text';
import { TextVariants } from '../../types/typography';

interface SwitchProps {
  checked: boolean;
  className?: string;
  disabled?: boolean;
  id?: string;
  label: React.ReactNode;
  onChange(val: boolean): any;
  tooltip?: string;
  trackClassName?: string;
}

/**
 * A beautiful, reusable, and accessible toggle switch component.
 *
 * @param {React.ReactNode} label - The text (or custom node) label for the switch.
 * @param {boolean} checked - The current state of the switch.
 * @param {function(boolean): void} onChange - Callback function that receives the new boolean state.
 * @param {boolean} [disabled=false] - Whether the switch is interactive.
 * @param {string} [className=''] - Additional classes for the container.
 * @param {string} [trackClassName] - Custom classes for the switch's background track.
 */
const Switch = ({
  checked,
  className = '',
  disabled = false,
  id,
  label,
  onChange,
  tooltip,
  trackClassName,
}: SwitchProps) => {
  const reactId = useId();
  const uniqueId = id || `switch-${reactId}`;

  const spring = {
    type: 'spring',
    stiffness: 700,
    damping: 30,
  } as const;

  return (
    <label
      className={clsx(
        'flex items-center justify-between',
        disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer',
        className,
      )}
      htmlFor={uniqueId}
      data-tooltip={tooltip}
    >
      <Text variant={TextVariants.label} className="select-none inline-flex items-center gap-1 min-w-0">
        {label}
      </Text>
      <div className="relative w-10 h-5 shrink-0">
        <input
          checked={checked}
          className="sr-only"
          disabled={disabled}
          id={uniqueId}
          onChange={(e: any) => !disabled && onChange(e.target.checked)}
          type="checkbox"
        />
        <div className={clsx('w-full h-full bg-card-active/50 rounded-full shadow-inner', trackClassName)}></div>
        <motion.div
          className={clsx('absolute top-0.5 left-0.5 w-4 h-4 rounded-full transition-colors', {
            'bg-accent': checked,
            'bg-text-secondary/80': !checked,
          })}
          transition={spring}
          initial={false}
          animate={{ x: checked ? 20 : 0 }}
        />
      </div>
    </label>
  );
};

export default Switch;
