/* Simple logger abstraction to keep logging consistent across services */
/* eslint-disable no-console */
export const logger = {
  info: (message: string, ...args: unknown[]) => console.log(`[INFO] ${message}`, ...args),
  warn: (message: string, ...args: unknown[]) => console.warn(`[WARN] ${message}`, ...args),
  error: (message: string, ...args: unknown[]) => console.error(`[ERROR] ${message}`, ...args),
  debug: (message: string, ...args: unknown[]) => {
    if (process.env.DEBUG) {
      console.debug(`[DEBUG] ${message}`, ...args);
    }
  },
};

export type Logger = typeof logger;





