import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { reportFrontendError } from './utils/crashReporter';
import { installFrontendLogBridge } from './utils/frontendLogBridge';
import './styles.css';

installFrontendLogBridge();

const root = createRoot(document.getElementById('root')!, {
  onUncaughtError: (error, errorInfo) => {
    console.error('Uncaught React error', error, errorInfo.componentStack);
    reportFrontendError('FRONTEND_FATAL', error, 'react-root', errorInfo.componentStack ?? undefined);
  },
});
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
