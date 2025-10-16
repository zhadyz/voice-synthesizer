import React from 'react';
import { TrainingFlow } from './pages/TrainingFlow';
import { ConversionFlow } from './pages/ConversionFlow';
import { useAppStore } from './store/appStore';

export default function App() {
  const { currentStep, reset } = useAppStore();

  const showConversionFlow = ['upload-target', 'converting', 'complete'].includes(currentStep);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header - Glassmorphism */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-white/80 border-b border-gray-200/50">
        <div className="max-w-6xl mx-auto px-8 py-6">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-semibold tracking-tight text-gray-900">
                Voice Synthesizer
              </h1>
              <p className="text-sm text-gray-600 mt-1">
                Professional voice cloning & synthesis
              </p>
            </div>
            <button
              onClick={reset}
              className="px-6 py-3 bg-white text-gray-700 font-medium rounded-xl border border-gray-300 hover:bg-gray-50 hover:border-gray-400 hover:shadow-sm transition-all duration-200"
            >
              Start Over
            </button>
          </div>
        </div>
      </header>

      {/* Progress Indicator - Enhanced Stepper */}
      <div className="bg-white/60 backdrop-blur border-b border-gray-200/50">
        <div className="max-w-6xl mx-auto px-8 py-6">
          <nav className="flex items-center justify-center gap-3" aria-label="Progress">
            <div className={`flex items-center transition-all duration-300 ${currentStep === 'upload' || currentStep === 'training' ? 'text-accent-600' : 'text-gray-400'}`}>
              <span className={`flex items-center justify-center w-10 h-10 rounded-full border-2 font-medium transition-all duration-300 ${currentStep === 'upload' || currentStep === 'training' ? 'border-accent-600 bg-accent-600 text-white shadow-md scale-110' : 'border-gray-300 bg-white'}`}>
                1
              </span>
              <span className="ml-3 text-base font-medium">Train Model</span>
            </div>
            <svg className="w-6 h-6 text-gray-300 mx-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
            </svg>
            <div className={`flex items-center transition-all duration-300 ${showConversionFlow ? 'text-accent-600' : 'text-gray-400'}`}>
              <span className={`flex items-center justify-center w-10 h-10 rounded-full border-2 font-medium transition-all duration-300 ${showConversionFlow ? 'border-accent-600 bg-accent-600 text-white shadow-md scale-110' : 'border-gray-300 bg-white'}`}>
                2
              </span>
              <span className="ml-3 text-base font-medium">Convert Audio</span>
            </div>
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <main className="py-12">
        {showConversionFlow ? <ConversionFlow /> : <TrainingFlow />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-6xl mx-auto px-8 py-8 text-center">
          <p className="text-sm font-medium text-gray-700">
            Powered by RVC + BS-RoFormer
          </p>
          <p className="text-xs text-gray-500 mt-2">
            100% Offline • Privacy-First Voice Cloning
          </p>
        </div>
      </footer>
    </div>
  );
}
