import React from 'react';
import { TrainingFlow } from './pages/TrainingFlow';
import { ConversionFlow } from './pages/ConversionFlow';
import { useAppStore } from './store/appStore';

export default function App() {
  const { currentStep, reset } = useAppStore();

  const showConversionFlow = ['upload-target', 'converting', 'complete'].includes(currentStep);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Voice Cloning Studio
              </h1>
              <p className="text-sm text-gray-500 mt-1">
                Train custom voice models and convert audio
              </p>
            </div>
            <button
              onClick={reset}
              className="px-4 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md transition"
            >
              Start Over
            </button>
          </div>
        </div>
      </header>

      {/* Progress Indicator */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-3 sm:px-6 lg:px-8">
          <nav className="flex items-center justify-center space-x-4" aria-label="Progress">
            <div className={`flex items-center ${currentStep === 'upload' || currentStep === 'training' ? 'text-primary' : 'text-gray-400'}`}>
              <span className={`flex items-center justify-center w-8 h-8 rounded-full border-2 ${currentStep === 'upload' || currentStep === 'training' ? 'border-primary bg-primary text-white' : 'border-gray-300'}`}>
                1
              </span>
              <span className="ml-2 text-sm font-medium">Train Model</span>
            </div>
            <svg className="w-5 h-5 text-gray-300" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
            </svg>
            <div className={`flex items-center ${showConversionFlow ? 'text-primary' : 'text-gray-400'}`}>
              <span className={`flex items-center justify-center w-8 h-8 rounded-full border-2 ${showConversionFlow ? 'border-primary bg-primary text-white' : 'border-gray-300'}`}>
                2
              </span>
              <span className="ml-2 text-sm font-medium">Convert Audio</span>
            </div>
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <main className="py-8">
        {showConversionFlow ? <ConversionFlow /> : <TrainingFlow />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 py-6 text-center">
          <p className="text-sm text-gray-500">
            Powered by RVC + BS-RoFormer
          </p>
          <p className="text-xs text-gray-400 mt-1">
            100% Offline • Privacy-First Voice Cloning
          </p>
        </div>
      </footer>
    </div>
  );
}
