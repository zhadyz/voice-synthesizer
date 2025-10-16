import React, { useState } from 'react';
import { useAppStore } from '../store/appStore';
import { AudioUpload } from '../components/AudioUpload';
import { ProgressBar } from '../components/ProgressBar';
import { QualityReport } from '../components/QualityReport';

export function TrainingFlow() {
  const [modelName, setModelName] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isStartingTraining, setIsStartingTraining] = useState(false);

  const {
    uploadTrainingAudio,
    startTraining,
    currentStep,
    progress,
    jobStatus,
    uploadedTrainingAudio,
    error
  } = useAppStore();

  const handleUpload = async (file) => {
    try {
      setIsUploading(true);
      await uploadTrainingAudio(file);
    } catch (err) {
      alert('Upload failed: ' + (err.message || 'Unknown error'));
    } finally {
      setIsUploading(false);
    }
  };

  const handleStartTraining = async () => {
    if (!modelName.trim()) {
      alert('Please enter a model name');
      return;
    }

    try {
      setIsStartingTraining(true);
      await startTraining(modelName);
    } catch (err) {
      alert('Training failed: ' + (err.message || 'Unknown error'));
    } finally {
      setIsStartingTraining(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto px-8">
      <div className="text-center mb-12">
        <h1 className="text-3xl font-bold mb-3 text-gray-900">Train Your Voice Model</h1>
        <p className="text-lg text-gray-600">
          Upload 5-10 minutes of your voice to create a custom voice model
        </p>
      </div>

      {/* Step 1: Upload */}
      {currentStep === 'upload' && (
        <div className="space-y-8 animate-fadeIn">
          {isUploading ? (
            <div className="bg-white rounded-2xl p-12 shadow-elevated">
              <div className="flex flex-col items-center justify-center">
                <svg
                  className="animate-spin h-16 w-16 text-accent-500 mb-6"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                <p className="text-lg text-gray-700 font-medium">Uploading and analyzing your audio...</p>
              </div>
            </div>
          ) : (
            <>
              <AudioUpload
                onUpload={handleUpload}
                label="Upload Your Voice (5-10 minutes)"
              />

              {uploadedTrainingAudio && (
                <>
                  <QualityReport
                    snr={uploadedTrainingAudio.quality_snr}
                    duration={uploadedTrainingAudio.duration}
                    quality={uploadedTrainingAudio.quality_score}
                  />

                  <div className="bg-white rounded-2xl p-8 shadow-elevated">
                    <label className="block text-base font-semibold text-gray-900 mb-3">
                      Model Name
                    </label>
                    <input
                      type="text"
                      value={modelName}
                      onChange={(e) => setModelName(e.target.value)}
                      placeholder="e.g., My Voice Model"
                      className="w-full px-4 py-3 bg-gray-50 border border-gray-300 rounded-lg text-base focus:outline-none focus:ring-2 focus:ring-accent-500 focus:border-transparent transition-all duration-200"
                      disabled={isStartingTraining}
                    />
                    <p className="text-sm text-gray-600 mt-3">
                      Choose a unique name to identify this voice model
                    </p>
                  </div>

                  <button
                    onClick={handleStartTraining}
                    disabled={isStartingTraining || !modelName.trim()}
                    className="w-full px-6 py-4 bg-gradient-to-r from-accent-500 to-accent-600 text-white font-semibold rounded-xl shadow-md hover:shadow-lg hover:scale-[1.02] active:scale-[0.98] transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                  >
                    {isStartingTraining ? 'Starting Training...' : 'Start Training (30-40 minutes)'}
                  </button>

                  <div className="bg-blue-50 border-2 border-blue-200 rounded-xl p-6">
                    <div className="flex items-start gap-3">
                      <svg className="w-6 h-6 text-blue-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                      </svg>
                      <p className="text-sm text-blue-700 leading-relaxed">
                        <strong>Note:</strong> Training takes approximately 30-40 minutes on RTX 3070.
                        You can close this page and come back later - training will continue in the background.
                      </p>
                    </div>
                  </div>
                </>
              )}
            </>
          )}
        </div>
      )}

      {/* Step 2: Training Progress */}
      {currentStep === 'training' && (
        <div className="space-y-8 animate-fadeIn">
          <div className="bg-white rounded-2xl p-10 shadow-elevated">
            <h2 className="text-2xl font-bold mb-6 text-gray-900">Training in Progress</h2>
            <ProgressBar
              progress={progress}
              status={jobStatus}
              message="Your voice model is being trained. This takes 30-40 minutes on RTX 3070."
            />

            <div className="mt-8 bg-gradient-to-r from-gray-50 to-gray-100 rounded-xl p-6 border border-gray-200">
              <h3 className="text-base font-bold text-gray-900 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-accent-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clipRule="evenodd" />
                </svg>
                What's happening?
              </h3>
              <ul className="text-sm text-gray-700 space-y-2">
                <li className="flex items-start gap-3">
                  <span className="text-accent-500 mt-0.5">•</span>
                  <span>Preprocessing audio and extracting features</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-accent-500 mt-0.5">•</span>
                  <span>Training RVC model on your voice characteristics</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-accent-500 mt-0.5">•</span>
                  <span>Optimizing model parameters for best quality</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-accent-500 mt-0.5">•</span>
                  <span>Validating model performance</span>
                </li>
              </ul>
            </div>

            <div className="mt-6 p-5 bg-yellow-50 border-2 border-yellow-200 rounded-xl">
              <div className="flex items-start gap-3">
                <svg className="w-6 h-6 text-yellow-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
                <p className="text-sm text-yellow-700 leading-relaxed">
                  You can safely close this page. Training will continue in the background,
                  and you can check back later to see the results.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mt-8 bg-red-50 border-2 border-red-200 rounded-xl p-6 animate-fadeIn">
          <div className="flex items-start gap-4">
            <svg
              className="w-6 h-6 text-red-500 flex-shrink-0 mt-0.5"
              fill="currentColor"
              viewBox="0 0 20 20"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clipRule="evenodd"
              />
            </svg>
            <div>
              <h3 className="text-base font-bold text-red-900 mb-1">Error</h3>
              <p className="text-sm text-red-700 leading-relaxed">{error}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
