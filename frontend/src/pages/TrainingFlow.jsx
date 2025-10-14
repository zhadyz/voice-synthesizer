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
    <div className="max-w-4xl mx-auto p-8">
      <h1 className="text-3xl font-bold mb-2 text-gray-900">Train Your Voice Model</h1>
      <p className="text-gray-600 mb-8">
        Upload 5-10 minutes of your voice to create a custom voice model
      </p>

      {/* Step 1: Upload */}
      {currentStep === 'upload' && (
        <div className="space-y-6">
          {isUploading ? (
            <div className="bg-white rounded-lg p-8 shadow">
              <div className="flex flex-col items-center justify-center">
                <svg
                  className="animate-spin h-12 w-12 text-primary mb-4"
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
                <p className="text-gray-600">Uploading and analyzing your audio...</p>
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

                  <div className="bg-white rounded-lg p-6 shadow">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Model Name
                    </label>
                    <input
                      type="text"
                      value={modelName}
                      onChange={(e) => setModelName(e.target.value)}
                      placeholder="e.g., My Voice Model"
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary outline-none"
                      disabled={isStartingTraining}
                    />
                    <p className="text-xs text-gray-500 mt-2">
                      Choose a unique name to identify this voice model
                    </p>
                  </div>

                  <button
                    onClick={handleStartTraining}
                    disabled={isStartingTraining || !modelName.trim()}
                    className="w-full bg-primary text-white py-3 rounded-lg font-medium hover:bg-primary/90 transition disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isStartingTraining ? 'Starting Training...' : 'Start Training (30-40 minutes)'}
                  </button>

                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <p className="text-sm text-blue-700">
                      <strong>Note:</strong> Training takes approximately 30-40 minutes on RTX 3070.
                      You can close this page and come back later - training will continue in the background.
                    </p>
                  </div>
                </>
              )}
            </>
          )}
        </div>
      )}

      {/* Step 2: Training Progress */}
      {currentStep === 'training' && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg p-8 shadow">
            <h2 className="text-xl font-semibold mb-4 text-gray-900">Training in Progress</h2>
            <ProgressBar
              progress={progress}
              status={jobStatus}
              message="Your voice model is being trained. This takes 30-40 minutes on RTX 3070."
            />

            <div className="mt-6 bg-gray-50 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">What's happening?</h3>
              <ul className="text-sm text-gray-600 space-y-1 list-disc list-inside">
                <li>Preprocessing audio and extracting features</li>
                <li>Training RVC model on your voice characteristics</li>
                <li>Optimizing model parameters for best quality</li>
                <li>Validating model performance</li>
              </ul>
            </div>

            <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
              <p className="text-sm text-yellow-700">
                You can safely close this page. Training will continue in the background,
                and you can check back later to see the results.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mt-6 bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-start">
            <svg
              className="w-5 h-5 text-red-400 mt-0.5 mr-3"
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
              <h3 className="text-sm font-medium text-red-800">Error</h3>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
