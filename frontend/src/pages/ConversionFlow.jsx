import React, { useState } from 'react';
import { useAppStore } from '../store/appStore';
import { AudioUpload } from '../components/AudioUpload';
import { ProgressBar } from '../components/ProgressBar';
import { WaveformPlayer } from '../components/WaveformPlayer';

export function ConversionFlow() {
  const [isUploading, setIsUploading] = useState(false);

  const {
    uploadTargetAudio,
    startConversion,
    currentStep,
    progress,
    jobStatus,
    convertedAudioUrl,
    trainedModel,
    error
  } = useAppStore();

  const handleUpload = async (file) => {
    try {
      setIsUploading(true);
      await uploadTargetAudio(file);
      await startConversion(trainedModel.id || trainedModel.model_id);
    } catch (err) {
      alert('Upload failed: ' + (err.message || 'Unknown error'));
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-8">
      <h1 className="text-3xl font-bold mb-2 text-gray-900">Convert Audio to Your Voice</h1>
      <p className="text-gray-600 mb-8">
        Upload any audio (song, speech, etc.) to convert it to your trained voice
      </p>

      {/* Model Info Card */}
      {trainedModel && (
        <div className="mb-6 bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center">
            <svg
              className="w-5 h-5 text-green-500 mr-2"
              fill="currentColor"
              viewBox="0 0 20 20"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                clipRule="evenodd"
              />
            </svg>
            <div>
              <p className="text-sm font-medium text-green-800">
                Using model: <strong>{trainedModel.model_name || trainedModel.name || 'Trained Model'}</strong>
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Upload Target Audio */}
      {currentStep === 'upload-target' && (
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
                <p className="text-gray-600">Uploading and starting conversion...</p>
              </div>
            </div>
          ) : (
            <>
              <AudioUpload
                onUpload={handleUpload}
                label="Upload Audio to Convert (e.g., song, speech)"
              />

              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <p className="text-sm text-blue-700">
                  <strong>Tip:</strong> For best results, upload clear audio without heavy background music.
                  The conversion typically takes 2-5 minutes depending on audio length.
                </p>
              </div>
            </>
          )}
        </div>
      )}

      {/* Conversion Progress */}
      {currentStep === 'converting' && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg p-8 shadow">
            <h2 className="text-xl font-semibold mb-4 text-gray-900">Converting Audio</h2>
            <ProgressBar
              progress={progress}
              status={jobStatus}
              message="Converting target audio to your voice..."
            />

            <div className="mt-6 bg-gray-50 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">Processing Steps</h3>
              <ul className="text-sm text-gray-600 space-y-1 list-disc list-inside">
                <li>Separating vocals from background music (if present)</li>
                <li>Extracting voice features from target audio</li>
                <li>Applying your trained voice model</li>
                <li>Generating final output with your voice</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Conversion Complete */}
      {currentStep === 'complete' && convertedAudioUrl && (
        <div className="space-y-6">
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="flex items-center">
              <svg
                className="w-6 h-6 text-green-500 mr-2"
                fill="currentColor"
                viewBox="0 0 20 20"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                  clipRule="evenodd"
                />
              </svg>
              <p className="text-green-700 font-medium">
                Conversion complete! Listen to your result below.
              </p>
            </div>
          </div>

          <WaveformPlayer
            audioUrl={convertedAudioUrl}
            title="Converted Audio (Your Voice)"
          />

          <a
            href={convertedAudioUrl}
            download="converted_audio.wav"
            className="block w-full bg-primary text-white text-center py-3 rounded-lg font-medium hover:bg-primary/90 transition"
          >
            Download Converted Audio
          </a>

          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-2">What's Next?</h3>
            <p className="text-sm text-gray-600">
              Want to convert another audio file? Click "Start Over" at the top to begin a new conversion,
              or train a new voice model for different voices.
            </p>
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
