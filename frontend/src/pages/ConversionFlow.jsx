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
    <div className="max-w-5xl mx-auto px-8">
      <div className="text-center mb-12">
        <h1 className="text-3xl font-bold mb-3 text-gray-900">Convert Audio to Your Voice</h1>
        <p className="text-lg text-gray-600">
          Upload any audio (song, speech, etc.) to convert it to your trained voice
        </p>
      </div>

      {/* Model Info Card */}
      {trainedModel && (
        <div className="mb-8 bg-green-50 border-2 border-green-200 rounded-xl p-6">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center flex-shrink-0">
              <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium text-green-700 mb-1">Using trained model</p>
              <p className="text-lg font-bold text-green-900">
                {trainedModel.model_name || trainedModel.name || 'Trained Model'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Upload Target Audio */}
      {currentStep === 'upload-target' && (
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
                <p className="text-lg text-gray-700 font-medium">Uploading and starting conversion...</p>
              </div>
            </div>
          ) : (
            <>
              <AudioUpload
                onUpload={handleUpload}
                label="Upload Audio to Convert (e.g., song, speech)"
              />

              <div className="bg-blue-50 border-2 border-blue-200 rounded-xl p-6">
                <div className="flex items-start gap-3">
                  <svg className="w-6 h-6 text-blue-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                  </svg>
                  <p className="text-sm text-blue-700 leading-relaxed">
                    <strong>Tip:</strong> For best results, upload clear audio without heavy background music.
                    The conversion typically takes 2-5 minutes depending on audio length.
                  </p>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* Conversion Progress */}
      {currentStep === 'converting' && (
        <div className="space-y-8 animate-fadeIn">
          <div className="bg-white rounded-2xl p-10 shadow-elevated">
            <h2 className="text-2xl font-bold mb-6 text-gray-900">Converting Audio</h2>
            <ProgressBar
              progress={progress}
              status={jobStatus}
              message="Converting target audio to your voice..."
            />

            <div className="mt-8 bg-gradient-to-r from-gray-50 to-gray-100 rounded-xl p-6 border border-gray-200">
              <h3 className="text-base font-bold text-gray-900 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-accent-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clipRule="evenodd" />
                </svg>
                Processing Steps
              </h3>
              <ul className="text-sm text-gray-700 space-y-2">
                <li className="flex items-start gap-3">
                  <span className="text-accent-500 mt-0.5">•</span>
                  <span>Separating vocals from background music (if present)</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-accent-500 mt-0.5">•</span>
                  <span>Extracting voice features from target audio</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-accent-500 mt-0.5">•</span>
                  <span>Applying your trained voice model</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-accent-500 mt-0.5">•</span>
                  <span>Generating final output with your voice</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Conversion Complete */}
      {currentStep === 'complete' && convertedAudioUrl && (
        <div className="space-y-8 animate-fadeIn">
          {/* Success Banner */}
          <div className="text-center py-12">
            <div className="w-24 h-24 mx-auto mb-6 bg-gradient-to-br from-green-400 to-green-600 rounded-full flex items-center justify-center shadow-lg animate-bounce-soft">
              <svg className="w-14 h-14 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            </div>
            <h2 className="text-3xl font-bold text-gray-900 mb-2">Conversion Complete!</h2>
            <p className="text-lg text-gray-600">Your synthesized audio is ready</p>
          </div>

          <WaveformPlayer
            audioUrl={convertedAudioUrl}
            title="Converted Audio (Your Voice)"
          />

          <a
            href={convertedAudioUrl}
            download="converted_audio.wav"
            className="block w-full px-6 py-4 bg-gradient-to-r from-accent-500 to-accent-600 text-white text-center font-semibold rounded-xl shadow-md hover:shadow-lg hover:scale-[1.02] active:scale-[0.98] transition-all duration-200"
          >
            Download Converted Audio
          </a>

          <div className="bg-gradient-to-r from-gray-50 to-gray-100 border border-gray-200 rounded-xl p-6">
            <h3 className="text-base font-bold text-gray-900 mb-3 flex items-center gap-2">
              <svg className="w-5 h-5 text-accent-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
              What's Next?
            </h3>
            <p className="text-sm text-gray-700 leading-relaxed">
              Want to convert another audio file? Click "Start Over" at the top to begin a new conversion,
              or train a new voice model for different voices.
            </p>
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
