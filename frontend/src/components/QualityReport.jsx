import React from 'react';

export function QualityReport({ snr, duration, quality }) {
  const getQualityColor = (quality) => {
    switch (quality) {
      case 'EXCELLENT':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'GOOD':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'ACCEPTABLE':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'POOR':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getQualityIcon = (quality) => {
    switch (quality) {
      case 'EXCELLENT':
      case 'GOOD':
        return (
          <svg
            className="w-5 h-5 text-green-500 inline-block mr-1"
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
        );
      case 'ACCEPTABLE':
        return (
          <svg
            className="w-5 h-5 text-yellow-500 inline-block mr-1"
            fill="currentColor"
            viewBox="0 0 20 20"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              fillRule="evenodd"
              d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
              clipRule="evenodd"
            />
          </svg>
        );
      case 'POOR':
        return (
          <svg
            className="w-5 h-5 text-red-500 inline-block mr-1"
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
        );
      default:
        return null;
    }
  };

  return (
    <div className="bg-white rounded-2xl p-8 shadow-elevated">
      <h3 className="text-xl font-semibold mb-6 text-gray-900">Audio Quality Report</h3>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        <div className="bg-gray-50 rounded-xl p-5 border border-gray-200">
          <p className="text-sm font-medium text-gray-600 mb-2">Signal-to-Noise Ratio</p>
          <p className="text-2xl font-bold text-gray-900">
            {snr !== null && snr !== undefined ? `${snr.toFixed(1)} dB` : 'N/A'}
          </p>
        </div>

        <div className="bg-gray-50 rounded-xl p-5 border border-gray-200">
          <p className="text-sm font-medium text-gray-600 mb-2">Duration</p>
          <p className="text-2xl font-bold text-gray-900">
            {duration !== null && duration !== undefined ? `${duration.toFixed(1)}s` : 'N/A'}
          </p>
        </div>
      </div>

      {/* Overall Quality Badge */}
      <div className="flex items-center justify-between p-5 bg-gradient-to-r from-gray-50 to-gray-100 rounded-xl border border-gray-200">
        <span className="text-base font-semibold text-gray-900">Overall Quality</span>
        <span className={`px-4 py-2 rounded-xl text-base font-semibold border-2 ${getQualityColor(quality)}`}>
          {getQualityIcon(quality)}
          {quality || 'UNKNOWN'}
        </span>
      </div>

      {/* Warning Messages */}
      {quality === 'POOR' && (
        <div className="mt-6 p-5 bg-red-50 border-2 border-red-200 rounded-xl">
          <div className="flex items-start gap-3">
            <svg className="w-6 h-6 text-red-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <p className="text-sm text-red-700 leading-relaxed">
              <strong>Warning:</strong> Low audio quality may result in poor voice model performance. Consider using a higher quality recording.
            </p>
          </div>
        </div>
      )}

      {quality === 'ACCEPTABLE' && (
        <div className="mt-6 p-5 bg-yellow-50 border-2 border-yellow-200 rounded-xl">
          <div className="flex items-start gap-3">
            <svg className="w-6 h-6 text-yellow-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <p className="text-sm text-yellow-700 leading-relaxed">
              <strong>Note:</strong> Audio quality is acceptable but could be better. Higher quality audio produces better results.
            </p>
          </div>
        </div>
      )}

      {(quality === 'GOOD' || quality === 'EXCELLENT') && (
        <div className="mt-6 p-5 bg-green-50 border-2 border-green-200 rounded-xl">
          <div className="flex items-start gap-3">
            <svg className="w-6 h-6 text-green-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            <p className="text-sm text-green-700 leading-relaxed">
              <strong>Great!</strong> Your audio quality is {quality.toLowerCase()} and perfect for training a high-quality voice model.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
