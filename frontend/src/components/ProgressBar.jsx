import React from 'react';

export function ProgressBar({ progress, status, message }) {
  const getStatusGradient = () => {
    switch (status) {
      case 'preprocessing':
      case 'pending':
        return 'from-accent-500 to-accent-600';
      case 'training':
      case 'processing':
        return 'from-accent-500 to-accent-600';
      case 'converting':
        return 'from-accent-500 to-accent-600';
      case 'completed':
        return 'from-green-500 to-green-600';
      case 'failed':
        return 'from-red-500 to-red-600';
      default:
        return 'from-gray-500 to-gray-600';
    }
  };

  return (
    <div className="w-full">
      <div className="flex justify-between mb-3">
        <span className="text-base font-semibold text-gray-900 capitalize">
          {status || 'Processing'}
        </span>
        <span className="text-base font-semibold text-gray-900">
          {Math.round(progress)}%
        </span>
      </div>

      <div className="relative h-3 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`absolute inset-y-0 left-0 bg-gradient-to-r ${getStatusGradient()} rounded-full transition-all duration-300 ease-out`}
          style={{ width: `${progress}%` }}
        >
          {/* Shimmer overlay */}
          <div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer"
            style={{ backgroundSize: '1000px 100%' }}
          />
        </div>
      </div>

      {message && (
        <p className="text-sm text-gray-600 mt-3">{message}</p>
      )}
    </div>
  );
}
