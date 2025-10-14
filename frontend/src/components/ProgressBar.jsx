import React from 'react';

export function ProgressBar({ progress, status, message }) {
  const getStatusColor = () => {
    switch (status) {
      case 'preprocessing':
      case 'pending':
        return 'bg-blue-500';
      case 'training':
      case 'processing':
        return 'bg-purple-500';
      case 'converting':
        return 'bg-green-500';
      case 'completed':
        return 'bg-green-600';
      case 'failed':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <div className="w-full">
      <div className="flex justify-between mb-2">
        <span className="text-sm font-medium text-gray-700 capitalize">
          {status || 'Processing'}
        </span>
        <span className="text-sm font-medium text-gray-700">
          {Math.round(progress)}%
        </span>
      </div>

      <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
        <div
          className={`h-full transition-all duration-500 ${getStatusColor()}`}
          style={{ width: `${progress}%` }}
        />
      </div>

      {message && (
        <p className="text-xs text-gray-500 mt-2">{message}</p>
      )}
    </div>
  );
}
