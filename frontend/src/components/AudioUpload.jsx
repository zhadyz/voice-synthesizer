import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

export function AudioUpload({ onUpload, accept = '.mp3,.wav,.m4a,.flac', label }) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onUpload(acceptedFiles[0]);
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'audio/*': ['.mp3', '.wav', '.m4a', '.flac'] },
    maxFiles: 1,
    maxSize: 100 * 1024 * 1024 // 100MB
  });

  return (
    <div
      {...getRootProps()}
      className={`
        border-2 border-dashed rounded-2xl p-16 text-center cursor-pointer
        transition-all duration-300 group
        ${isDragActive
          ? 'border-accent-500 bg-accent-500/10 scale-[1.02] shadow-elevated'
          : 'border-gray-300 hover:border-accent-500 hover:bg-accent-500/5 hover:shadow-elevated hover:scale-[1.01]'
        }
      `}
    >
      <input {...getInputProps()} />
      <div className={`transition-all duration-300 ${isDragActive ? 'scale-110' : 'group-hover:scale-105'}`}>
        <svg
          className={`w-20 h-20 mx-auto mb-6 transition-colors duration-300 ${
            isDragActive ? 'text-accent-500' : 'text-gray-400 group-hover:text-accent-500'
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
          />
        </svg>
      </div>
      <p className="text-xl font-semibold text-gray-900 mb-2">
        {label || 'Upload Audio File'}
      </p>
      <p className="text-base text-gray-600 mb-1">
        Drag & drop or click to select
      </p>
      <p className="text-sm text-gray-500">
        MP3, WAV, M4A, FLAC (max 100MB)
      </p>
    </div>
  );
}
