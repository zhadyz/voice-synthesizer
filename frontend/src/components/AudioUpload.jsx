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
        border-2 border-dashed rounded-lg p-12 text-center cursor-pointer
        transition-colors duration-200
        ${isDragActive ? 'border-primary bg-primary/5' : 'border-gray-300 hover:border-primary'}
      `}
    >
      <input {...getInputProps()} />
      <svg
        className="w-16 h-16 mx-auto text-gray-400 mb-4"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
        />
      </svg>
      <p className="text-lg font-medium text-gray-700">
        {label || 'Upload Audio File'}
      </p>
      <p className="text-sm text-gray-500 mt-2">
        Drag & drop or click to select
      </p>
      <p className="text-xs text-gray-400 mt-1">
        MP3, WAV, M4A, FLAC (max 100MB)
      </p>
    </div>
  );
}
