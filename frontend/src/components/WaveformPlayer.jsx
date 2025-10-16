import React, { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';

export function WaveformPlayer({ audioUrl, title }) {
  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!audioUrl || !waveformRef.current) return;

    setIsLoading(true);

    // Create WaveSurfer instance
    wavesurfer.current = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#E5E7EB',
      progressColor: '#3B82F6',
      cursorColor: '#2563EB',
      barWidth: 3,
      barRadius: 3,
      responsive: true,
      height: 100,
      normalize: true,
    });

    // Load audio
    wavesurfer.current.load(audioUrl);

    // Event listeners
    wavesurfer.current.on('ready', () => {
      setIsLoading(false);
    });

    wavesurfer.current.on('finish', () => {
      setIsPlaying(false);
    });

    wavesurfer.current.on('error', (err) => {
      console.error('WaveSurfer error:', err);
      setIsLoading(false);
    });

    // Cleanup
    return () => {
      wavesurfer.current?.destroy();
    };
  }, [audioUrl]);

  const togglePlay = () => {
    if (wavesurfer.current) {
      wavesurfer.current.playPause();
      setIsPlaying(!isPlaying);
    }
  };

  return (
    <div className="bg-white rounded-2xl p-8 shadow-elevated hover:shadow-xl transition-shadow duration-300">
      <div className="flex items-center gap-5 mb-6">
        <button
          onClick={togglePlay}
          disabled={isLoading}
          className="w-16 h-16 flex items-center justify-center rounded-full bg-gradient-to-r from-accent-500 to-accent-600 text-white shadow-md hover:shadow-lg hover:scale-105 active:scale-95 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
        >
          {isLoading ? (
            <svg
              className="animate-spin h-7 w-7"
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
          ) : isPlaying ? (
            <svg
              className="w-7 h-7"
              fill="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
            </svg>
          ) : (
            <svg
              className="w-7 h-7 ml-1"
              fill="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>
        <span className="text-lg font-semibold text-gray-900">{title}</span>
      </div>
      <div ref={waveformRef} className="rounded-lg overflow-hidden" />
    </div>
  );
}
