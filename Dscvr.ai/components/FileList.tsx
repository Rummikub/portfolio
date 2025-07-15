import React, { useState, useRef } from 'react';

interface FileListProps {
  isOpen: boolean;
  onClose: () => void;
}

const FileList: React.FC<FileListProps> = ({ isOpen, onClose }) => {
  const [files, setFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      const fileList = Array.from(event.target.files);
      setFiles(prevFiles => [...prevFiles, ...fileList]);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed left-16 top-0 h-screen w-72 bg-white border-r border-gray-200 shadow-lg z-50 flex flex-col">
      <div className="p-4 border-b border-gray-200">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Files</h2>
          <button
            onClick={onClose}
            className="text-gray-600 hover:text-gray-900"
          >
            ✕
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {files.length === 0 ? (
          <div className="text-gray-600 text-center">
            No files uploaded yet. Click the Upload button below to add files.
          </div>
        ) : (
          <div className="space-y-2">
            {files.map((file, index) => (
              <div
                key={index}
                className="flex items-center p-2 hover:bg-gray-100 rounded transition-colors group"
              >
                <span className="text-gray-600 mr-2">📄</span>
                <span className="text-gray-900 flex-1 truncate">{file.name}</span>
                <span className="text-sm text-gray-500">
                  {(file.size / 1024).toFixed(1)} KB
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="p-4 border-t border-gray-200 bg-gray-50">
        <button
          onClick={handleUploadClick}
          className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
        >
          <span>Upload Files</span>
        </button>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          className="hidden"
          accept=".txt,.doc,.docx,.pdf"
        />
      </div>
    </div>
  );
};

export default FileList;
