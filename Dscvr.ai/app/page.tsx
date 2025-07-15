"use client";

import dynamic from 'next/dynamic';
import { useState } from 'react';

const Editor = dynamic(() => import('../components/Editor'), {
  ssr: false,
});

const AISidebar = dynamic(() => import('../components/AISidebar'), {
  ssr: false,
});

const FileList = dynamic(() => import('../components/FileList'), {
  ssr: false,
});

const TableOfContents = dynamic(() => import('../components/TableOfContents'), {
  ssr: false,
});

export default function Home() {
  const [editorContent, setEditorContent] = useState('');
  const [isFileListOpen, setIsFileListOpen] = useState(false);
  const [isTocOpen, setIsTocOpen] = useState(false);
  const [editor, setEditor] = useState<any>(null);

  const handleUpdateContent = (content: string) => {
    setEditorContent(content);
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Navigation Sidebar */}
      <nav className="w-16 bg-white border-r border-gray-200 flex flex-col items-center py-4 gap-4">
        <div className="w-8 h-8 bg-gray-800 rounded-full flex items-center justify-center">
          <span className="text-white text-sm">📝</span>
        </div>
        <div 
          className="w-8 h-8 rounded-full flex items-center justify-center hover:bg-gray-100 cursor-pointer"
          onClick={() => setIsTocOpen(!isTocOpen)}
        >
          <span className="text-gray-600">📑</span>
        </div>
        <div 
          className="w-8 h-8 rounded-full flex items-center justify-center hover:bg-gray-100 cursor-pointer"
          onClick={() => setIsFileListOpen(!isFileListOpen)}
        >
          <span className="text-gray-600">📁</span>
        </div>
      </nav>

      {/* Table of Contents */}
      <TableOfContents
        isOpen={isTocOpen}
        onClose={() => setIsTocOpen(false)}
        editor={editor}
      />

      {/* File List */}
      <FileList
        isOpen={isFileListOpen}
        onClose={() => setIsFileListOpen(false)}
      />

      {/* Main content area */}
      <main className="flex-1 flex flex-col h-full bg-white">
        {/* Top toolbar */}
        <div className="h-14 border-b flex items-center justify-between px-4 bg-white">
          <div className="flex items-center gap-4">
            <h1 className="text-lg font-semibold text-gray-800">Research Paper</h1>
            <div className="h-6 w-px bg-gray-200"></div>
            <div className="text-sm text-gray-500">Last edited 2 mins ago</div>
          </div>
          <div className="flex items-center gap-2">
            <button className="px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-100 rounded-md transition-colors">
              Share
            </button>
            <button className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors">
              Save
            </button>
          </div>
        </div>
        
        {/* Editor area */}
        <div className="flex-1 overflow-auto p-8 max-w-4xl mx-auto w-full">
          <Editor
            content={editorContent}
            onInit={setEditor}
            shouldUpdate={false}
            onUpdateComplete={() => {}}
          />
        </div>
      </main>

      {/* AI Sidebar */}
      <aside className="w-80 border-l border-gray-200 bg-white">
        <AISidebar onUpdateContent={handleUpdateContent} />
      </aside>
    </div>
  );
}
