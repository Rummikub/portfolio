'use client';

import { useState, useRef } from 'react';
import TypingAnimation from './TypingAnimation';
import LoadingDots from './LoadingDots';

const mockSuggestions = [
  {
    id: 1,
    type: 'reference',
    title: 'Relevant Paper',
    content: 'Smith et al. (2023) discusses similar concepts in "AI-Powered Research Environments"',
  },
  {
    id: 2,
    type: 'improvement',
    title: 'Style Suggestion',
    content: 'Consider breaking this paragraph into smaller sections for better readability',
  },
  {
    id: 3,
    type: 'outline',
    title: 'Structure Recommendation',
    content: 'Add a methodology section here to explain your research approach',
  },
];

const mockEditorContent = `<h1>Sample Content from AI Sidebar</h1>
<p>This is a sample text that demonstrates how we can pass content from the AI Sidebar to the Editor component. It includes various formatting like <strong>bold</strong> and <em>italic</em> text.</p>
<p>We can also include multiple paragraphs and different types of content to show the flexibility of our editor.</p>`;

const mockReferences = [
  {
    id: 1,
    title: 'Understanding AI',
    content: 'A comprehensive guide to AI technologies and their applications.',
  },
  {
    id: 2,
    title: 'Machine Learning Basics',
    content: 'An introduction to machine learning concepts and techniques.',
  },
  {
    id: 3,
    title: 'Deep Learning Explained',
    content: 'A detailed overview of deep learning and its impact on AI.',
  },
];

interface AISidebarProps {
  onUpdateContent: (content: string) => void;
}

export default function AISidebar({ onUpdateContent }: AISidebarProps) {
  const [activeTab, setActiveTab] = useState('suggestions');
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [responseText, setResponseText] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSend = () => {
    if (!inputValue.trim() || isTyping) return;
    
    // Pass mock content to editor when sending a message
 

    // Mock response for demonstration
    const mockResponse = "I'll analyze relevant papers and suggest improvements to enhance your document. This might take a moment...";
    
    setIsLoading(true);
    setInputValue('');
    
    // Delay before starting to type
    setTimeout(() => {
      setIsLoading(false);
      setIsTyping(true);
      setResponseText(mockResponse);
    }, 1500); // 1.5 second delay
    
    // Focus back on input after sending
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Sidebar Header */}
      <div className="h-14 border-b flex items-center justify-between px-4 bg-white">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 bg-blue-600 rounded flex items-center justify-center">
            <span className="text-white text-xs">AI</span>
          </div>
          <h2 className="text-sm font-medium text-gray-900">AI Assistant</h2>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-2 h-2 bg-green-500 rounded-full"></span>
          <span className="text-xs text-gray-500">Active</span>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex px-1 pt-2 pb-1 gap-1 bg-white border-b">
        <button
          className={`flex-1 px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
            activeTab === 'suggestions'
              ? 'bg-blue-50 text-blue-700'
              : 'text-gray-600 hover:bg-gray-50'
          }`}
          onClick={() => setActiveTab('suggestions')}
        >
          Suggestions
        </button>
        <button
          className={`flex-1 px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
            activeTab === 'references'
              ? 'bg-blue-50 text-blue-700'
              : 'text-gray-600 hover:bg-gray-50'
          }`}
          onClick={() => setActiveTab('references')}
        >
          References
        </button>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-auto p-3 space-y-3">
        {activeTab === 'suggestions' ? (
          mockSuggestions.map((suggestion) => (
            <div
              key={suggestion.id}
              className="p-3 bg-white rounded-lg border border-gray-100 hover:border-blue-100 transition-colors"
            >
              <div className="flex items-center gap-2 mb-2">
                <div className="w-4 h-4 rounded bg-blue-100 flex items-center justify-center">
                  <span className="text-blue-600 text-xs">✓</span>
                </div>
                <h3 className="text-xs font-medium text-gray-900">
                  {suggestion.title}
                </h3>
              </div>
              <p className="text-xs text-gray-600 leading-relaxed pl-6">{suggestion.content}</p>
            </div>
          ))
        ) : (
          mockReferences.map((reference) => (
            <div
              key={reference.id}
              className="p-3 bg-white rounded-lg border border-gray-100 hover:border-blue-100 transition-colors"
            >
              <div className="flex items-center gap-2 mb-2">
                <h3 className="text-xs font-medium text-gray-900">
                  {reference.title}
                </h3>
              </div>
              <p className="text-xs text-gray-600 leading-relaxed pl-6">{reference.content}</p>
            </div>
          ))
        )}
      </div>


      {/* Input Area */}
      {/* Response Area */}
      {(isLoading || isTyping) && (
        <div className="p-3 border-t bg-gray-50">
          <div className="flex items-start gap-2">
            <div className="w-6 h-6 bg-blue-600 rounded flex-shrink-0 flex items-center justify-center mt-0.5">
              <span className="text-white text-xs">AI</span>
            </div>
            <div className="flex-1">
              {isLoading ? (
                <LoadingDots />
              ) : (
                <TypingAnimation
                  text={responseText}
                  className="text-sm text-gray-800 leading-relaxed"
                  onComplete={() => {
                    setIsTyping(false);
                    onUpdateContent(mockEditorContent);
                  }}  
                />
              )}
            </div>
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="p-3 border-t bg-white">
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Ask anything..."
              className="w-full pl-3 pr-8 py-1.5 text-sm border border-gray-200 rounded-md text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
            />
            <button className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600">
              🎤
            </button>
          </div>
          <button 
            onClick={handleSend}
            disabled={isLoading || isTyping || !inputValue.trim()}
            className="px-3 py-1.5 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
