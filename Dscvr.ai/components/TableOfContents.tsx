import React, { useState, useEffect } from 'react';
import { Editor } from '@tiptap/react';

interface HeadingItem {
  id: string;
  text: string;
  level: number;
  position: number;
  number: string;
  isCollapsed: boolean;
  children: HeadingItem[];
}

interface TableOfContentsProps {
  isOpen: boolean;
  onClose: () => void;
  editor: Editor | null;
}

const TableOfContents: React.FC<TableOfContentsProps> = ({ isOpen, onClose, editor }) => {
  const [headings, setHeadings] = useState<HeadingItem[]>([]);
  const [showNumbers, setShowNumbers] = useState(true);

  const buildTableOfContents = () => {
    if (!editor) return [];

    const headings: HeadingItem[] = [];
    let currentNumber = [0];

    editor.state.doc.descendants((node, pos) => {
      if (node.type.name === 'heading') {
        const level = node.attrs.level;
        while (currentNumber.length < level) currentNumber.push(1);
        while (currentNumber.length > level) currentNumber.pop();
        currentNumber[currentNumber.length - 1]++;

        const id = `heading-${pos}`;
        const text = node.textContent;
        const number = currentNumber.join('.');

        headings.push({
          id,
          text,
          level,
          position: pos,
          number,
          isCollapsed: false,
          children: [],
        });
      }
    });

    // Build hierarchy
    const root: HeadingItem[] = [];
    const stack: HeadingItem[] = [];

    headings.forEach(heading => {
      while (stack.length > 0 && stack[stack.length - 1].level >= heading.level) {
        stack.pop();
      }

      if (stack.length === 0) {
        root.push(heading);
      } else {
        stack[stack.length - 1].children.push(heading);
      }

      stack.push(heading);
    });

    return root;
  };

  useEffect(() => {
    if (editor) {
      const updateHeadings = () => {
        setHeadings(buildTableOfContents());
      };

      updateHeadings();
      editor.on('update', updateHeadings);

      return () => {
        editor.off('update', updateHeadings);
      };
    }
  }, [editor]);

  const handleHeadingClick = (position: number) => {
    if (editor) {
      editor.commands.setTextSelection(position);
      editor.commands.scrollIntoView();
    }
  };

  const toggleCollapse = (id: string) => {
    setHeadings(prevHeadings => {
      const toggleHeading = (items: HeadingItem[]): HeadingItem[] => {
        return items.map(item => {
          if (item.id === id) {
            return { ...item, isCollapsed: !item.isCollapsed };
          }
          if (item.children.length > 0) {
            return { ...item, children: toggleHeading(item.children) };
          }
          return item;
        });
      };

      return toggleHeading(prevHeadings);
    });
  };

  const renderHeading = (heading: HeadingItem, level: number = 0) => {
    const hasChildren = heading.children.length > 0;
    
    return (
      <div key={heading.id}>
        <div
          className="flex items-center hover:bg-gray-100 rounded p-2 transition-colors group"
          style={{
            paddingLeft: `${(heading.level - 1) * 24}px`,
            cursor: 'pointer'
          }}
          onClick={() => handleHeadingClick(heading.position)}
        >
          {hasChildren && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                toggleCollapse(heading.id);
              }}
              className="w-6 h-6 flex items-center justify-center text-gray-600 hover:text-gray-800"
            >
              {heading.isCollapsed ? '▶' : '▼'}
            </button>
          )}
          {!hasChildren && <div className="w-6" />}
          <div className="flex-1 flex items-center">
            {showNumbers && heading.number && (
              <span className="text-gray-600 mr-2 min-w-[32px]">{heading.number}</span>
            )}
            <span className={`text-gray-900 hover:text-black ${
              heading.level === 1 ? 'font-semibold text-base' :
              heading.level === 2 ? 'font-medium' : 'font-normal text-sm'
            }`}>
              {heading.text}
            </span>
          </div>
        </div>
        {!heading.isCollapsed && heading.children.length > 0 && (
          <div>
            {heading.children.map(child => renderHeading(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  if (!isOpen) return null;

  return (
    <div className="fixed left-16 top-0 h-screen w-72 bg-white border-r border-gray-200 shadow-lg z-50 flex flex-col">
      <div className="p-4 border-b border-gray-200">
        <div className="flex justify-between items-center">
          <h2 className="text-lg font-semibold text-gray-900">Table of Contents</h2>
          <button
            onClick={onClose}
            className="text-gray-600 hover:text-gray-900"
          >
            ✕
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {headings.length === 0 ? (
          <div className="text-gray-600 text-center">
            No headings found. Add headings to your document to create a table of contents.
          </div>
        ) : (
          <div className="space-y-1">
            {headings.map(heading => renderHeading(heading))}
          </div>
        )}
      </div>

      <div className="p-4 border-t border-gray-200 bg-gray-50">
        <div className="flex items-center justify-between">
          <label className="flex items-center space-x-2 text-sm text-gray-800">
            <input
              type="checkbox"
              checked={showNumbers}
              onChange={(e) => setShowNumbers(e.target.checked)}
              className="rounded text-blue-500 focus:ring-blue-500"
            />
            <span>Show numbers</span>
          </label>
          <span className="text-sm text-gray-600">Click to navigate</span>
        </div>
      </div>
    </div>
  );
};

export default TableOfContents;
