import { Node, mergeAttributes } from '@tiptap/core';
import { ReactNodeViewRenderer, NodeViewWrapper, NodeViewContent } from '@tiptap/react';

// React component for rendering insertion nodes
const InsertionComponent = ({ node, getPos, editor }) => {
  const acceptChange = () => {
    editor.commands.command(({ tr, dispatch }) => {
      if (dispatch) {
        tr.replaceRangeWith(getPos(), getPos() + node.nodeSize, node.content);
        dispatch(tr);
      }
      return true;
    });
  };

  const rejectChange = () => {
    editor.commands.deleteRange({ from: getPos(), to: getPos() + node.nodeSize });
  };

  return (
    <NodeViewWrapper as="span" className="diff-suggestion insertion group">
      <NodeViewContent as="span" className="diff-text" />
      <span className="diff-controls" contentEditable={false}>
        <button 
          onClick={acceptChange} 
          title="Accept insertion" 
          className="diff-button accept"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        </button>
        <button 
          onClick={rejectChange} 
          title="Reject insertion" 
          className="diff-button reject"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </span>
    </NodeViewWrapper>
  );
};

// TipTap extension for insertions
export const Insertion = Node.create({
  name: 'insertion',
  inline: true,
  group: 'inline',
  content: 'inline*',
  parseHTML() {
    return [{ tag: 'insertion' }];
  },
  renderHTML({ HTMLAttributes }) {
    return ['insertion', mergeAttributes(HTMLAttributes), 0];
  },
  addNodeView() {
    return ReactNodeViewRenderer(InsertionComponent);
  },
});
