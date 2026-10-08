"""
Unit tests for the Floating Draggable Researcher Scratchpad widget.
Verifies that render_floating_scratchpad:
1. Generates complete HTML/CSS/JavaScript with drag-and-drop capability.
2. Supports mouse and touch events for free movement anywhere on the screen.
3. Includes window boundary constraints and position/dimension caching.
4. Provides instant client-side downloads as Markdown (.md) and plain text (.txt).
5. Supports minimize/restore toggling and note clearing/synchronization.
"""

import unittest
from unittest.mock import patch
import json
from src.ui_components import render_floating_scratchpad


class TestFloatingScratchpad(unittest.TestCase):

    @patch("streamlit.components.v1.html")
    def test_render_floating_scratchpad_structure(self, mock_components_html):
        """Test that render_floating_scratchpad invokes components.html with complete widget code."""
        render_floating_scratchpad(new_pinned="Test note citation", reset_pos=False, clear_notes=False)
        self.assertTrue(mock_components_html.called)
        
        call_args = mock_components_html.call_args[0]
        html_code = call_args[0]
        
        # Verify DOM elements
        self.assertIn("chem-floating-scratchpad-root", html_code)
        self.assertIn("chem-scratchpad-header", html_code)
        self.assertIn("chem-scratchpad-textarea", html_code)
        self.assertIn("chem-scratchpad-counts", html_code)
        self.assertIn("chem-pad-min-btn", html_code)
        
        # Verify drag & drop handlers (both mouse and touch)
        self.assertIn("mousedown", html_code)
        self.assertIn("mousemove", html_code)
        self.assertIn("mouseup", html_code)
        self.assertIn("touchstart", html_code)
        self.assertIn("touchmove", html_code)
        self.assertIn("touchend", html_code)
        self.assertIn("cursor: grab", html_code)
        self.assertIn("cursor: grabbing", html_code)
        
        # Verify position and dimension persistence
        self.assertIn("chem_scratchpad_pos", html_code)
        self.assertIn("chem_scratchpad_dim", html_code)
        self.assertIn("chem_floating_notes", html_code)
        self.assertIn("chem_scratchpad_minimized", html_code)
        
        # Verify client-side downloads
        self.assertIn("chem-pad-dl-md", html_code)
        self.assertIn("chem-pad-dl-txt", html_code)
        self.assertIn(".md", html_code)
        self.assertIn(".txt", html_code)
        self.assertIn("Blob", html_code)
        
        # Verify tools: timestamp, copy, clear
        self.assertIn("chem-pad-time", html_code)
        self.assertIn("chem-pad-copy", html_code)
        self.assertIn("chem-pad-dock", html_code)
        self.assertIn("chem-pad-clear", html_code)

    @patch("streamlit.components.v1.html")
    def test_render_floating_scratchpad_flags(self, mock_components_html):
        """Test that reset_pos and clear_notes flags are correctly injected into JavaScript."""
        render_floating_scratchpad(new_pinned="New Pin", reset_pos=True, clear_notes=True)
        html_code = mock_components_html.call_args[0][0]
        
        self.assertIn("const ResetPos = true;", html_code)
        self.assertIn("const ClearNotes = true;", html_code)
        self.assertIn('"New Pin"', html_code)


if __name__ == "__main__":
    unittest.main()
