# Frontend Modification Plan for Steam Deck

This document outlines the plan for modifying the ComfyUI frontend to be a first-class citizen on the Steam Deck.

## 1. Frontend Build & Development Environment

The frontend is a modern Vue.js application built with Vite and managed with pnpm.

*   **Package Manager**: `pnpm` must be used. The repository enforces this with a `preinstall` script.
*   **Build Command**: The production build is created by running `pnpm build`.
*   **Development Server**: The development server is started with `pnpm dev`.
*   **Core Dependencies**: The key libraries are Vue 3, Pinia for state management, Vue Router for navigation, and PrimeVue with TailwindCSS for the UI components.

## 2. Forking and Build Process

Since the frontend is a separate project and we cannot build it in the current environment, the plan is as follows:

1.  **Fork the Repository**: A fork of the `ComfyUI_frontend` repository will be created under the same organization or user that owns the main `ComfyUI` fork.
2.  **Clone the Fork**: The forked repository will be cloned locally.
3.  **Install Dependencies**: `pnpm install` will be run to install all necessary dependencies.
4.  **Build the Frontend**: `pnpm build` will be run to create the distributable frontend files.
5.  **Replace the Pre-packaged Frontend**: The newly built frontend files will be used to replace the pre-packaged `comfyui-frontend-package` in the main ComfyUI repository. This will likely involve creating a new `web/` directory in the ComfyUI repo and copying the built files there, then modifying `server.py` to serve files from this new directory instead of the package.

## 3. UI/UX Overhaul for Steam Deck

The following changes will be made to the frontend codebase to optimize for the Steam Deck's 1280x800 touchscreen and trackpad interface.

### 3.1. Global Changes

*   **Increase Font Size**: The base font size will be increased globally for better readability.
*   **Increase Touch Target Sizes**: All interactive elements (buttons, sliders, menus, node handles) will be enlarged to be easily tappable. A minimum touch target size of 44x44 CSS pixels will be enforced.
*   **Remove Hover-based Interactions**: All functionality that relies on mouse hover events will be re-implemented to work with touch events (e.g., tap-and-hold, or a dedicated button).
*   **High-Contrast Theme**: A high-contrast theme will be implemented as the default to improve visibility in various lighting conditions.

### 3.2. Specific Component Changes

| Component/Area      | Problem                                                                  | Proposed Solution                                                                                                   |
| ------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| **Main Menu**       | Menu items are small and close together.                                 | Redesign as a full-screen or large modal overlay with large, clearly labeled buttons for each function.             |
| **Node Graph**      | Nodes and their connection points (dots) are too small. Panning and zooming can be finicky with a trackpad. | Increase the size of nodes and their connection points. Implement more robust touch-based panning (two-finger drag) and zooming (pinch-to-zoom). |
| **Node Widgets**    | Sliders, text inputs, and dropdowns within nodes are tiny.               | Replace with larger, touch-friendly custom components. Sliders will have larger handles. Text inputs will be taller. |
| **Context Menu**    | Right-click context menu is difficult to use with a trackpad.            | Replace with a "long-press" or "tap-and-hold" gesture that brings up a large, finger-friendly radial menu or modal dialog. |
| **File/Workflow Dialogs** | Standard file open/save dialogs are not touch-friendly.                  | Implement custom, full-screen file browser components with large icons and clear navigation.                        |
| **Keyboard Shortcuts** | Core functions like "Queue Prompt" (Ctrl+Enter) and "Save" (Ctrl+S) are keyboard-only. | Add large, persistent, on-screen buttons for all core actions to the main toolbar. The toolbar will be permanently visible at the top or bottom of the screen. |
| **Search Box**      | The node search box is small and relies on keyboard input.               | Enlarge the search box and improve the on-screen keyboard experience if possible. Consider a more visual, category-based node browser as an alternative. |
| **History/Queue Panel** | The side panel for history and queue is narrow and items are small.   | Widen the panel by default. Increase the size of each item in the list and make the "Re-run" and "Delete" buttons larger. |

## 4. Implementation and Testing

*   **Branching**: All frontend changes will be made on a `steamdeck-native` branch in the forked frontend repository.
*   **Component-by-Component Refactoring**: The UI will be refactored one component at a time, following the plan above.
*   **Testing**: After each major change, the frontend will be built and tested with the refactored ComfyUI backend to ensure everything works as expected. The Playwright test suite will be updated to reflect the new UI and ensure no regressions have been introduced.
