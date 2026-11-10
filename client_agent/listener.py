# --- Core Telemetry & Networking ---
psutil>=5.9
requests>=2.31
aiohttp>=3.9

# --- OS-Specific Window Manager Hooks ---
# Windows: Win32 GUI handles
pywin32>=306; sys_platform == 'win32'

# macOS: AppKit & Cocoa Framework
pyobjc-framework-Cocoa>=10.0; sys_platform == 'darwin'

# Linux: X11 Display Server
python-xlib>=0.33; sys_platform == 'linux'