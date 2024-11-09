# Middle Button Click by Your Mouth

This project allows you to emulate a middle mouse button click by opening your mouth, using a web camera. It also enables 3-button mouse emulation based on face direction.

## Requirements
- Windows
- Python 3.10

## Setup
1. Clone the repository:
```sh
> git clone https://github.com/thecoolmuseum/trackface.git
> cd middle-button-click
```

2. Run setup.bat to create a virtual environment and install dependencies:
```
> setup.bat
```

## How to Start
1. Run run.bat to start the application:
```
> run.bat
```

## Key Commands
- Q: Quit the application
- C: Change the camera
- Space: Move the mouse based on face direction

## Additional Information
- Ensure your web camera is properly connected and recognized by the system.
- The application uses MediaPipe for face tracking and PyAutoGUI for mouse control.
