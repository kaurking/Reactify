# Reactify

Discord settings:
Input Device: CABLE Output / Voicemeeter Output
Camera: OBS Virtual Camera
Noise suppression: Off
Echo cancellation: Off
Automatic input sensitivity: Off

                ┌-────────────────────┐
                │     Reactify        │
                │                     │
Webcam ───────► │ MediaPipe detector  │
                │ Gesture matcher     │
                │ Overlay renderer    │
                └───────┬────────────-┘
                        │
             video frame via Spout/Syphon
                        │
                        ▼
                ┌────────────────────┐
                │        OBS         │
                │ Spout/Syphon source│
                │ Audio media source │
                │ Mic source         │
                └───────┬──────┬─────┘
                        │      │
             virtual cam│      │audio mix
                        ▼      ▼
                 Discord Cam  BlackHole
                               │
                               ▼
                         Discord Mic

Lisada MIDI MultiOutput device, et kasutaja ise ka kuuleks meme sound effekti, mitte ainult ei edastataks seda discord?