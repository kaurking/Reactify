# Reactify

Discord settings:
Input Device: CABLE Output / blackhole Output
Camera: OBS Virtual Camera
Noise suppression: Off
Echo cancellation: Off
Automatic input sensitivity: Off


Nessecary applications and plugins (Windows):
1. Install OBS Studio
    - install Spout plugin
    - install Audio monitor plugin (not mandatory to use the app)
2. Install VB audio virtual cable.

Nessecary applications and plugins (MAC silicon and intel):
1. Install OBS Studio
    - Install Syphon plugin
2. Install Audio MIDI Setup.
3. Install blackhole


Setup guide (Windows) NB For first version, will be updated
1. Install obs studio. Install Spout plugin and Audio monitor plugin. Install VB virtual cable and restart your PC.
2. open OBS. Create a new scene called "Reactify". Add a Spout source (it should automatically start getting the video feed from the app
when you run it). 
3. Add a media Source with a mp3 file you want to use as a media - this will be automatic later # TODO
4. in OBS setting set Audio monitoring device to VB cable Input.
5. Set under audio mixer mic and your added soundeffects to monitor and output (under gear icon)
6. (not mandatory) If you want to hear the soundeffects youself you have to add a Audio monitor filter to the added effects. With device as 
your headphones. 
7. Start virtual camera in obs.
8. In discord when inside a call apply the discord settings (subject to change) and set audio input as VB Output and camera to OBS virtual cam.
9. Run the app with running main.py and Start your camera. This was the final step for now. s

Setup guide (MAC silicon and intel)


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
                 Discord Cam  Mac -> Blackhole, MIDI | Windows -> VB cable, Audio monitor plugin
                               │
                               ▼
                         Discord Mic