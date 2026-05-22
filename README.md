# Reactify

Nessecary applications and plugins (Windows):
1. Install OBS Studio
    - install Spout plugin
    - install Audio monitor plugin (not mandatory to use the app)
2. Install VB audio virtual cable.

Nessecary applications and plugins (MAC silicon and intel):
1. Install OBS Studio
2. Install Audio MIDI Setup (default installed on macs usually)
3. Install blackhole

#### Discord settings:
Input Device: VB CABLE Output / blackhole Output
Camera: OBS Virtual Camera
Noise suppression: Off
Echo cancellation: Off
Automatic input sensitivity: Off

### Setup guide (Windows) NB For first version, will be updated
1. Install OBS studio. Install Spout plugin and Audio monitor plugin. Install VB virtual cable and restart your PC.
2. in OBS settings set Audio monitoring device to VB cable Input.
3. Enable OBS WebSocket server. Go Tools -> WebSocket Server setting -> enable. keep port default. TODO: auth should be enabled but works well without
4. open OBS. Create a new scene called "Reactify".
5. Add a media Source to Reactify scene and call it exactly "ReactifySound". Add a Spout source. Keep settings default.
6. Under audio mixer set "Mic/Aux" and "ReactifySound" to monitor and output (under gear icon).
7. (not mandatory) If you want to hear the soundeffects youself you have to add a Audio monitor filter to the added effects. Add device as 
your headphones. 
8. Start virtual camera in OBS.
9. In discord when inside a call apply the discord settings and set audio input as VB Output and camera to OBS virtual camera.
10. Run the app with running main.py and Start your camera. This was the final step for now.

### Setup guide (MAC silicon and intel)
1. Install OBS studio. Install blackhole and restart your PC.
2. In Audio MIDI setup app create a new multi-output device, call it "Reactify output". Enable blackhole and your headphone/speaker. Set the latter as primary.
3. in OBS settings set Audio monitoring device to "Reactify output".
4. Enable OBS WebSocket server. Go Tools -> WebSocket Server setting -> enable. keep port default. TODO: auth should be enabled but works well without
5. Open OBS. On the left create a scene called "Reactify". 
6. To the Reactify scene add a image source called exactly "ReactifyOverlay" and Media source called exactly "ReactifySound". Leave both with default settings.
7. Under audio mixer set "Mic/Aux" and "ReactifySound" to monitor and output (under gear icon). TODO: you will hear yourself (will be fixed)
8. In discord when inside a call apply the discord settings and set audio input as blackhole and camera to OBS virtual camera.
9. Run the app with running main.py and Start your camera. This was the final step for now.

### App use guide:
To run the app OBS must be running and set up before according to the previous guide. To run the app, run main.py. 
There are 2 tabs. Camera tab and emote profiles tab. To addd an emote, klick New Emote and start filling in the nessecary parameters at the
top. choose parameters and add a picture as png or a gif and add an mp3 sound. Once all is filled, klick Start Guided Sampling and take the 
pose you want the uploded image and sound to react to. Once that is done, klick Save profiles. Klick Reload Profiles and a new emote has
been added. It is not reccomennded to add emotes with similar body positions as it will detect them both at the same time. It is also not
reccommended to add more than 5 - 10 emotes at the same itme. This figure is based on the body positions though. 


For .md source viewing only:

Pipeline
                ┌-────────────────────┐
                │     Reactify        │
                │                     │
Webcam ───────► │ MediaPipe detector  │
                │ Gesture matcher     │
                │ Overlay renderer    │
                └───────┬────────────-┘
                        │
             video frame via Spout
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
