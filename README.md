# Reactify

Nessecary applications and plugins (Windows):
1. Install OBS Studio
    - install Spout plugin
    - install Audio monitor plugin
2. Install VB audio virtual cable.

Nessecary applications and plugins (MAC silicon):
1. Install OBS Studio
3. Install blackhole2ch

#### Discord settings:
Input Device: VB CABLE Output / blackhole Output <br>
Camera: OBS Virtual Camera <br>
Noise suppression: Off <br>
Echo cancellation: Off <br>
Automatic input sensitivity: Off 

### Setup guide (Windows)
1. In github under Releases install the compressed file for Windows and unzip it. It should contain the app, assets, models, internal folders and .env file. NB run Reactify.exe only at the end of the guide.
2. Install OBS studio. Install Spout plugin and Audio monitor plugin. Install VB virtual cable and restart your PC.
3. in OBS settings set Audio monitoring device to VB cable Input.
4. Enable OBS WebSocket server. Go Tools -> WebSocket Server setting -> enable. keep port default.
5. Inside the `.env` set `OBS_PASSWORD` to the OBS WebSocket server password (make sure the format stays the same) from Tools -> WebSocket Server Settings.
6. open OBS. Create a new scene called "Reactify".
7. Add a media Source to Reactify scene and call it exactly "ReactifySound". Add a Spout source. Keep settings default.
8. Under audio mixer set "Mic/Aux" and "ReactifySound" to monitor and output (under gear icon).
9. (not mandatory) If you want to hear the soundeffects youself you have to add a Audio monitor filter to ReactifySound. Add device as
your headphones. 
10. Start virtual camera in OBS.
11. In discord when inside a call apply the discord settings and set audio input as VB Output and camera to OBS virtual camera.
12. Run OBS and Discord with the necessary settings and only then run Reactify.app

### Setup guide (MAC silicon)
1. In github under Releases install the compressed file for MacOS and unzip it. It should contain the app, assets, models folders and .env file. NB run Reactify.app only at the end of the guide.
2. Install OBS studio. Install blackhole and restart your PC.
3. In OBS settings set Audio monitoring device to "blackhole".
4. Enable OBS WebSocket server. Go Tools -> WebSocket Server setting -> enable. keep port default.
5. Inside the `.env` set `OBS_PASSWORD` to the OBS WebSocket server password (make sure the format stays the same) from Tools -> WebSocket Server Settings.
6. Open OBS. On the left create a scene called "Reactify".
7. To the Reactify scene add a image source called exactly "ReactifyOverlay" and Media source called exactly "ReactifySound". Leave both with default settings.
8. Under audio mixer set "Mic/Aux" and "ReactifySound" to monitor and output (under gear icon).
9. In discord when inside a call apply the discord settings and set audio input as blackhole and camera to OBS virtual camera.
10. Run OBS and Discord with the necessary settings and only then run Reactify.exe

## App use guide:
To run the app OBS must be running and set up before according to the previous guide. To run the app, run Reactify.exe. <br>
There are 2 tabs. Camera tab and emote profiles tab. To addd an emote, klick New Emote and start filling in the nessecary parameters at the
top. choose parameters and add a picture as png or a gif and add an mp3 sound. <br> Once all is filled, klick Start Guided Sampling and take the 
pose you want the uploded image and sound to react to. <br> Once that is done, klick Save profiles. Klick Reload Profiles and a new emote has
been added. <br> It is not reccomennded to add emotes with very similar body positions as it will detect them both at the same time. It is also not
reccommended to add more than 5 - 10 emotes at the same itme. This figure is based on the body positions though. 


#### Pipeline (Windows): 
Webcam -> Reacify (MediaPipe detecor, Gesture matcher, Overlay renderer) -> via Spout to -> OBS (Spout source, Audio source) -> virtual camera to discord / audio via VB cable and OBS Audio monitor plugin to discord. 

#### Pipeline (MacOS):
Webcam -> Reacify (MediaPipe detecor, Gesture matcher, Overlay renderer) and OBS (Audio source) -> virtual camera to discord / audio via blackhole to discord. 

The difference in MacOS is that the drivers of mac cameras allow multiple sources to own the camera at the same time (Reactify and OBS). In Windows camera feed is sent via Spout to OBS.
