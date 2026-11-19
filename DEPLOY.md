# Deploy (free) - Streamlit Community Cloud
1. Create a GitHub repo, push everything in this folder (app.py at the repo root).
   If you have trained weights, also push runs/road_damage/weights/best.pt (a yolov8n is ~6 MB).
2. Go to https://share.streamlit.io -> sign in with GitHub -> "Create app".
3. Pick the repo, branch main, main file = app.py -> Deploy.
4. You get a public link like https://<name>.streamlit.app - share that with sir.
Without best.pt the app automatically uses the Classical CV baseline.
