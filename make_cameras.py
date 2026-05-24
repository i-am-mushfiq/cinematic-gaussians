import os, json
from PIL import Image

# --- parameters to tweak ---
IMAGE_DIR    = r"Training_Data_v3/images"      # folder containing your PNG slices
OUTPUT_JSON  = r"Training_Data_v3/cameras.json" # where to write the camera file
SLICE_THICK  = 0.02525                              # slice spacing (mm)
ORTHO_FOVY   = 1e-4                               # tiny vertical FOV → orthographic
# ---------------------------

filenames = sorted(fn for fn in os.listdir(IMAGE_DIR) if fn.lower().endswith('.png'))
frames = []

for idx, fn in enumerate(filenames):
    img_path = os.path.join(IMAGE_DIR, fn)
    with Image.open(img_path) as im:
        w, h = im.size
        # composite away alpha if present
        if im.mode == 'RGBA':
            bg = Image.new('RGB', im.size, (0,0,0))
            bg.paste(im, mask=im.split()[3])
            bg.save(img_path)

    # camera position steps along Z by slice thickness
    pos = [0.0, 0.0, idx * SLICE_THICK]
    # identity rotation (camera looks down -Z axis)
    rot = [1,0,0,  0,1,0,  0,0,1]

    frames.append({
        "id":       idx,
        "fg_name":  fn,
        "width":    w,
        "height":   h,
        "fovy":     ORTHO_FOVY,
        "position": pos,
        "rotation": rot
    })

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(frames, f, indent=2)

print(f"Generated {len(frames)} orthographic cameras → {OUTPUT_JSON}")
